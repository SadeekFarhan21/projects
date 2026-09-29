# kvdb design

kvdb is a single-node, single-threaded, persistent key-value store. Keys and values are byte strings. The store is a B+ tree of 4 KiB pages in one file, cached by a buffer pool, made durable by a logical write-ahead log and made crash-atomic at checkpoints by a page journal.

This document describes v0 as built. The code is the source of truth; file references point at it.

## Files on disk

A database at path `P` is three files:

| File | Contents | Normal size |
|---|---|---|
| `P` | Page file: page 0 is the meta page, every other page is a B+ tree node | grows by 4 KiB per allocated page |
| `P-wal` | Redo log: one record per committed `WriteBatch` since the last checkpoint | 0 to `wal_checkpoint_bytes` (64 MiB default), preallocated in 4 MiB steps |
| `P-journal` | Copy of every page a checkpoint is about to overwrite | 0 bytes except while a checkpoint runs |

## Architecture

```
                       DB::put / del / write(batch)          DB::get / scan
                                    |                              |
                                    v                              |
   +---------------------------------------------+                 |
   | DB (src/db.cc)                              |                 |
   |  1. encode batch (WriteBatch rep)           |                 |
   |  2. Wal::append + sync   <-- commit point   |                 |
   |  3. apply ops to BTree                      |                 |
   |  4. maybe checkpoint (pool or WAL too full) |                 |
   +------+-------------------------+------------+                 |
          |                         |                              |
          v                         v                              v
   +--------------+         +-------------------------------------------+
   | Wal          |         | BTree (src/btree.cc)                      |
   | (src/wal.cc) |         |  descend root -> leaf, split on overflow, |
   | CRC-framed   |         |  lazy delete, leaf chain for range scans  |
   | records      |         +---------------------+---------------------+
   +------+-------+                               | fetch/create/unpin (PageGuard)
          |                                       v
          |                 +-------------------------------------------+
          |                 | BufferPool (src/buffer_pool.cc)           |
          |                 |  N frames of 4 KiB, page table, pin       |
          |                 |  counts, LRU list of clean unpinned       |
          |                 |  frames. NO-STEAL: dirty pages stay put.  |
          |                 +---------+-------------------+-------------+
          |                           | read on miss      | dirty pages at checkpoint
          |                           v                   v
          |                 +-----------------+   +------------------------+
          |                 | Pager           |<--| checkpoint (db.cc)     |
          |                 | pread/pwrite of |   |  a. pages -> journal,  |
          |                 | whole pages     |   |     header last, sync  |
          |                 +--------+--------+   |  b. pages -> in place, |
          |                          |            |     sync               |
          v                          v            |  c. truncate WAL and   |
     [ P-wal ]                    [ P ]  <--------+     journal            |
                                                  +-----------+------------+
                                                              |
                                                              v
                                                        [ P-journal ]

   Recovery at open:  P-journal (complete? replay pages : discard)
                   -> load meta from P (checkpoint LSN)
                   -> replay P-wal records with LSN > checkpoint LSN
                   -> checkpoint, so the WAL starts empty
```

## Key data structures

### Meta page (page 0)

Fixed offsets (`include/kvdb/btree.h`, `struct Meta`): magic `!KVVDB01` (u64), format version (u32), root page id (u32), page count (u32, next page to allocate), checkpoint LSN (u64), tree height (u32). The meta page lives in the buffer pool like any node, so root changes and allocation are checkpointed atomically with the nodes they refer to.

### Slotted node page

```
 0      1      2        4        8           10     12      16
 +------+------+--------+--------+-----------+------+-------+
 | type | rsvd | nslots |  link  | cellStart | frag | rsvd  |
 +------+------+--------+--------+-----------+------+-------+
 | slot[0] slot[1] ...  (u16 offsets, sorted by key)  -->   |
 |                     free space                           |
 |          <-- cells (klen u16, vlen u16, key, value)      |
 +----------------------------------------------------------+
```

Slots grow down from the header and cells grow up from the end of the page. `frag` counts bytes of dead cells left by erase or by an update that moved a cell; when an insert does not fit in the contiguous gap but would fit after reclaiming `frag`, the page is compacted in place.

A leaf's `link` is its right sibling (0 means none). An internal node's `link` is its leftmost child; each cell's value is a 4-byte child id, and child `i` holds keys greater than or equal to `key[i]`.

Keys are limited to 128 bytes and values to 512 bytes. A worst-case cell plus its slot is 646 bytes, so any overflowing node holds at least 6 cells and the byte-balanced split always produces two halves that fit. This limit is what lets v0 skip overflow pages.

### WAL record

```
 u32 crc32 | u32 len | u64 lsn | payload (len bytes, a WriteBatch encoding)
```

The CRC covers `len`, `lsn` and the payload. Replay stops at the first record whose length runs past the end of the file, whose CRC fails, or whose LSN is not larger than the previous one, and truncates the file there. The zero-filled preallocated tail fails the length or CRC test the same way a torn write does.

### WriteBatch

`u32 count` followed by `u8 kind, u16 klen, u16 vlen, key, value` per op. The same bytes are the WAL payload, so logging a batch is one copy and one `pwrite`.

### Checkpoint journal

```
 u64 magic | u32 page count | u32 crc32(body) | body: (u32 page id, 4096 bytes) * count
```

The body is written first and the header last. The journal is valid only if the magic matches, the file size matches the count exactly and the CRC over the whole body matches, so a journal that was cut short at any byte is detected even if the header reached the disk before the body did.

### Buffer pool frames

A single `aligned_alloc` block of `capacity * 4096` bytes, a `std::vector<Frame>` of `{page_id, pin_count, dirty, lru iterator}`, an `unordered_map<PageId, frame>` page table, a free list, and a `std::list` LRU of frames that are both unpinned and clean. `PageGuard` is an RAII pin that unpins on destruction and carries the dirty bit.

## Operations

### Write path

`DB::write(batch)` first checks whether a checkpoint is due, then assigns the next LSN, appends the record to the WAL and syncs it according to `SyncMode`. Once the append returns, the batch is committed. It is then applied to the tree in memory. `put` and `del` are one-op batches; `del` of a missing key is not logged.

### B+ tree insert

Descend from the root recording the internal path. At the leaf, update in place if the key exists and the new value fits, otherwise insert the cell. If it does not fit, collect all entries plus the new one and split:

1. If the new key is the last entry of the rightmost leaf (ascending inserts), keep the old leaf full and start a fresh right leaf. This gives about 100% fill for sequential loads instead of 50%.
2. Otherwise split at the byte midpoint, not the count midpoint, since cells vary in size.

The first key of the right half is pushed into the parent. Internal nodes split the same way by bytes, and the middle key moves up (it is not copied). A root split allocates a new root and bumps the height in the meta page.

### Delete

Deletes are lazy: the cell is erased from its leaf and nothing else happens. Nodes never merge or borrow, empty leaves stay in the chain, and pages are never freed. Lookups and scans remain correct because separators still bound their subtrees.

### Range scan

Descend to the leaf that would hold `lo`, then walk slots and follow leaf `link` pointers until a key is `>= hi` or the callback returns false. Only one page is pinned at a time.

### Checkpoint

1. Stamp the meta page with `applied_lsn` (every batch at or below it is fully in the tree).
2. Collect all dirty pages, sorted by page id. Write them to the journal in 1 MiB chunks, then the header, then sync the journal. This sync is the atomicity point.
3. Write the pages in place in the page file and sync it.
4. Truncate the WAL (unless the checkpoint happened in the middle of applying a batch, see below), truncate the journal, mark all frames clean.

Checkpoints are triggered when the WAL passes `wal_checkpoint_bytes`, when dirty pages come within 64 frames of the pool capacity, at clean close, and at the end of recovery.

### Recovery

1. Journal: if it is complete and valid, write every page in it to its home location and sync. If it is incomplete, the crash happened before any in-place write, so the page file is still the previous checkpoint; discard it. Truncate the journal either way.
2. Open the page file. A new file is formatted in memory only; an existing one must have the right magic.
3. Replay WAL records whose LSN is greater than the meta page's checkpoint LSN, applying each batch to the tree.
4. Run a checkpoint so the WAL is empty and the page file is current.

## Invariants

1. Between checkpoints the page file on disk is byte-identical to the state after the last completed checkpoint. Only checkpoint writes to it (NO-STEAL), and it writes through the journal.
2. A batch is committed if and only if its WAL record is intact on disk. The record reaches the file before any page reflecting the batch is modified in memory.
3. The meta page's checkpoint LSN never exceeds the last batch that is fully applied, so replaying WAL records above it reconstructs every committed batch.
4. Replaying a batch on top of a page file that already contains part or all of it gives the same result, because put and delete of a whole key are idempotent. This is what makes a mid-batch checkpoint safe.
5. The buffer pool never evicts a pinned or dirty frame. The LRU list contains exactly the frames that are unpinned and clean.
6. Every key in the subtree under child `i` of an internal node lies in `[key[i], key[i+1])`; leaves are all at the same depth and the leaf chain visits them in key order. `BTree::verify()` checks all of this and the tests call it after every crash.
7. Only one process opens a database at a time (`flock` on the page file). The lock dies with the process, so a killed writer never blocks recovery.

## Trade-offs

### Chosen: NO-STEAL buffer pool with a logical redo log

Because dirty pages are never written back before a checkpoint, the page file is always a consistent snapshot, and the WAL can log operations (`put k v`) instead of page images or page diffs. The WAL has no undo records, no page LSNs, and no compensation records. Recovery is "load snapshot, redo ops".

The cost is that every page dirtied since the last checkpoint must fit in the pool. When it does not, kvdb checkpoints early. A single batch larger than the pool is handled by checkpointing in the middle of applying it without truncating the WAL, and relying on invariant 4.

### Rejected: ARIES-style STEAL/NO-FORCE with physiological logging

It would let the pool evict dirty pages and would make checkpoints fuzzy and cheap, but it needs page LSNs, undo logging (or a rule that uncommitted data never reaches pages), and a three-pass recovery. That is the right design once there are concurrent transactions; for v0 it was more machinery than the problem needed.

### Chosen: journal (double write) for crash-atomic checkpoints

A checkpoint overwrites many pages in place; a crash halfway would leave a tree whose parent points at a child from a different generation. Writing all pages to a journal first, syncing, then writing in place means a crash at any byte leaves either a complete journal (redo it) or an untouched page file. It costs writing every checkpointed page twice.

### Rejected: copy-on-write (shadow paging) B+ tree

LMDB-style copy-on-write avoids the double write and gives atomic commits by swapping the root, but it needs a free-page list and a way to know which old pages are still reachable, and it turns every leaf update into a rewrite of the whole root-to-leaf path. Page reclamation is out of scope for v0, so the journal was simpler.

### Chosen: lazy delete, no free list

Deleting everything and reinserting works (there is a test for it), but space is never reclaimed and a tree with many deletes has sparse leaves. Merge and redistribution are subtle and were explicitly allowed to be skipped.

### Chosen: fixed size limits instead of overflow pages

128-byte keys and 512-byte values keep every cell inside a page and make splits always succeed. Large values would need overflow chains.

### Chosen: byte-balanced splits and the rightmost-append split

Splitting by count with variable cell sizes can leave one side overfull. Splitting by bytes fixes that. The append special case is what makes the sequential load produce nearly full leaves.

### Chosen: exact LRU with a linked list

A `std::list` plus iterators in each frame gives O(1) touch and eviction and is easy to test. CLOCK would avoid list pointer churn on every hit and is what I would use with concurrency.

### Chosen: `pread`/`pwrite`, not `mmap`

With `mmap` the OS decides when dirty pages reach the file, which would break invariant 1. Explicit I/O keeps write ordering under the engine's control.

### Chosen: three sync modes

`SyncMode::kNone` (write only, survives a process crash), `kFsync` (on macOS this reaches the drive but not necessarily its cache), and `kFullFsync` (`F_FULLFSYNC`, the only real barrier on macOS). The default is `kFsync`, matching SQLite's default on macOS. Group commit is available by putting many ops in one `WriteBatch`.

### Chosen: preallocating the WAL

The WAL grows in 4 MiB `ftruncate` steps so appends usually land inside the file. The measured gain is modest (see DEVLOG.md, `results/bench_walgrow.csv`), but it costs nothing and replay already has to handle a garbage tail.
