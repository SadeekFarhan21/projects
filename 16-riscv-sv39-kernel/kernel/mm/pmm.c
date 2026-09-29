/* Physical page frame allocator: one bit per 4 KiB frame of RAM (1 = in use).
 * A bitmap costs 1 bit/page (4 KiB for 128 MiB) and makes double frees and
 * frees of never-allocated pages detectable, which a free list cannot do
 * without extra metadata. Allocation scans 64-bit words starting at a
 * rotating hint, so the common case touches one word. */
#include "kernel.h"
#include "mm.h"
#include "spinlock.h"

static struct spinlock pmm_lock = {.name = "pmm"};
static u64 *bitmap;
static u64 ram_base, ram_pages, nwords;
static u64 nfree;
static u64 hint; /* word index where the next search starts */
u64 pmm_double_frees;

static inline bool bit_get(u64 i) { return (bitmap[i / 64] >> (i % 64)) & 1; }
static inline void bit_set(u64 i) { bitmap[i / 64] |= 1UL << (i % 64); }
static inline void bit_clear(u64 i) { bitmap[i / 64] &= ~(1UL << (i % 64)); }

static void mark_range(u64 base, u64 size, bool used) {
    u64 lo = ALIGN_DOWN(base, PGSIZE), hi = ALIGN_UP(base + size, PGSIZE);
    if (hi <= ram_base || lo >= ram_base + ram_pages * PGSIZE) return;
    lo = MAX(lo, ram_base);
    hi = MIN(hi, ram_base + ram_pages * PGSIZE);
    for (u64 pa = lo; pa < hi; pa += PGSIZE) {
        u64 i = (pa - ram_base) / PGSIZE;
        if (used && !bit_get(i)) { bit_set(i); nfree--; }
        else if (!used && bit_get(i)) { bit_clear(i); nfree++; }
    }
}

void pmm_init(struct boot_info *bi) {
    /* v0 manages the first RAM range; QEMU virt has exactly one */
    ram_base = bi->mem[0].base;
    ram_pages = bi->mem[0].size / PGSIZE;
    nwords = (ram_pages + 63) / 64;

    bitmap = (u64 *)ALIGN_UP((u64)_kernel_end, PGSIZE);
    u64 bitmap_bytes = ALIGN_UP(nwords * 8, PGSIZE);
    memset(bitmap, 0xff, nwords * 8); /* everything starts "used" */
    nfree = 0;

    u64 first_free = (u64)bitmap + bitmap_bytes;
    mark_range(first_free, ram_base + ram_pages * PGSIZE - first_free, false);
    /* then take back what firmware and the DTB still need */
    for (int i = 0; i < bi->nrsv; i++) mark_range(bi->rsv[i].base, bi->rsv[i].size, true);
    mark_range(bi->dtb_pa, bi->dtb_size, true);
    hint = (first_free - ram_base) / PGSIZE / 64;

    kprintf("pmm: RAM %p..%p (%lu pages), bitmap %lu bytes at %p, %lu pages free\n",
            (void *)ram_base, (void *)(ram_base + ram_pages * PGSIZE), ram_pages, nwords * 8,
            (void *)bitmap, nfree);
}

static u64 alloc_locked(void) {
    for (u64 n = 0; n < nwords; n++) {
        u64 w = (hint + n) % nwords;
        if (bitmap[w] != ~0UL) {
            u64 bit = (u64)__builtin_ctzl(~bitmap[w]);
            u64 i = w * 64 + bit;
            if (i >= ram_pages) continue;
            bit_set(i);
            nfree--;
            hint = w;
            return ram_base + i * PGSIZE;
        }
    }
    return 0;
}

u64 pmm_alloc_nozero(void) {
    spin_lock(&pmm_lock);
    u64 pa = alloc_locked();
    spin_unlock(&pmm_lock);
    return pa;
}

u64 pmm_alloc(void) {
    u64 pa = pmm_alloc_nozero();
    if (pa) memset((void *)pa, 0, PGSIZE);
    return pa;
}

int pmm_free(u64 pa) {
    if (pa & (PGSIZE - 1) || pa < ram_base || pa >= ram_base + ram_pages * PGSIZE)
        return -E_INVAL;
    u64 i = (pa - ram_base) / PGSIZE;
    spin_lock(&pmm_lock);
    if (!bit_get(i)) {
        pmm_double_frees++;
        spin_unlock(&pmm_lock);
        return -E_DOUBLEFREE;
    }
    bit_clear(i);
    nfree++;
    if (i / 64 < hint) hint = i / 64; /* keep allocations low and dense */
    spin_unlock(&pmm_lock);
    return 0;
}

/* First fit over the bitmap for n physically contiguous pages. */
u64 pmm_alloc_contig(u64 npages) {
    if (npages == 0) return 0;
    if (npages == 1) return pmm_alloc();
    spin_lock(&pmm_lock);
    u64 run = 0;
    for (u64 i = 0; i < ram_pages; i++) {
        if (bit_get(i)) { run = 0; continue; }
        if (++run == npages) {
            u64 first = i + 1 - npages;
            for (u64 j = first; j <= i; j++) bit_set(j);
            nfree -= npages;
            spin_unlock(&pmm_lock);
            u64 pa = ram_base + first * PGSIZE;
            memset((void *)pa, 0, npages * PGSIZE);
            return pa;
        }
    }
    spin_unlock(&pmm_lock);
    return 0;
}

int pmm_free_contig(u64 pa, u64 npages) {
    int err = 0;
    for (u64 i = 0; i < npages; i++) {
        int r = pmm_free(pa + i * PGSIZE);
        if (r && !err) err = r;
    }
    return err;
}

bool pmm_is_allocated(u64 pa) {
    if (pa < ram_base || pa >= ram_base + ram_pages * PGSIZE) return true;
    return bit_get((pa - ram_base) / PGSIZE);
}

u64 pmm_free_pages(void) { return nfree; }
u64 pmm_total_pages(void) { return ram_pages; }
