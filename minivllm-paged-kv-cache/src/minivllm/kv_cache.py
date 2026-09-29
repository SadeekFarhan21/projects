"""Paged KV cache: a fixed pool of blocks plus a free-list allocator.

Each sequence owns a *block table*, a list of physical block ids. Logical token
position p of a sequence lives in physical slot

    slot = block_table[p // block_size] * block_size + p % block_size

so a sequence's KV does not need to be contiguous in memory, and the only
fragmentation is the unused tail of each sequence's last block.
"""

from __future__ import annotations

import torch


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    """Free-list allocator over block ids [0, num_blocks).

    Invariant: every block id is either in the free list or owned by exactly one
    sequence. `num_free + num_used == num_blocks` at all times.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # Pop from the end, so low ids are handed out first (easier to debug).
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)

    def can_allocate(self, n: int) -> bool:
        return n <= len(self._free)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise OutOfBlocks(f"need {n} blocks, only {len(self._free)} free")
        out = [self._free.pop() for _ in range(n)]
        self._used.update(out)
        return out

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if b not in self._used:
                raise ValueError(f"double free or foreign block {b}")
            self._used.remove(b)
            self._free.append(b)


def blocks_needed(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size


class KVCache:
    """The physical KV storage: one tensor of shape
    [num_layers, 2 (k/v), num_blocks * block_size, num_kv_heads, head_dim].

    Slots are flattened across blocks so that writes are a single index_copy_
    per layer and reads are a single gather per layer.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.data = torch.zeros(
            (num_layers, 2, num_blocks * block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

    def layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[i, 0], self.data[i, 1]

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()
