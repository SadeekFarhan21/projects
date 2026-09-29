/* Sv39 page tables. Three levels of 512 eight-byte PTEs; VA bits 38..30,
 * 29..21 and 20..12 index the levels. The kernel page table identity-maps
 * RAM and devices (VA == PA) and puts each thread's stack at its own high VA
 * with an unmapped guard page below it. */
#include "kernel.h"
#include "mm.h"
#include "riscv.h"

pagetable_t kernel_pt;
u64 kernel_pt_pages;

pte_t *vm_walk(pagetable_t pt, u64 va, bool alloc) {
    if (va >= MAXVA) return NULL;
    for (int level = 2; level > 0; level--) {
        pte_t *pte = &pt[PX(level, va)];
        if (*pte & PTE_V) {
            if (*pte & (PTE_R | PTE_W | PTE_X)) return NULL; /* superpage: not used in v0 */
            pt = (pagetable_t)PTE2PA(*pte);
        } else {
            if (!alloc) return NULL;
            u64 page = pmm_alloc();
            if (!page) return NULL;
            kernel_pt_pages++; /* v0 has only the kernel page table */
            *pte = PA2PTE(page) | PTE_V; /* non-leaf: only V set */
            pt = (pagetable_t)page;
        }
    }
    return &pt[PX(0, va)];
}

/* Map [va, va+size) to [pa, pa+size). Both must be page aligned. */
int vm_map(pagetable_t pt, u64 va, u64 pa, u64 size, u64 perm) {
    if ((va | pa | size) & (PGSIZE - 1)) return -E_INVAL;
    if (!(perm & (PTE_R | PTE_W | PTE_X))) return -E_INVAL;
    for (u64 off = 0; off < size; off += PGSIZE) {
        pte_t *pte = vm_walk(pt, va + off, true);
        if (!pte) return -E_NOMEM;
        if (*pte & PTE_V) return -E_EXIST;
        /* Set A (and D for writable pages) up front: without Svadu the
         * hardware raises a page fault instead of setting them. */
        u64 ad = PTE_A | ((perm & PTE_W) ? PTE_D : 0);
        *pte = PA2PTE(pa + off) | perm | ad | PTE_V;
    }
    return 0;
}

int vm_unmap(pagetable_t pt, u64 va, u64 npages, bool free_frames) {
    for (u64 i = 0; i < npages; i++) {
        u64 a = va + i * PGSIZE;
        pte_t *pte = vm_walk(pt, a, false);
        if (!pte || !(*pte & PTE_V)) return -E_NOENT;
        u64 pa = PTE2PA(*pte);
        *pte = 0;
        sfence_vma(a);
        if (free_frames) pmm_free(pa);
    }
    return 0;
}

u64 vm_translate(pagetable_t pt, u64 va) {
    pte_t *pte = vm_walk(pt, va, false);
    if (!pte || !(*pte & PTE_V)) return 0;
    return PTE2PA(*pte) | (va & (PGSIZE - 1));
}

/* Explain how a VA resolves, for page-fault reports. */
void vm_describe(pagetable_t pt, u64 va) {
    if (!pt) { kprintf("    (paging not enabled)\n"); return; }
    if (va >= MAXVA) { kprintf("    va %p is outside the Sv39 range we use\n", (void *)va); return; }
    for (int level = 2; level >= 0; level--) {
        pte_t pte = pt[PX(level, va)];
        if (!(pte & PTE_V)) {
            kprintf("    L%d[%3lu] invalid: no mapping for this address\n", level, PX(level, va));
            return;
        }
        if (pte & (PTE_R | PTE_W | PTE_X)) {
            kprintf("    L%d[%3lu] leaf pa=%p perms=%c%c%c%c%c%c\n", level, PX(level, va),
                    (void *)PTE2PA(pte), pte & PTE_R ? 'R' : '-', pte & PTE_W ? 'W' : '-',
                    pte & PTE_X ? 'X' : '-', pte & PTE_U ? 'U' : '-', pte & PTE_A ? 'A' : '-',
                    pte & PTE_D ? 'D' : '-');
            return;
        }
        pt = (pagetable_t)PTE2PA(pte);
    }
}

static void map_or_die(u64 va, u64 pa, u64 size, u64 perm, const char *what) {
    int r = vm_map(kernel_pt, va, pa, ALIGN_UP(size, PGSIZE), perm | PTE_G);
    if (r) panic("kvm: mapping %s failed (%d)", what, r);
}

void kvm_init(struct boot_info *bi) {
    kernel_pt = (pagetable_t)pmm_alloc();
    kernel_pt_pages = 1;
    u64 ks = (u64)_kernel_start;

    if (bi->test_base) map_or_die(bi->test_base, bi->test_base, PGSIZE, PTE_R | PTE_W, "test dev");
    map_or_die(bi->uart_base, bi->uart_base, PGSIZE, PTE_R | PTE_W, "uart");
    map_or_die(bi->plic_base, bi->plic_base, bi->plic_size, PTE_R | PTE_W, "plic");

    map_or_die(ks, ks, (u64)_text_end - ks, PTE_R | PTE_X, "text");
    map_or_die((u64)_text_end, (u64)_text_end, (u64)_rodata_end - (u64)_text_end, PTE_R, "rodata");
    map_or_die((u64)_rodata_end, (u64)_rodata_end, (u64)_stacks_start - (u64)_rodata_end,
               PTE_R | PTE_W, "data+bss");

    /* boot stacks: skip the first page of every slot (the guard) */
    for (int h = 0; h < NCPU; h++) {
        u64 slot = (u64)boot_stacks + (u64)h * (BOOT_STACK_PAGES + 1) * PGSIZE;
        map_or_die(slot + PGSIZE, slot + PGSIZE, BOOT_STACK_PAGES * PGSIZE, PTE_R | PTE_W,
                   "boot stack");
    }

    /* the rest of RAM (page bitmap, free frames, DTB) */
    u64 ram_end = bi->mem[0].base + bi->mem[0].size;
    map_or_die((u64)_kernel_end, (u64)_kernel_end, ram_end - (u64)_kernel_end, PTE_R | PTE_W,
               "free RAM");

    kprintf("kvm: text   %p..%p R-X\n", (void *)ks, (void *)_text_end);
    kprintf("kvm: rodata %p..%p R--\n", (void *)_text_end, (void *)_rodata_end);
    kprintf("kvm: data   %p..%p RW-\n", (void *)_rodata_end, (void *)_stacks_start);
    kprintf("kvm: stacks %p..%p RW- (%d boot stacks, guard page below each)\n",
            (void *)_stacks_start, (void *)_stacks_end, NCPU);
    kprintf("kvm: RAM    %p..%p RW-\n", (void *)_kernel_end, (void *)ram_end);
    kprintf("kvm: kernel page table uses %lu pages\n", kernel_pt_pages);
}

void kvm_enable(void) {
    sfence_vma_all();
    csr_write(satp, MAKE_SATP(kernel_pt));
    sfence_vma_all();
}

/* Kernel stacks: KSTACK_PAGES separately allocated frames mapped
 * contiguously at KSTACK_LO(slot); the page below stays unmapped. */
int kvm_map_kstack(int slot) {
    for (u64 i = 0; i < KSTACK_PAGES; i++) {
        u64 pa = pmm_alloc();
        if (!pa || vm_map(kernel_pt, KSTACK_LO(slot) + i * PGSIZE, pa, PGSIZE,
                          PTE_R | PTE_W | PTE_G)) {
            if (pa) pmm_free(pa);
            if (i) vm_unmap(kernel_pt, KSTACK_LO(slot), i, true);
            return -E_NOMEM;
        }
    }
    return 0;
}

void kvm_unmap_kstack(int slot) {
    if (vm_unmap(kernel_pt, KSTACK_LO(slot), KSTACK_PAGES, true))
        panic("kvm_unmap_kstack: slot %d not mapped", slot);
}
