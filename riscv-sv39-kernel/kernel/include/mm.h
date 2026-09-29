/* Physical page allocator, Sv39 page tables and kernel heap. */
#pragma once
#include "types.h"
#include "kernel.h"

/* mm/pmm.c */
void pmm_init(struct boot_info *bi);
u64 pmm_alloc(void);        /* zeroed page, or 0 */
u64 pmm_alloc_nozero(void); /* page with stale contents, or 0 */
int pmm_free(u64 pa);       /* 0, -E_DOUBLEFREE or -E_INVAL */
u64 pmm_alloc_contig(u64 npages);
int pmm_free_contig(u64 pa, u64 npages);
u64 pmm_free_pages(void);
u64 pmm_total_pages(void);
bool pmm_is_allocated(u64 pa);
extern u64 pmm_double_frees;

/* mm/vm.c */
typedef u64 pte_t;
typedef pte_t *pagetable_t;
#define PTE_V (1UL << 0)
#define PTE_R (1UL << 1)
#define PTE_W (1UL << 2)
#define PTE_X (1UL << 3)
#define PTE_U (1UL << 4)
#define PTE_G (1UL << 5)
#define PTE_A (1UL << 6)
#define PTE_D (1UL << 7)
#define PA2PTE(pa) ((((u64)(pa)) >> 12) << 10)
#define PTE2PA(pte) ((((u64)(pte)) >> 10) << 12)
#define PX(level, va) ((((u64)(va)) >> (12 + 9 * (level))) & 0x1FF)

extern pagetable_t kernel_pt;
extern u64 kernel_pt_pages;
void kvm_init(struct boot_info *bi);
void kvm_enable(void);
pte_t *vm_walk(pagetable_t pt, u64 va, bool alloc);
int vm_map(pagetable_t pt, u64 va, u64 pa, u64 size, u64 perm);
int vm_unmap(pagetable_t pt, u64 va, u64 npages, bool free_frames);
u64 vm_translate(pagetable_t pt, u64 va);
void vm_describe(pagetable_t pt, u64 va);
int kvm_map_kstack(int slot);
void kvm_unmap_kstack(int slot);

/* mm/kmalloc.c */
void kmalloc_init(void);
void *kmalloc(size_t n);
void *kzalloc(size_t n);
void kfree(void *p);
int kfree_checked(void *p);
struct kmalloc_stats {
    u64 allocs, frees, bytes_in_use, pages_used, large_allocs, bad_frees;
};
void kmalloc_get_stats(struct kmalloc_stats *s);

/* linker symbols */
extern char _kernel_start[], _text_end[], _rodata_end[], _data_start[];
extern char _bss_start[], _bss_end[], _stacks_start[], _stacks_end[], _kernel_end[];
extern char boot_stacks[];
