/* Kernel heap: power-of-two size classes from 32 to 2048 bytes, each backed
 * by whole pages cut into equal blocks (a minimal slab). Every block starts
 * with a 16-byte header so kfree needs no size argument and can reject
 * double frees and wild pointers. Requests too big for the largest class get
 * physically contiguous pages from the page allocator.
 * Trade-off: slab pages are never returned to the page allocator. */
#include "kernel.h"
#include "mm.h"
#include "spinlock.h"

#define NCLASS 7 /* 32 64 128 256 512 1024 2048 */
#define MIN_SHIFT 5
#define MAGIC_USED 0x4b4d5553u /* "KMUS" */
#define MAGIC_FREE 0x4b4d4652u /* "KMFR" */
#define CLASS_LARGE 0xffu

struct hdr {
    u32 magic;
    u32 cls;
    u64 size; /* requested bytes, or page count for large allocations */
};
_Static_assert(sizeof(struct hdr) == 16, "header keeps payload 16-byte aligned");

struct freeblk {
    struct hdr h;
    struct freeblk *next;
};

static struct spinlock heap_lock = {.name = "kmalloc"};
static struct freeblk *freelist[NCLASS];
static struct kmalloc_stats st;

void kmalloc_init(void) { kprintf("kmalloc: %d size classes, 32..2048 bytes\n", NCLASS); }

static int size_class(size_t n) {
    for (int c = 0; c < NCLASS; c++)
        if (n + sizeof(struct hdr) <= (1UL << (c + MIN_SHIFT))) return c;
    return -1;
}

static bool refill(int c) {
    u64 page = pmm_alloc_nozero();
    if (!page) return false;
    u64 bs = 1UL << (c + MIN_SHIFT);
    for (u64 off = 0; off + bs <= PGSIZE; off += bs) {
        struct freeblk *b = (struct freeblk *)(page + off);
        b->h.magic = MAGIC_FREE;
        b->h.cls = (u32)c;
        b->next = freelist[c];
        freelist[c] = b;
    }
    st.pages_used++;
    return true;
}

void *kmalloc(size_t n) {
    if (n == 0) return NULL;
    int c = size_class(n);
    if (c < 0) {
        u64 np = ALIGN_UP(n + sizeof(struct hdr), PGSIZE) / PGSIZE;
        u64 pa = pmm_alloc_contig(np);
        if (!pa) return NULL;
        struct hdr *h = (struct hdr *)pa;
        h->magic = MAGIC_USED;
        h->cls = CLASS_LARGE;
        h->size = np;
        spin_lock(&heap_lock);
        st.allocs++;
        st.large_allocs++;
        st.bytes_in_use += np * PGSIZE;
        spin_unlock(&heap_lock);
        return h + 1;
    }
    spin_lock(&heap_lock);
    if (!freelist[c] && !refill(c)) {
        spin_unlock(&heap_lock);
        return NULL;
    }
    struct freeblk *b = freelist[c];
    freelist[c] = b->next;
    b->h.magic = MAGIC_USED;
    b->h.size = n;
    st.allocs++;
    st.bytes_in_use += 1UL << (c + MIN_SHIFT);
    spin_unlock(&heap_lock);
    return &b->h + 1;
}

void *kzalloc(size_t n) {
    void *p = kmalloc(n);
    if (p) memset(p, 0, n);
    return p;
}

int kfree_checked(void *p) {
    if (!p) return 0;
    if ((u64)p & 15) { st.bad_frees++; return -E_BADPTR; }
    struct hdr *h = (struct hdr *)p - 1;
    spin_lock(&heap_lock);
    if (h->magic == MAGIC_FREE) {
        st.bad_frees++;
        spin_unlock(&heap_lock);
        return -E_DOUBLEFREE;
    }
    if (h->magic != MAGIC_USED) {
        st.bad_frees++;
        spin_unlock(&heap_lock);
        return -E_BADPTR;
    }
    if (h->cls == CLASS_LARGE) {
        u64 np = h->size;
        h->magic = MAGIC_FREE;
        st.frees++;
        st.bytes_in_use -= np * PGSIZE;
        spin_unlock(&heap_lock);
        return pmm_free_contig((u64)h, np);
    }
    if (h->cls >= NCLASS) {
        st.bad_frees++;
        spin_unlock(&heap_lock);
        return -E_BADPTR;
    }
    struct freeblk *b = (struct freeblk *)h;
    b->h.magic = MAGIC_FREE;
    b->next = freelist[h->cls];
    freelist[h->cls] = b;
    st.frees++;
    st.bytes_in_use -= 1UL << (h->cls + MIN_SHIFT);
    spin_unlock(&heap_lock);
    return 0;
}

void kfree(void *p) {
    int r = kfree_checked(p);
    if (r) panic("kfree(%p): %s", p, r == -E_DOUBLEFREE ? "double free" : "bad pointer");
}

void kmalloc_get_stats(struct kmalloc_stats *s) {
    spin_lock(&heap_lock);
    *s = st;
    spin_unlock(&heap_lock);
}
