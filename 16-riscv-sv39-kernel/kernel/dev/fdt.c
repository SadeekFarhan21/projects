/* Flattened device tree (DTB) parser. We walk the structure block once and
 * pull out only what v0 needs: RAM ranges, reserved ranges (OpenSBI's own
 * memory), UART, PLIC, the test/poweroff device, timebase and bootargs.
 * All DTB fields are big-endian. */
#include "kernel.h"

#define FDT_MAGIC 0xd00dfeed
#define FDT_BEGIN_NODE 1
#define FDT_END_NODE 2
#define FDT_PROP 3
#define FDT_NOP 4
#define FDT_END 9
#define MAX_DEPTH 16

static u32 be32(const void *p) {
    const u8 *b = p;
    return ((u32)b[0] << 24) | ((u32)b[1] << 16) | ((u32)b[2] << 8) | b[3];
}
static u64 be64(const void *p) { return ((u64)be32(p) << 32) | be32((const u8 *)p + 4); }

static u64 read_cells(const u8 *p, u32 cells) {
    u64 v = 0;
    for (u32 i = 0; i < cells; i++) v = (v << 32) | be32(p + 4 * i);
    return v;
}

/* Properties of the node currently being parsed. */
struct node {
    const char *name;
    const u8 *reg;
    u32 reg_len;
    const char *compat;
    u32 compat_len;
    const char *devtype;
    u32 irq;
    bool has_irq;
    u32 addr_cells, size_cells; /* for this node's children */
    bool done;
};

static bool compat_has(const struct node *n, const char *want) {
    const char *p = n->compat;
    if (!p) return false;
    const char *end = p + n->compat_len;
    while (p < end) {
        if (!strcmp(p, want)) return true;
        p += strlen(p) + 1;
    }
    return false;
}

static void add_region(struct region *arr, int *n, u64 base, u64 size) {
    if (*n < MAX_REGIONS && size) arr[(*n)++] = (struct region){base, size};
}

/* Called once per node after all of its properties have been seen. */
static void finish_node(struct node *stack, int depth, struct boot_info *bi) {
    struct node *n = &stack[depth];
    struct node *parent = depth > 0 ? &stack[depth - 1] : NULL;
    if (n->done) return;
    n->done = true;
    u32 ac = parent ? parent->addr_cells : 2, sc = parent ? parent->size_cells : 1;
    u32 entry = 4 * (ac + sc);
    u64 base0 = 0, size0 = 0;
    if (n->reg && n->reg_len >= entry) {
        base0 = read_cells(n->reg, ac);
        size0 = read_cells(n->reg + 4 * ac, sc);
    }

    if (n->devtype && !strcmp(n->devtype, "memory") && n->reg) {
        for (u32 off = 0; off + entry <= n->reg_len; off += entry)
            add_region(bi->mem, &bi->nmem, read_cells(n->reg + off, ac),
                       read_cells(n->reg + off + 4 * ac, sc));
    }
    if (parent && !strcmp(parent->name, "reserved-memory") && n->reg) {
        for (u32 off = 0; off + entry <= n->reg_len; off += entry)
            add_region(bi->rsv, &bi->nrsv, read_cells(n->reg + off, ac),
                       read_cells(n->reg + off + 4 * ac, sc));
    }
    if (n->devtype && !strcmp(n->devtype, "cpu")) bi->ncpus++;
    if (compat_has(n, "ns16550a") && !bi->uart_base) {
        bi->uart_base = base0;
        bi->uart_size = size0;
        bi->uart_irq = n->has_irq ? n->irq : 10;
    }
    if ((compat_has(n, "riscv,plic0") || compat_has(n, "sifive,plic-1.0.0")) && !bi->plic_base) {
        bi->plic_base = base0;
        bi->plic_size = size0;
    }
    if (compat_has(n, "sifive,test0") && !bi->test_base) bi->test_base = base0;
}

int fdt_parse(u64 dtb_pa, struct boot_info *bi) {
    const u8 *fdt = (const u8 *)dtb_pa;
    if (be32(fdt) != FDT_MAGIC) return -E_INVAL;
    u32 totalsize = be32(fdt + 4);
    u32 off_struct = be32(fdt + 8);
    u32 off_strings = be32(fdt + 12);
    u32 off_rsvmap = be32(fdt + 16);
    bi->dtb_pa = dtb_pa;
    bi->dtb_size = totalsize;

    /* memory reservation block: (address, size) pairs ending with (0, 0) */
    for (const u8 *r = fdt + off_rsvmap;; r += 16) {
        u64 a = be64(r), s = be64(r + 8);
        if (!a && !s) break;
        add_region(bi->rsv, &bi->nrsv, a, s);
    }

    const char *strings = (const char *)fdt + off_strings;
    const u8 *p = fdt + off_struct;
    struct node stack[MAX_DEPTH];
    int depth = -1;

    for (;;) {
        u32 tok = be32(p);
        p += 4;
        if (tok == FDT_BEGIN_NODE) {
            /* properties always precede child nodes, so the parent is complete */
            if (depth >= 0) finish_node(stack, depth, bi);
            if (++depth >= MAX_DEPTH) return -E_INVAL;
            struct node *n = &stack[depth];
            memset(n, 0, sizeof(*n));
            n->name = (const char *)p;
            n->addr_cells = 2;
            n->size_cells = 1;
            p += ALIGN_UP(strlen(n->name) + 1, 4);
        } else if (tok == FDT_END_NODE) {
            finish_node(stack, depth, bi);
            depth--;
        } else if (tok == FDT_PROP) {
            u32 len = be32(p);
            const char *pname = strings + be32(p + 4);
            const u8 *val = p + 8;
            p += 8 + ALIGN_UP(len, 4);
            struct node *n = &stack[depth];
            if (!strcmp(pname, "reg")) { n->reg = val; n->reg_len = len; }
            else if (!strcmp(pname, "compatible")) { n->compat = (const char *)val; n->compat_len = len; }
            else if (!strcmp(pname, "device_type")) n->devtype = (const char *)val;
            else if (!strcmp(pname, "#address-cells")) n->addr_cells = be32(val);
            else if (!strcmp(pname, "#size-cells")) n->size_cells = be32(val);
            else if (!strcmp(pname, "interrupts") && len >= 4) { n->irq = be32(val); n->has_irq = true; }
            else if (!strcmp(pname, "riscv,ndev") && len >= 4) bi->plic_ndev = be32(val);
            else if (!strcmp(pname, "timebase-frequency") && !strcmp(n->name, "cpus"))
                bi->timebase_hz = len == 8 ? be64(val) : be32(val);
            else if (!strcmp(pname, "bootargs") && !strcmp(n->name, "chosen"))
                strlcpy(bi->bootargs, (const char *)val, sizeof(bi->bootargs));
        } else if (tok == FDT_NOP) {
            continue;
        } else if (tok == FDT_END) {
            break;
        } else {
            return -E_INVAL;
        }
    }
    return bi->nmem > 0 ? 0 : -E_NOENT;
}
