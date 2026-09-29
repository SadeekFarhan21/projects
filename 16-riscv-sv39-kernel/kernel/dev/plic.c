/* Platform-Level Interrupt Controller. On QEMU virt, context 2*hart is the
 * hart's M-mode context and 2*hart+1 its S-mode context. */
#include "kernel.h"
#include "riscv.h"
#include "trap.h"

#define PLIC_PRIORITY(irq) (plic_base + 4 * (irq))
#define PLIC_ENABLE(ctx) (plic_base + 0x2000 + 0x80 * (ctx))
#define PLIC_THRESHOLD(ctx) (plic_base + 0x200000 + 0x1000 * (ctx))
#define PLIC_CLAIM(ctx) (plic_base + 0x200004 + 0x1000 * (ctx))

static u64 plic_base;
static int plic_ctx;

void plic_init(u64 base, int hart) {
    plic_base = base;
    plic_ctx = 2 * hart + 1;
    mmio_write32(PLIC_THRESHOLD(plic_ctx), 0); /* accept every priority > 0 */
}

void plic_enable(u32 irq) {
    mmio_write32(PLIC_PRIORITY(irq), 1);
    u64 reg = PLIC_ENABLE(plic_ctx) + 4 * (irq / 32);
    mmio_write32(reg, mmio_read32(reg) | (1u << (irq % 32)));
}

u32 plic_claim(void) { return mmio_read32(PLIC_CLAIM(plic_ctx)); }
void plic_complete(u32 irq) { mmio_write32(PLIC_CLAIM(plic_ctx), irq); }
