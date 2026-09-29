/* kmain: the boot hart's C entry point, called from boot/entry.S with the
 * MMU off, on the hart's boot stack, with BSS already zeroed. */
#include "kernel.h"
#include "riscv.h"
#include "cpu.h"
#include "mm.h"
#include "thread.h"
#include "trap.h"
#include "test.h"

struct cpu cpus[NCPU];
struct boot_info bootinfo;
u64 boot_hartid;
static u8 emerg_stacks[NCPU][EMERG_STACK_SIZE] __attribute__((aligned(16)));

static const char *hsm_state(long s) {
    switch (s) {
    case 0: return "started";
    case 1: return "stopped";
    case 2: return "start pending";
    case 3: return "stop pending";
    case 4: return "suspended";
    default: return "unknown";
    }
}

static void print_bootinfo(struct boot_info *bi) {
    kprintf("fdt: blob at %p, %lu bytes, %d cpu(s)\n", (void *)bi->dtb_pa, bi->dtb_size, bi->ncpus);
    for (int i = 0; i < bi->nmem; i++)
        kprintf("fdt: memory   %p..%p (%lu MiB)\n", (void *)bi->mem[i].base,
                (void *)(bi->mem[i].base + bi->mem[i].size), bi->mem[i].size >> 20);
    for (int i = 0; i < bi->nrsv; i++)
        kprintf("fdt: reserved %p..%p\n", (void *)bi->rsv[i].base,
                (void *)(bi->rsv[i].base + bi->rsv[i].size));
    kprintf("fdt: uart ns16550a at %p irq %u, plic at %p (%u sources), test device at %p\n",
            (void *)bi->uart_base, bi->uart_irq, (void *)bi->plic_base, bi->plic_ndev,
            (void *)bi->test_base);
    kprintf("fdt: timebase %lu Hz, bootargs \"%s\"\n", bi->timebase_hz, bi->bootargs);
}

void kmain(u64 hartid, u64 dtb) {
    struct cpu *c = &cpus[hartid];
    asm volatile("mv tp, %0" ::"r"(c));
    boot_hartid = hartid;
    c->hartid = (int)hartid;
    u64 slot = (u64)boot_stacks + hartid * (BOOT_STACK_PAGES + 1) * PGSIZE;
    c->kstack_lo = slot + PGSIZE; /* the first page of the slot is the guard */
    c->kstack_hi = slot + (BOOT_STACK_PAGES + 1) * PGSIZE;
    c->emerg_sp = (u64)&emerg_stacks[hartid][EMERG_STACK_SIZE];

    /* FS=Off: the kernel is built without FP, so any FP instruction traps
     * instead of silently using registers we never save. */
    csr_clear(sstatus, SSTATUS_FS | SSTATUS_SUM | SSTATUS_SIE);
    csr_write(sie, 0);
    trap_init_hart();

    kprintf("\n16-os v0: RISC-V 64 kernel, boot hart %lu, dtb at %p\n", hartid, (void *)dtb);
    sbi_print_info();

    int r = fdt_parse(dtb, &bootinfo);
    if (r) panic("device tree parse failed (%d)", r);
    print_bootinfo(&bootinfo);
    if (!bootinfo.uart_base || !bootinfo.plic_base || !bootinfo.timebase_hz)
        panic("device tree is missing the uart, plic or timebase");

    uart_init(bootinfo.uart_base);
    console_use_uart();
    kprintf("console: now on ns16550a at %p (was SBI debug console)\n", (void *)bootinfo.uart_base);

    pmm_init(&bootinfo);
    kvm_init(&bootinfo);
    kvm_enable();
    kprintf("kvm: Sv39 paging enabled, satp=%p\n", (void *)csr_read(satp));
    kmalloc_init();
    thread_init();

    plic_init(bootinfo.plic_base, (int)hartid);
    plic_enable(bootinfo.uart_irq);
    uart_enable_rx_irq();
    csr_set(sie, SIE_SEIE | SIE_SSIE);
    timer_init(bootinfo.timebase_hz);

    for (int h = 0; h < bootinfo.ncpus && h < NCPU; h++)
        if ((u64)h != hartid)
            kprintf("smp: hart %d is %s (parked by SBI HSM; v0 runs on one hart)\n", h,
                    hsm_state(sbi_hart_status((u64)h)));

    char mode[16] = "demo", crash[24];
    bootarg_value(bootinfo.bootargs, "mode", mode, sizeof(mode));
    void (*entry)(void *) = demo_main;
    if (bootarg_value(bootinfo.bootargs, "crash", crash, sizeof(crash))) entry = crash_main;
    else if (!strcmp(mode, "test")) entry = selftest_main;
    else if (!strcmp(mode, "bench")) entry = bench_main;
    kprintf("boot: %lu us since reset (timebase clock), starting \"%s\"\n",
            rdtime() * 1000000 / bootinfo.timebase_hz, entry == crash_main ? "crash" : mode);

    if (!thread_create("main", entry, NULL)) panic("cannot create main thread");
    intr_on();
    idle_loop();
}
