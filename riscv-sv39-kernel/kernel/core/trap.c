/* Trap dispatch. kernelvec (arch/trapvec.S) saves the full register frame on
 * the current kernel stack and calls kernel_trap(tf). scause bit 63 tells an
 * interrupt from an exception; the low bits give the cause code. */
#include "kernel.h"
#include "riscv.h"
#include "cpu.h"
#include "mm.h"
#include "thread.h"
#include "trap.h"

extern char kernelvec[];
bool trap_quiet_recover;
u64 last_recovered_scause, last_recovered_stval;

static const char *exc_names[] = {
    [EXC_INST_MISALIGNED] = "instruction address misaligned",
    [EXC_INST_ACCESS] = "instruction access fault",
    [EXC_ILLEGAL_INST] = "illegal instruction",
    [EXC_BREAKPOINT] = "breakpoint",
    [EXC_LOAD_MISALIGNED] = "load address misaligned",
    [EXC_LOAD_ACCESS] = "load access fault",
    [EXC_STORE_MISALIGNED] = "store/AMO address misaligned",
    [EXC_STORE_ACCESS] = "store/AMO access fault",
    [EXC_ECALL_U] = "environment call from U-mode",
    [EXC_ECALL_S] = "environment call from S-mode",
    [10] = NULL,
    [11] = NULL,
    [EXC_INST_PAGE_FAULT] = "instruction page fault",
    [EXC_LOAD_PAGE_FAULT] = "load page fault",
    [14] = NULL,
    [EXC_STORE_PAGE_FAULT] = "store/AMO page fault",
};

const char *scause_name(u64 scause) {
    u64 code = scause & ~SCAUSE_INTR;
    if (scause & SCAUSE_INTR) {
        if (code == IRQ_S_SOFT) return "supervisor software interrupt";
        if (code == IRQ_S_TIMER) return "supervisor timer interrupt";
        if (code == IRQ_S_EXT) return "supervisor external interrupt";
        return "unknown interrupt";
    }
    if (code < ARRAY_SIZE(exc_names) && exc_names[code]) return exc_names[code];
    return "unknown exception";
}

void trap_init_hart(void) { csr_write(stvec, (u64)kernelvec); }

static const char *regnames[31] = {"ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0",
                                   "s1", "a0", "a1", "a2", "a3", "a4", "a5", "a6",
                                   "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8",
                                   "s9", "s10", "s11", "t3", "t4", "t5", "t6"};

static bool is_page_fault(u64 code) {
    return code == EXC_INST_PAGE_FAULT || code == EXC_LOAD_PAGE_FAULT ||
           code == EXC_STORE_PAGE_FAULT;
}

static void report_trap(struct trapframe *tf) {
    u64 code = tf->scause & ~SCAUSE_INTR;
    u64 off;
    const char *fn = ksym_lookup(tf->sepc, &off);
    struct thread *t = mycpu()->cur;
    kprintf("\n================ KERNEL TRAP ================\n");
    kprintf("cause   : %s (scause=%lu)\n", scause_name(tf->scause), code);
    kprintf("pc      : %p <%s+0x%lx>\n", (void *)tf->sepc, fn, off);
    kprintf("stval   : %p\n", (void *)tf->stval);
    kprintf("thread  : %s (tid %d) on hart %d\n", t ? t->name : "?", t ? t->tid : -1,
            mycpu()->hartid);
    if (is_page_fault(code)) {
        const char *kind = code == EXC_LOAD_PAGE_FAULT    ? "read from"
                           : code == EXC_STORE_PAGE_FAULT ? "write to"
                                                          : "instruction fetch from";
        kprintf("access  : %s %p\n", kind, (void *)tf->stval);
        kprintf("walk    :\n");
        u64 satp = csr_read(satp);
        vm_describe(satp ? (pagetable_t)((satp & ((1UL << 44) - 1)) << 12) : NULL, tf->stval);
        if (tf->stval < PGSIZE) kprintf("hint    : address is in page 0, probably a NULL pointer\n");
    } else if (code == EXC_ILLEGAL_INST) {
        kprintf("instr   : 0x%08lx (faulting instruction bits)\n", tf->stval);
    }
    kprintf("sstatus : 0x%lx (SPP=%s SPIE=%lu)\n", tf->sstatus,
            tf->sstatus & SSTATUS_SPP ? "S" : "U", (tf->sstatus >> 5) & 1);
    u64 *regs = (u64 *)tf;
    for (int i = 0; i < 31; i++)
        kprintf("%4s=%016lx%s", regnames[i], regs[i], (i % 4 == 3 || i == 30) ? "\n" : "  ");
    kprintf("=============================================\n");
}

void kernel_trap(struct trapframe *tf) {
    struct cpu *c = mycpu();
    u64 cause = tf->scause;
    if (!(tf->sstatus & SSTATUS_SPP)) panic("trap from U-mode, but v0 has no user mode");
    if (intr_get()) panic("kernel_trap: interrupts enabled on entry");

    if (cause & SCAUSE_INTR) {
        u64 code = cause & ~SCAUSE_INTR;
        if (code == IRQ_S_TIMER) {
            c->ntimer++;
            timer_interrupt();
            preempt_tick(); /* may switch threads; we come back here later */
        } else if (code == IRQ_S_EXT) {
            c->nexternal++;
            u32 irq = plic_claim();
            if (irq == bootinfo.uart_irq) uart_intr();
            else if (irq) kprintf("trap: unexpected external irq %u\n", irq);
            if (irq) plic_complete(irq);
        } else if (code == IRQ_S_SOFT) {
            csr_clear(sip, SIE_SSIE);
        } else {
            panic("unexpected interrupt, scause=0x%lx", cause);
        }
        return;
    }

    c->nexceptions++;
    if (c->onfault) {
        /* An expected fault from a probe helper: report briefly and resume. */
        last_recovered_scause = cause;
        last_recovered_stval = tf->stval;
        if (!trap_quiet_recover) {
            u64 off;
            const char *fn = ksym_lookup(tf->sepc, &off);
            kprintf("    [trap] recovered %s at %s+0x%lx, stval=%p\n", scause_name(cause), fn,
                    off, (void *)tf->stval);
        }
        tf->sepc = c->onfault;
        c->onfault = 0;
        c->nrecovered++;
        return;
    }
    report_trap(tf);
    panic("unhandled exception: %s", scause_name(cause));
}

/* Called from kernelvec on the emergency stack when sp ran into the guard. */
NORETURN void kstack_overflow_panic(u64 sepc, u64 stval, u64 scause, u64 sp, u64 fp) {
    struct cpu *c = mycpu();
    struct thread *t = c->cur;
    panicking = 1;
    u64 off;
    const char *fn = ksym_lookup(sepc, &off);
    kprintf("\n================ KERNEL STACK OVERFLOW ================\n");
    kprintf("thread  : %s (tid %d)\n", t->name, t->tid);
    kprintf("stack   : %p..%p, guard page %p..%p\n", (void *)c->kstack_lo, (void *)c->kstack_hi,
            (void *)(c->kstack_lo - PGSIZE), (void *)c->kstack_lo);
    kprintf("sp      : %p (%ld bytes below the stack bottom)\n", (void *)sp,
            (long)(c->kstack_lo - sp));
    kprintf("trap    : %s at %p <%s+0x%lx>, stval=%p\n", scause_name(scause), (void *)sepc, fn,
            off, (void *)stval);
    kprintf("backtrace (innermost first):\n");
    backtrace_from(fp, c->kstack_lo, c->kstack_hi);
    kprintf("KERNEL PANIC: kernel stack overflow\n");
    system_exit(EXIT_PANIC);
}

int try_read(u64 addr, u64 *out) {
    push_off();
    int r = probe_read64(addr, out);
    pop_off();
    return r;
}
int try_write(u64 addr, u64 val) {
    push_off();
    int r = probe_write64(addr, val);
    pop_off();
    return r;
}
int try_exec(u64 addr) {
    push_off();
    int r = probe_exec(addr);
    pop_off();
    return r;
}
