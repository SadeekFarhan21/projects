/* Panic, frame-pointer backtraces, symbol lookup and power-off.
 * With -fno-omit-frame-pointer every function saves ra at fp-8 and the
 * caller's fp at fp-16, so the stack is a linked list of frame records. */
#include "kernel.h"
#include "riscv.h"
#include "cpu.h"
#include "thread.h"
#include "mm.h"

volatile int panicking;

struct ksym {
    u64 addr;
    const char *name;
};
extern const struct ksym ksyms[];
extern const u64 ksyms_count;

const char *ksym_lookup(u64 addr, u64 *off) {
    *off = 0;
    if (ksyms_count == 0 || addr < ksyms[0].addr) return "?";
    u64 lo = 0, hi = ksyms_count; /* find last entry with ksyms[i].addr <= addr */
    while (hi - lo > 1) {
        u64 mid = (lo + hi) / 2;
        if (ksyms[mid].addr <= addr) lo = mid;
        else hi = mid;
    }
    if (addr >= (u64)_text_end) return "?";
    *off = addr - ksyms[lo].addr;
    return ksyms[lo].name;
}

void backtrace_from(u64 fp, u64 lo, u64 hi) {
    u64 last_ra = 0, repeats = 0;
    int printed = 0;
    for (int depth = 0; depth < 4096 && printed < 24; depth++) {
        if (fp < lo + 16 || fp > hi || (fp & 7)) break;
        u64 ra = ((u64 *)fp)[-1];
        u64 prev = ((u64 *)fp)[-2];
        if (ra == 0) break;
        if (ra == last_ra) { /* deep recursion: print one line per run */
            repeats++;
        } else {
            if (repeats) kprintf("      ... same frame repeated %lu more times\n", repeats);
            repeats = 0;
            u64 off;
            /* ra points after the call; look up ra-1 so a noreturn call at
             * the very end of a function is not attributed to the next one */
            const char *name = ksym_lookup(ra - 1, &off);
            kprintf("  #%-2d %p  %s+0x%lx\n", depth, (void *)ra, name, off + 1);
            printed++;
            last_ra = ra;
        }
        if (prev <= fp) break; /* frames must move toward the stack top */
        fp = prev;
    }
    if (repeats) kprintf("      ... same frame repeated %lu more times\n", repeats);
}

void panic_at(const char *file, int line, const char *fmt, ...) {
    intr_off();
    if (panicking > 1) system_exit(EXIT_PANIC);
    panicking++;
    struct cpu *c = mycpu();
    va_list ap;
    va_start(ap, fmt);
    kprintf("\n!!! KERNEL PANIC at %s:%d\n!!! ", file, line);
    kvprintf(fmt, ap);
    va_end(ap);
    struct thread *t = c->cur;
    kprintf("\n!!! hart %d, thread %s (tid %d)\nbacktrace (innermost first):\n", c->hartid,
            t ? t->name : "?", t ? t->tid : -1);
    backtrace_from(read_fp(), c->kstack_lo, c->kstack_hi);
    system_exit(EXIT_PANIC);
}

/* Power off QEMU. Exit code 0 goes through SBI SRST (the standard path).
 * A nonzero code needs the sifive,test device, because SRST has no way to
 * carry an exit status: writing (code << 16) | 0x3333 makes QEMU exit(code). */
void system_exit(int code) {
    intr_off();
    if (code == 0) sbi_system_reset(0 /* shutdown */, 0 /* no reason */);
    if (bootinfo.test_base) {
        mmio_write32(bootinfo.test_base, code == 0 ? 0x5555 : (((u32)code << 16) | 0x3333));
        /* QEMU acts on this write asynchronously. Do not fall through to the
         * SRST call below: OpenSBI's poweroff writes the PASS value to the
         * same device, and that second request could win the race and turn
         * a failing exit code into 0 (seen in crash-pagefault under load). */
        for (;;) wfi();
    }
    sbi_system_reset(0, 1 /* system failure */);
    for (;;) wfi();
}
