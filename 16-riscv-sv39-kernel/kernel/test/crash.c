/* Deliberate crashes (bootargs "crash=<kind>") used by `make test` to check
 * that fatal errors produce readable reports and a nonzero exit code. Each
 * crash runs in its own kernel thread so it has a guarded stack. */
#include "kernel.h"
#include "riscv.h"
#include "thread.h"
#include "test.h"

static volatile int sink;

__attribute__((noinline)) static void crash_leaf(const char *kind) {
    if (!strcmp(kind, "panic")) panic("deliberate panic (crash=panic)");
    if (!strcmp(kind, "pagefault")) *(volatile u64 *)0x10 = 1; /* store to page 0 */
    if (!strcmp(kind, "illegal")) asm volatile(".4byte 0xc0001073"); /* csrw cycle */
    sink++;
}
__attribute__((noinline)) static void crash_middle(const char *kind) { crash_leaf(kind); sink++; }
__attribute__((noinline)) static void crash_outer(const char *kind) { crash_middle(kind); sink++; }

static volatile bool keep_going = true;
__attribute__((noinline)) static u64 recurse(u64 depth) {
    volatile char frame[200];
    frame[0] = (char)depth;
    if (!keep_going) return depth;
    return recurse(depth + 1) + frame[0];
}

static char kind[24];

static void crash_thread(void *arg) {
    (void)arg;
    kprintf("crash: triggering '%s'\n", kind);
    if (!strcmp(kind, "stackoverflow")) recurse(0);
    crash_outer(kind);
    kprintf("crash: '%s' did not crash\n", kind);
}

void crash_main(void *arg) {
    (void)arg;
    bootarg_value(bootinfo.bootargs, "crash", kind, sizeof kind);
    struct thread *t = thread_create("crasher", crash_thread, NULL);
    thread_join(t);
    system_exit(EXIT_TESTFAIL); /* reaching here means nothing crashed */
}
