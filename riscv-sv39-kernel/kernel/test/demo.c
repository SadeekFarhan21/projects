/* `make run` demo: several kernel threads share the CPU and print
 * interleaved output, then an interactive monitor reads commands typed on
 * the UART (the characters are echoed by the UART interrupt handler). */
#include "kernel.h"
#include "riscv.h"
#include "cpu.h"
#include "mm.h"
#include "thread.h"
#include "trap.h"
#include "test.h"

static struct spinlock tally_lock = {.name = "tally"};
static u64 tally;

static void busy_ticks(u64 n) {
    u64 end = ticks + n;
    while (ticks < end)
        ;
}

static void worker(void *arg) {
    char id = (char)(u64)arg;
    for (int step = 0; step < 4; step++) {
        kprintf("[worker-%c] step %d at tick %lu\n", id, step, ticks);
        busy_ticks(2); /* burn CPU so the timer has to preempt us */
        spin_lock(&tally_lock);
        tally++;
        spin_unlock(&tally_lock);
        if (step % 2) thread_sleep_ticks((u64)(id - 'A' + 1));
    }
    kprintf("[worker-%c] done, preempted %lu times\n", id, thread_current()->npreempt);
}

void demo_main(void *arg) {
    (void)arg;
    kprintf("\n=== demo: 4 kernel threads, round-robin, %d ms quantum ===\n",
            1000 * QUANTUM_TICKS / HZ);
    struct thread *t[4];
    for (int i = 0; i < 4; i++) t[i] = thread_create("worker", worker, (void *)(u64)('A' + i));
    kprintf("[main] threads while workers run:\n");
    thread_dump();
    for (int i = 0; i < 4; i++) thread_join(t[i]);
    kprintf("[main] all workers joined, tally=%lu (want 16), %lu context switches so far\n",
            tally, sched_nswitch());
    if (bootarg_has(bootinfo.bootargs, "autoexit")) {
        kprintf("[main] autoexit set, powering off\n");
        system_exit(EXIT_OK);
    }
    monitor_main(NULL);
}

static void cmd_help(void) {
    kprintf("commands: help ps mem ticks stats pf panic poweroff\n");
}

void monitor_main(void *arg) {
    (void)arg;
    char line[64];
    kprintf("\nkernel monitor ready (type 'help'; Ctrl-A X quits QEMU)\n");
    for (;;) {
        kprintf("kmon> ");
        console_getline(line, sizeof line);
        if (!line[0]) continue;
        if (!strcmp(line, "help")) cmd_help();
        else if (!strcmp(line, "ps")) thread_dump();
        else if (!strcmp(line, "mem")) {
            struct kmalloc_stats s;
            kmalloc_get_stats(&s);
            kprintf("pages: %lu free of %lu; heap: %lu bytes in use, %lu slab pages, %lu allocs\n",
                    pmm_free_pages(), pmm_total_pages(), s.bytes_in_use, s.pages_used, s.allocs);
        } else if (!strcmp(line, "ticks")) {
            kprintf("ticks=%lu (%lu s)\n", ticks, ticks / HZ);
        } else if (!strcmp(line, "stats")) {
            struct cpu *c = mycpu();
            kprintf("timer=%lu external=%lu exceptions=%lu switches=%lu uart_irqs=%lu\n",
                    c->ntimer, c->nexternal, c->nexceptions, sched_nswitch(), uart_irqs);
        } else if (!strcmp(line, "pf")) {
            u64 v;
            kprintf("reading address 0 through a fault probe:\n");
            try_read(0, &v);
        } else if (!strcmp(line, "panic")) {
            panic("panic requested from the monitor");
        } else if (!strcmp(line, "poweroff")) {
            system_exit(EXIT_OK);
        } else {
            kprintf("unknown command '%s'\n", line);
        }
    }
}
