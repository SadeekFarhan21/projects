/* Boot-time self-test suite (bootargs "mode=test"). Each test returns 0 on
 * success. Results print as PASS/FAIL lines and the kernel powers QEMU off
 * with exit code 0 (all passed) or 1 (any failure), which `make test` checks. */
#include "kernel.h"
#include "riscv.h"
#include "cpu.h"
#include "mm.h"
#include "thread.h"
#include "trap.h"
#include "test.h"

#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            kprintf("    check failed: %s (%s:%d)\n", #c, __FILE__, __LINE__);      \
            return -1;                                                               \
        }                                                                            \
    } while (0)

/* ---------------------------------------------------------------- printf */
static int t_printf(void) {
    char b[96];
    ksnprintf(b, sizeof b, "[%5d|%-5d|%05d]", 42, 42, 42);
    CHECK(!strcmp(b, "[   42|42   |00042]"));
    ksnprintf(b, sizeof b, "%d %d %ld", 0, -7, -9223372036854775807L - 1);
    CHECK(!strcmp(b, "0 -7 -9223372036854775808"));
    ksnprintf(b, sizeof b, "%x %X %08lx %lu", 0xbeefu, 0xbeefu, 0x1234UL, 18446744073709551615UL);
    CHECK(!strcmp(b, "beef BEEF 00001234 18446744073709551615"));
    ksnprintf(b, sizeof b, "%p", (void *)0x80200000UL);
    CHECK(!strcmp(b, "0x0000000080200000"));
    ksnprintf(b, sizeof b, "%s|%-4s|%4s|%c|%%|%s", "ab", "x", "y", 'z', (char *)NULL);
    CHECK(!strcmp(b, "ab|x   |   y|z|%|(null)"));
    ksnprintf(b, sizeof b, "%06d", -42);
    CHECK(!strcmp(b, "-00042"));
    CHECK(ksnprintf(b, 4, "%s", "truncate") == 8 && !strcmp(b, "tru"));
    return 0;
}

/* ------------------------------------------------------------ device tree */
static int t_fdt(void) {
    CHECK(bootinfo.nmem >= 1);
    CHECK(bootinfo.mem[0].base == 0x80000000UL);
    CHECK(bootinfo.mem[0].size >= (64UL << 20) && (bootinfo.mem[0].size % PGSIZE) == 0);
    CHECK((u64)_kernel_end <= bootinfo.mem[0].base + bootinfo.mem[0].size);
    CHECK(bootinfo.nrsv >= 1); /* OpenSBI reserves its own memory */
    CHECK(bootinfo.uart_base && bootinfo.uart_irq);
    CHECK(bootinfo.plic_base && bootinfo.test_base);
    CHECK(bootinfo.timebase_hz > 0);
    char mode[16];
    CHECK(bootarg_value(bootinfo.bootargs, "mode", mode, sizeof mode) && !strcmp(mode, "test"));
    kprintf("    RAM %lu MiB at %p, %d reserved range(s), timebase %lu Hz\n",
            bootinfo.mem[0].size >> 20, (void *)bootinfo.mem[0].base, bootinfo.nrsv,
            bootinfo.timebase_hz);
    return 0;
}

/* ------------------------------------------------------ physical memory */
static int t_pmm_basic(void) {
    u64 f0 = pmm_free_pages();
    u64 pa = pmm_alloc();
    CHECK(pa != 0 && (pa & (PGSIZE - 1)) == 0);
    CHECK(pmm_free_pages() == f0 - 1);
    CHECK(pmm_is_allocated(pa));
    for (u64 i = 0; i < PGSIZE / 8; i++) CHECK(((u64 *)pa)[i] == 0); /* zeroed */
    CHECK(pmm_free(pa) == 0);
    CHECK(!pmm_is_allocated(pa));
    CHECK(pmm_free_pages() == f0);
    /* reserved memory is never handed out */
    CHECK(pmm_is_allocated(bootinfo.mem[0].base));      /* OpenSBI */
    CHECK(pmm_is_allocated((u64)_kernel_start));         /* kernel image */
    CHECK(pmm_is_allocated(ALIGN_DOWN(bootinfo.dtb_pa, PGSIZE)));
    return 0;
}

static int t_pmm_double_free(void) {
    u64 before = pmm_double_frees;
    u64 f0 = pmm_free_pages();
    u64 pa = pmm_alloc();
    CHECK(pa);
    CHECK(pmm_free(pa) == 0);
    CHECK(pmm_free(pa) == -E_DOUBLEFREE);
    CHECK(pmm_double_frees == before + 1);
    CHECK(pmm_free(pa + 8) == -E_INVAL);   /* unaligned */
    CHECK(pmm_free(0x1000) == -E_INVAL);   /* outside RAM */
    CHECK(pmm_free_pages() == f0);         /* the bad frees changed nothing */
    return 0;
}

/* Allocate every free page, chaining them through their first word, then
 * free them all. Checks uniqueness (tags survive), exhaustion and recovery. */
static int t_pmm_exhaust(void) {
    u64 f0 = pmm_free_pages();
    u64 head = 0, n = 0;
    for (;;) {
        u64 pa = pmm_alloc_nozero();
        if (!pa) break;
        ((u64 *)pa)[0] = head;
        ((u64 *)pa)[1] = pa ^ 0x5a5a5a5a5a5a5a5aUL; /* tag: detects a page handed out twice */
        head = pa;
        n++;
    }
    CHECK(n == f0);
    CHECK(pmm_free_pages() == 0);
    CHECK(pmm_alloc() == 0);
    u64 freed = 0;
    while (head) {
        u64 next = ((u64 *)head)[0];
        CHECK(((u64 *)head)[1] == (head ^ 0x5a5a5a5a5a5a5a5aUL));
        CHECK(pmm_free(head) == 0);
        head = next;
        freed++;
    }
    CHECK(freed == n);
    CHECK(pmm_free_pages() == f0);
    kprintf("    allocated and freed all %lu free pages\n", n);
    return 0;
}

static int t_pmm_contig(void) {
    u64 f0 = pmm_free_pages();
    u64 pa = pmm_alloc_contig(8);
    CHECK(pa);
    for (int i = 0; i < 8; i++) CHECK(pmm_is_allocated(pa + (u64)i * PGSIZE));
    CHECK(pmm_free_pages() == f0 - 8);
    CHECK(pmm_free_contig(pa, 8) == 0);
    CHECK(pmm_free_pages() == f0);
    return 0;
}

/* ------------------------------------------------------- virtual memory */
static int t_vm_map_unmap(void) {
    u64 pa = pmm_alloc();
    CHECK(pa);
    *(u64 *)pa = 0x1122334455667788UL;
    CHECK(vm_map(kernel_pt, TEST_VA, pa, PGSIZE, PTE_R | PTE_W) == 0);
    CHECK(vm_translate(kernel_pt, TEST_VA + 0x10) == pa + 0x10);
    pte_t *pte = vm_walk(kernel_pt, TEST_VA, false);
    CHECK(pte && (*pte & (PTE_V | PTE_R | PTE_W | PTE_A | PTE_D)) == (PTE_V | PTE_R | PTE_W | PTE_A | PTE_D));
    CHECK(!(*pte & (PTE_X | PTE_U)));
    CHECK(*(volatile u64 *)TEST_VA == 0x1122334455667788UL); /* read through the new VA */
    *(volatile u64 *)(TEST_VA + 8) = 0xabcdef;                /* write through it */
    CHECK(((u64 *)pa)[1] == 0xabcdef);                        /* visible via the PA alias */
    CHECK(vm_map(kernel_pt, TEST_VA, pa, PGSIZE, PTE_R) == -E_EXIST);
    CHECK(vm_unmap(kernel_pt, TEST_VA, 1, false) == 0);
    CHECK(vm_translate(kernel_pt, TEST_VA) == 0);
    u64 v;
    CHECK(try_read(TEST_VA, &v) == -1 && last_recovered_scause == EXC_LOAD_PAGE_FAULT);
    CHECK(last_recovered_stval == TEST_VA);
    CHECK(try_write(TEST_VA, 1) == -1 && last_recovered_scause == EXC_STORE_PAGE_FAULT);
    CHECK(vm_unmap(kernel_pt, TEST_VA, 1, false) == -E_NOENT);
    /* map read-only: reads work, writes fault */
    CHECK(vm_map(kernel_pt, TEST_VA, pa, PGSIZE, PTE_R) == 0);
    CHECK(try_read(TEST_VA, &v) == 0 && v == 0x1122334455667788UL);
    CHECK(try_write(TEST_VA, 2) == -1 && last_recovered_scause == EXC_STORE_PAGE_FAULT);
    CHECK(vm_unmap(kernel_pt, TEST_VA, 1, true) == 0); /* also frees pa */
    CHECK(!pmm_is_allocated(pa));
    return 0;
}

static const u64 ro_const = 0xC0FFEE;
static u32 data_ret_insn[4] = {0x00008067, 0, 0, 0}; /* "ret" placed in .data */
static void __attribute__((noinline)) text_fn(void) { asm volatile(""); }

static int t_vm_kernel_perms(void) {
    u64 v;
    pte_t *pte;
    /* text: R-X, not writable (write back the same value so a bug is harmless) */
    pte = vm_walk(kernel_pt, (u64)text_fn, false);
    CHECK(pte && (*pte & PTE_X) && (*pte & PTE_R) && !(*pte & PTE_W));
    CHECK(try_read((u64)text_fn & ~7UL, &v) == 0);
    CHECK(try_write((u64)text_fn & ~7UL, v) == -1 && last_recovered_scause == EXC_STORE_PAGE_FAULT);
    CHECK(try_exec((u64)text_fn) == 0);
    /* rodata: R-- */
    pte = vm_walk(kernel_pt, (u64)&ro_const, false);
    CHECK(pte && (*pte & PTE_R) && !(*pte & (PTE_W | PTE_X)));
    CHECK(try_write((u64)&ro_const, ro_const) == -1);
    /* data: RW-, never executable (W^X) */
    pte = vm_walk(kernel_pt, (u64)data_ret_insn, false);
    CHECK(pte && (*pte & PTE_W) && !(*pte & PTE_X));
    CHECK(try_exec((u64)data_ret_insn) == -1 && last_recovered_scause == EXC_INST_PAGE_FAULT);
    /* OpenSBI's memory is not mapped at all */
    CHECK(try_read(bootinfo.mem[0].base, &v) == -1);
    return 0;
}

static int t_vm_guard_pages(void) {
    struct thread *t = thread_current();
    u64 v;
    CHECK(try_read(t->kstack_lo, &v) == 0);             /* lowest stack page is mapped */
    CHECK(try_read(t->kstack_lo - 8, &v) == -1);        /* the page below is the guard */
    CHECK(last_recovered_scause == EXC_LOAD_PAGE_FAULT);
    CHECK(try_read(t->kstack_hi, &v) == -1);            /* next slot's guard page */
    for (int h = 0; h < NCPU; h++) {                     /* boot stack guards */
        u64 slot = (u64)boot_stacks + (u64)h * (BOOT_STACK_PAGES + 1) * PGSIZE;
        CHECK(try_read(slot, &v) == -1);
        CHECK(try_read(slot + PGSIZE, &v) == 0);
    }
    return 0;
}

/* ------------------------------------------------------------------ heap */
static int t_heap_basic(void) {
    struct kmalloc_stats s0, s1;
    kmalloc_get_stats(&s0);
    static const size_t sizes[] = {1, 8, 15, 16, 17, 31, 48, 100, 500, 1000, 2000, 2032, 2033, 4096, 10000, 65536};
    void *p[ARRAY_SIZE(sizes)];
    for (u64 i = 0; i < ARRAY_SIZE(sizes); i++) {
        p[i] = kmalloc(sizes[i]);
        CHECK(p[i] != NULL);
        CHECK(((u64)p[i] & 15) == 0);
        memset(p[i], (int)(i + 1), sizes[i]);
    }
    for (u64 i = 0; i < ARRAY_SIZE(sizes); i++)       /* no allocation overlapped another */
        for (u64 j = 0; j < sizes[i]; j++) CHECK(((u8 *)p[i])[j] == (u8)(i + 1));
    for (u64 i = 0; i < ARRAY_SIZE(sizes); i++) kfree(p[i]);
    kmalloc_get_stats(&s1);
    CHECK(s1.bytes_in_use == s0.bytes_in_use);
    CHECK(s1.allocs - s0.allocs == ARRAY_SIZE(sizes) && s1.frees - s0.frees == ARRAY_SIZE(sizes));
    CHECK(kmalloc(0) == NULL);
    void *z = kzalloc(300);
    CHECK(z);
    for (int i = 0; i < 300; i++) CHECK(((u8 *)z)[i] == 0);
    kfree(z);
    return 0;
}

static int t_heap_reuse_and_double_free(void) {
    void *a = kmalloc(64);
    CHECK(a);
    kfree(a);
    void *b = kmalloc(64);
    CHECK(b == a); /* LIFO free list hands the same block back */
    CHECK(kfree_checked(b) == 0);
    CHECK(kfree_checked(b) == -E_DOUBLEFREE);
    CHECK(kfree_checked((u8 *)b + 8) == -E_BADPTR);
    void *big = kmalloc(3 * PGSIZE);
    CHECK(big);
    CHECK(kfree_checked(big) == 0);
    CHECK(kfree_checked(big) == -E_DOUBLEFREE);
    return 0;
}

static int t_heap_stress(void) {
    struct kmalloc_stats s0, s1;
    kmalloc_get_stats(&s0);
    enum { SLOTS = 96 };
    u8 *slot[SLOTS] = {0};
    size_t len[SLOTS] = {0};
    u64 seed = 12345;
    for (int op = 0; op < 5000; op++) {
        seed = seed * 6364136223846793005UL + 1442695040888963407UL;
        int i = (int)((seed >> 33) % SLOTS);
        if (slot[i]) {
            for (size_t j = 0; j < len[i]; j++) CHECK(slot[i][j] == (u8)i);
            kfree(slot[i]);
            slot[i] = NULL;
        } else {
            len[i] = 1 + (seed >> 40) % 3000;
            slot[i] = kmalloc(len[i]);
            CHECK(slot[i]);
            memset(slot[i], i, len[i]);
        }
    }
    for (int i = 0; i < SLOTS; i++) if (slot[i]) kfree(slot[i]);
    kmalloc_get_stats(&s1);
    CHECK(s1.bytes_in_use == s0.bytes_in_use);
    kprintf("    5000 random ops, heap now holds %lu slab pages\n", s1.pages_used);
    return 0;
}

/* ----------------------------------------------------------------- traps */
static int t_traps(void) {
    u64 v;
    struct cpu *c = mycpu();
    u64 rec0 = c->nrecovered;
    CHECK(try_read(0, &v) == -1);
    CHECK(last_recovered_scause == EXC_LOAD_PAGE_FAULT && last_recovered_stval == 0);
    CHECK(try_exec(0) == -1 && last_recovered_scause == EXC_INST_PAGE_FAULT);
    push_off();
    int ill = probe_illegal();
    int brk = probe_ebreak();
    pop_off();
    CHECK(ill == -1);
    CHECK(brk == -1 && last_recovered_scause == EXC_BREAKPOINT);
    push_off();
    ill = probe_illegal();
    pop_off();
    CHECK(ill == -1 && last_recovered_scause == EXC_ILLEGAL_INST);
    CHECK(last_recovered_stval == 0xc0001073UL); /* stval holds the instruction bits */
    CHECK(c->nrecovered - rec0 == 5);
    return 0;
}

/* ----------------------------------------------------------------- timer */
/* Read ticks and this hart's timer-interrupt count as one snapshot. With
 * interrupts on, a tick can land between the two loads and the pair would
 * disagree by one (this test failed that way once before it used push_off). */
static void tick_snapshot(u64 *t, u64 *n) {
    push_off();
    *t = ticks;
    *n = mycpu()->ntimer;
    pop_off();
}

static int t_timer_ticks(void) {
    u64 t0, n0, t1, n1;
    tick_snapshot(&t0, &n0);
    u64 start = rdtime(), wait = timebase_hz / 10; /* 100 ms busy wait */
    while (rdtime() - start < wait)
        ;
    tick_snapshot(&t1, &n1);
    u64 dt = t1 - t0;
    kprintf("    %lu ticks during 100 ms busy wait (expect about %d)\n", dt, HZ / 10);
    /* lower bound is loose: a loaded host can starve the vCPU thread for
     * longer than a tick, and missed ticks are dropped, not replayed */
    CHECK(dt >= 1 && dt <= 15);
    CHECK(n1 - n0 == dt);
    u64 s = ticks;
    thread_sleep_ticks(5);
    CHECK(ticks - s >= 5 && ticks - s <= 7);
    return 0;
}

/* ------------------------------------------------------ context switching */
static struct context co_main_ctx, co_ctx;
static volatile u64 co_count;
static void co_fn(void) {
    for (;;) {
        co_count++;
        swtch(&co_ctx, &co_main_ctx);
    }
}

static int t_swtch_raw(void) {
    u64 stack = pmm_alloc();
    CHECK(stack);
    memset(&co_ctx, 0, sizeof co_ctx);
    co_ctx.ra = (u64)co_fn;
    co_ctx.sp = stack + PGSIZE;
    co_count = 0;
    push_off(); /* no preemption while on a stack the scheduler does not know */
    for (int i = 0; i < 1000; i++) swtch(&co_main_ctx, &co_ctx);
    pop_off();
    CHECK(co_count == 1000);
    CHECK(pmm_free(stack) == 0);
    return 0;
}

static volatile int turn;
static char pp_log[64];
static volatile int pp_len;
static void pingpong(void *arg) {
    int me = (int)(u64)arg;
    for (int i = 0; i < 10; i++) {
        while (turn != me) yield();
        pp_log[pp_len++] = me ? 'B' : 'A';
        turn = !me;
    }
}

static int t_yield_pingpong(void) {
    turn = 0;
    pp_len = 0;
    u64 sw0 = sched_nswitch();
    struct thread *a = thread_create("ping", pingpong, (void *)0);
    struct thread *b = thread_create("pong", pingpong, (void *)1);
    CHECK(a && b);
    CHECK(thread_join(a) == 0 && thread_join(b) == 0);
    pp_log[pp_len] = 0;
    CHECK(!strcmp(pp_log, "ABABABABABABABABABAB"));
    CHECK(sched_nswitch() - sw0 >= 20);
    return 0;
}

static volatile u64 spin_cnt[2], seen_other[2];
static void spinner(void *arg) {
    int me = (int)(u64)arg;
    u64 end = ticks + 20;
    while (ticks < end) spin_cnt[me]++; /* never yields voluntarily */
    seen_other[me] = spin_cnt[!me];
}

static int t_preemption(void) {
    spin_cnt[0] = spin_cnt[1] = 0;
    struct thread *a = thread_create("spin0", spinner, (void *)0);
    struct thread *b = thread_create("spin1", spinner, (void *)1);
    CHECK(a && b);
    thread_join(a);
    thread_join(b);
    kprintf("    spin0 saw spin1 at %lu, spin1 saw spin0 at %lu\n", seen_other[0], seen_other[1]);
    /* without preemption spin1 could not start before spin0 finished */
    CHECK(seen_other[0] > 0 && seen_other[1] > 0);
    return 0;
}

/* producer/consumer through a bounded buffer with sleep_on/wakeup */
static struct {
    struct spinlock lk;
    int buf[4];
    int r, w;
    u64 sum;
} pc = {.lk = {.name = "pc"}};

static void producer(void *arg) {
    (void)arg;
    for (int i = 1; i <= 200; i++) {
        spin_lock(&pc.lk);
        while (pc.w - pc.r == 4) sleep_on(&pc.w, &pc.lk);
        pc.buf[pc.w++ % 4] = i;
        wakeup(&pc.r);
        spin_unlock(&pc.lk);
    }
}

static void consumer(void *arg) {
    (void)arg;
    for (int i = 1; i <= 200; i++) {
        spin_lock(&pc.lk);
        while (pc.w == pc.r) sleep_on(&pc.r, &pc.lk);
        pc.sum += (u64)pc.buf[pc.r++ % 4];
        wakeup(&pc.w);
        spin_unlock(&pc.lk);
    }
}

static int t_sleep_wakeup(void) {
    pc.r = pc.w = 0;
    pc.sum = 0;
    struct thread *c = thread_create("consumer", consumer, NULL);
    struct thread *p = thread_create("producer", producer, NULL);
    CHECK(c && p);
    thread_join(p);
    thread_join(c);
    CHECK(pc.sum == 200 * 201 / 2);
    return 0;
}

static struct spinlock cnt_lock = {.name = "counter"};
static volatile u64 locked_counter, racy_counter;
static void incrementer(void *arg) {
    (void)arg;
    for (int i = 0; i < 20000; i++) {
        spin_lock(&cnt_lock);
        u64 v = locked_counter;
        for (volatile int d = 0; d < 20; d++)
            ;
        locked_counter = v + 1;
        spin_unlock(&cnt_lock);

        u64 r = racy_counter; /* same pattern without the lock */
        for (volatile int d = 0; d < 20; d++)
            ;
        racy_counter = r + 1;
    }
}

static int t_spinlock(void) {
    locked_counter = racy_counter = 0;
    struct thread *t[4];
    for (int i = 0; i < 4; i++) CHECK((t[i] = thread_create("incr", incrementer, NULL)));
    for (int i = 0; i < 4; i++) thread_join(t[i]);
    kprintf("    locked counter %lu (want 80000), unlocked control %lu (lost %lu updates)\n",
            locked_counter, racy_counter, 80000 - racy_counter);
    CHECK(locked_counter == 80000);
    return 0;
}

static struct sleeplock slk;
static volatile int inside, overlaps, sl_done;
static void sleeplock_user(void *arg) {
    (void)arg;
    for (int i = 0; i < 5; i++) {
        sleeplock_acquire(&slk);
        if (inside++) overlaps++;
        thread_sleep_ticks(1); /* sleeping while holding a sleep lock is allowed */
        inside--;
        sl_done++;
        sleeplock_release(&slk);
    }
}

static int t_sleeplock(void) {
    sleeplock_init(&slk, "test");
    inside = overlaps = sl_done = 0;
    struct thread *t[3];
    for (int i = 0; i < 3; i++) CHECK((t[i] = thread_create("slk", sleeplock_user, NULL)));
    for (int i = 0; i < 3; i++) thread_join(t[i]);
    CHECK(overlaps == 0 && sl_done == 15);
    return 0;
}

static void exit42(void *arg) {
    (void)arg;
    thread_exit(42);
}

static int t_thread_lifecycle(void) {
    struct thread *w = thread_create("warm", exit42, NULL); /* allocate page-table pages once */
    CHECK(w && thread_join(w) == 42);
    u64 f0 = pmm_free_pages();
    for (int i = 0; i < 50; i++) {
        struct thread *t = thread_create("exit42", exit42, NULL);
        CHECK(t);
        CHECK(thread_join(t) == 42);
    }
    CHECK(pmm_free_pages() == f0); /* stacks were freed on join */
    struct thread *many[NTHREAD];
    int n = 0;
    while (n < NTHREAD && (many[n] = thread_create("fill", exit42, NULL))) n++;
    kprintf("    created %d threads before the table was full\n", n);
    CHECK(n > 0 && n < NTHREAD); /* main holds one slot */
    for (int i = 0; i < n; i++) CHECK(thread_join(many[i]) == 42);
    return 0;
}

/* -------------------------------------------------------------- UART rx */
static int t_uart_rx(void) {
    u64 deadline = ticks + 5 * HZ;
    while (console_rx_chars < 5 && ticks < deadline) thread_sleep_ticks(1);
    CHECK(console_rx_chars >= 5);
    char line[32];
    console_getline(line, sizeof line);
    kprintf("    received \"%s\" through %lu UART interrupts\n", line, uart_irqs);
    CHECK(!strcmp(line, "ping"));
    CHECK(uart_irqs > 0);
    return 0;
}

struct test {
    const char *name;
    int (*fn)(void);
};

static const struct test tests[] = {
    {"printf formatting", t_printf},
    {"device tree parse", t_fdt},
    {"pmm alloc/free", t_pmm_basic},
    {"pmm double-free detection", t_pmm_double_free},
    {"pmm exhaust and recover", t_pmm_exhaust},
    {"pmm contiguous alloc", t_pmm_contig},
    {"vm map/unmap", t_vm_map_unmap},
    {"vm kernel permissions", t_vm_kernel_perms},
    {"vm stack guard pages", t_vm_guard_pages},
    {"heap basic", t_heap_basic},
    {"heap reuse and double free", t_heap_reuse_and_double_free},
    {"heap random stress", t_heap_stress},
    {"trap recovery", t_traps},
    {"timer ticks", t_timer_ticks},
    {"raw context switch", t_swtch_raw},
    {"yield ping-pong", t_yield_pingpong},
    {"timer preemption", t_preemption},
    {"sleep/wakeup", t_sleep_wakeup},
    {"spinlock mutual exclusion", t_spinlock},
    {"sleep lock", t_sleeplock},
    {"thread create/exit/join", t_thread_lifecycle},
};

void selftest_main(void *arg) {
    (void)arg;
    int pass = 0, fail = 0;
    trap_quiet_recover = true; /* expected faults are checked, not printed */
    kprintf("\n=== kernel self-test ===\n");
    bool rx = bootarg_has(bootinfo.bootargs, "rxtest");
    char only[32] = ""; /* "only=timer" runs just the tests whose name contains "timer" */
    bootarg_value(bootinfo.bootargs, "only", only, sizeof only);
    int n = (int)ARRAY_SIZE(tests) + (rx ? 1 : 0), total = 0;
    for (int i = 0; i < n; i++) {
        const struct test *t = i < (int)ARRAY_SIZE(tests) ? &tests[i]
                                                          : &(struct test){"uart rx interrupt", t_uart_rx};
        if (only[0] && !strstr(t->name, only)) continue;
        total++;
        u64 t0 = rdtime();
        int r = t->fn();
        u64 us = (rdtime() - t0) * 1000000 / timebase_hz;
        kprintf("[%s] %-28s %8lu us\n", r == 0 ? "PASS" : "FAIL", t->name, us);
        if (r == 0) pass++;
        else fail++;
    }
    struct cpu *c = mycpu();
    kprintf("stats: ticks=%lu timer_irqs=%lu ext_irqs=%lu exceptions=%lu recovered=%lu switches=%lu\n",
            ticks, c->ntimer, c->nexternal, c->nexceptions, c->nrecovered, sched_nswitch());
    kprintf("SELFTEST SUMMARY: %d passed, %d failed, %d total\n", pass, fail, total);
    if (fail == 0) kprintf("ALL TESTS PASSED\n");
    system_exit(fail ? EXIT_TESTFAIL : EXIT_OK);
}
