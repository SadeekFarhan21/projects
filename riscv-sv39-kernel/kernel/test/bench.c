/* In-kernel micro-benchmarks (bootargs "mode=bench").
 * Every number is measured inside QEMU. Under TCG, QEMU translates guest code
 * to host code on the fly, so these are emulator costs, not the cost on a
 * real RISC-V core. rdtime ticks at the DT timebase (10 MHz on virt, 100 ns
 * resolution), so every measurement loops many times and divides.
 * Output lines starting with "BENCH" are machine readable:
 *   BENCH <name> <value> <unit> */
#include "kernel.h"
#include "riscv.h"
#include "cpu.h"
#include "mm.h"
#include "thread.h"
#include "trap.h"
#include "test.h"

static bool have_cycle;

static u64 cyc(void) {
    if (!have_cycle) return 0;
    return rdcycle();
}

/* timebase units -> nanoseconds, scaled by 1000 to keep 3 decimals */
static u64 tb_to_ns_x1000(u64 tb) { return tb * 1000000000000UL / timebase_hz; }

static void report(const char *name, u64 dt_tb, u64 dcyc, u64 dinst, u64 n) {
    u64 ns1000 = tb_to_ns_x1000(dt_tb) / n;
    kprintf("BENCH %-28s %lu.%03lu ns/op", name, ns1000 / 1000, ns1000 % 1000);
    if (have_cycle) kprintf("  %lu cycles/op  %lu instret/op", dcyc / n, dinst / n);
    kprintf("  (n=%lu, total %lu us)\n", n, dt_tb * 1000000 / timebase_hz);
}

#define MEASURE(name, n, body)                                              \
    do {                                                                    \
        u64 _t0 = rdtime(), _c0 = cyc(), _i0 = have_cycle ? rdinstret() : 0; \
        body;                                                               \
        u64 _t1 = rdtime(), _c1 = cyc(), _i1 = have_cycle ? rdinstret() : 0; \
        report(name, _t1 - _t0, _c1 - _c0, _i1 - _i0, n);                   \
    } while (0)

/* --- raw swtch between two contexts, no scheduler involved --- */
static struct context bm_main, bm_co;
static void bm_co_fn(void) {
    for (;;) swtch(&bm_co, &bm_main);
}

static void bench_swtch(void) {
    u64 stack = pmm_alloc();
    memset(&bm_co, 0, sizeof bm_co);
    bm_co.ra = (u64)bm_co_fn;
    bm_co.sp = stack + PGSIZE;
    const u64 N = 200000;
    push_off();
    swtch(&bm_main, &bm_co); /* warm up */
    MEASURE("swtch (one direction)", 2 * N, for (u64 i = 0; i < N; i++) swtch(&bm_main, &bm_co));
    pop_off();
    pmm_free(stack);
}

/* --- yield between two threads through the scheduler --- */
static volatile bool yb_go;
static void yield_loop(void *arg) {
    u64 n = (u64)arg;
    while (!yb_go) yield();
    for (u64 i = 0; i < n; i++) yield();
}

static void bench_yield(void) {
    const u64 N = 50000;
    yb_go = false;
    struct thread *a = thread_create("yA", yield_loop, (void *)N);
    struct thread *b = thread_create("yB", yield_loop, (void *)N);
    thread_sleep_ticks(1); /* let both reach the start line */
    u64 s0 = sched_nswitch();
    u64 t0 = rdtime(), c0 = cyc(), i0 = have_cycle ? rdinstret() : 0;
    yb_go = true;
    thread_join(a);
    thread_join(b);
    u64 t1 = rdtime(), c1 = cyc(), i1 = have_cycle ? rdinstret() : 0;
    u64 sw = sched_nswitch() - s0;
    kprintf("  (yield benchmark performed %lu switches for %lu yields)\n", sw, 2 * N);
    report("yield thread->thread switch", t1 - t0, c1 - c0, i1 - i0, sw);
}

/* --- trap round trip: ebreak into kernelvec and back --- */
static void bench_trap(void) {
    const u64 N = 50000;
    trap_quiet_recover = true;
    push_off();
    MEASURE("trap round trip (ebreak)", N, for (u64 i = 0; i < N; i++) probe_ebreak());
    pop_off();
    MEASURE("SBI ecall round trip", N, for (u64 i = 0; i < N; i++) sbi_probe(SBI_EXT_TIME));
}

/* --- timer interrupt latency --- */
static int cmp_sort(volatile u64 *a, u64 n) {
    for (u64 i = 1; i < n; i++) {
        u64 v = a[i], j = i;
        while (j > 0 && a[j - 1] > v) { a[j] = a[j - 1]; j--; }
        a[j] = v;
    }
    return 0;
}

static void print_latency(const char *label) {
    u64 n = MIN(timer_lat.n, ARRAY_SIZE(timer_lat.samples));
    if (n == 0) {
        kprintf("BENCH timer_latency_%-8s no samples recorded\n", label);
        return;
    }
    cmp_sort(timer_lat.samples, n);
    u64 ns_per_tb = 1000000000UL / timebase_hz;
    u64 p50 = timer_lat.samples[n / 2], p99 = timer_lat.samples[(n * 99) / 100];
    kprintf("BENCH timer_latency_%-8s n=%lu min=%lu p50=%lu p99=%lu max=%lu mean=%lu ns "
            "(resolution %lu ns)\n",
            label, timer_lat.n, timer_lat.min * ns_per_tb, p50 * ns_per_tb, p99 * ns_per_tb,
            timer_lat.max * ns_per_tb, timer_lat.sum * ns_per_tb / timer_lat.n, ns_per_tb);
}

static void bench_timer_latency(void) {
    /* idle: the hart sits in wfi between ticks */
    memset((void *)&timer_lat, 0, sizeof timer_lat);
    timer_lat.recording = true;
    thread_sleep_ticks(200);
    timer_lat.recording = false;
    print_latency("idle");
    /* busy: this thread spins, so the tick interrupts running code */
    memset((void *)&timer_lat, 0, sizeof timer_lat);
    timer_lat.recording = true;
    u64 end = ticks + 200;
    while (ticks < end)
        ;
    timer_lat.recording = false;
    print_latency("busy");
}

/* --- page allocator throughput --- */
static void bench_pmm(void) {
    const u64 N = 20000;
    static u64 pages[20000];
    push_off();
    MEASURE("pmm_alloc_nozero", N, for (u64 i = 0; i < N; i++) pages[i] = pmm_alloc_nozero());
    MEASURE("pmm_free", N, for (u64 i = 0; i < N; i++) pmm_free(pages[i]));
    MEASURE("pmm_alloc (zeroing 4 KiB)", N, for (u64 i = 0; i < N; i++) pages[i] = pmm_alloc());
    for (u64 i = 0; i < N; i++) pmm_free(pages[i]);
    MEASURE("pmm alloc+free pair", 5 * N, for (u64 i = 0; i < 5 * N; i++) pmm_free(pmm_alloc_nozero()));
    pop_off();
}

static void bench_kmalloc(void) {
    const u64 N = 10000;
    static void *p[10000];
    push_off();
    MEASURE("kmalloc(64)+kfree pair", 10 * N, for (u64 i = 0; i < 10 * N; i++) kfree(kmalloc(64)));
    MEASURE("kmalloc(64) batch", N, for (u64 i = 0; i < N; i++) p[i] = kmalloc(64));
    MEASURE("kfree(64) batch", N, for (u64 i = 0; i < N; i++) kfree(p[i]));
    pop_off();
}

void bench_main(void *arg) {
    (void)arg;
    u64 dummy;
    push_off();
    have_cycle = probe_rdcycle(&dummy) == 0;
    pop_off();
    kprintf("\n=== kernel benchmarks (QEMU %s) ===\n",
            bootarg_has(bootinfo.bootargs, "icount") ? "TCG with -icount shift=0" : "TCG");
    kprintf("caveat: timings are for QEMU's emulated RISC-V hart, not real hardware\n");
    kprintf("rdcycle from S-mode: %s; timebase %lu Hz (resolution %lu ns)\n",
            have_cycle ? "available" : "not permitted", timebase_hz, 1000000000UL / timebase_hz);
    bench_swtch();
    bench_yield();
    bench_trap();
    bench_timer_latency();
    bench_pmm();
    bench_kmalloc();
    kprintf("BENCH done\n");
    system_exit(EXIT_OK);
}
