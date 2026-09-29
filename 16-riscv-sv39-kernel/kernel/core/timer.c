/* Periodic timer via the SBI TIME extension. The deadline is kept on a fixed
 * grid (next += interval) so ticks do not drift, and the gap between the
 * deadline and the moment the C handler runs is recorded as interrupt
 * latency when a benchmark asks for it. */
#include "kernel.h"
#include "riscv.h"
#include "cpu.h"
#include "thread.h"
#include "trap.h"

u64 timer_interval, timebase_hz;
extern u64 boot_hartid;
static u64 next_deadline;
struct latency_stats timer_lat;

void timer_init(u64 hz) {
    timebase_hz = hz;
    timer_interval = hz / HZ;
    next_deadline = rdtime() + timer_interval;
    sbi_set_timer(next_deadline);
    csr_set(sie, SIE_STIE);
    kprintf("timer: timebase %lu Hz, tick every %lu timebase units (%d Hz)\n", hz,
            timer_interval, HZ);
}

void timer_interrupt(void) {
    u64 now = rdtime();
    if (timer_lat.recording) {
        u64 lat = now - next_deadline;
        if (timer_lat.n < ARRAY_SIZE(timer_lat.samples)) timer_lat.samples[timer_lat.n] = lat;
        timer_lat.n++;
        timer_lat.sum += lat;
        if (lat < timer_lat.min || timer_lat.n == 1) timer_lat.min = lat;
        if (lat > timer_lat.max) timer_lat.max = lat;
    }
    next_deadline += timer_interval;
    if (next_deadline <= now) next_deadline = now + timer_interval; /* we fell behind */
    sbi_set_timer(next_deadline); /* also clears the pending STIP */
    if ((u64)mycpu()->hartid == boot_hartid) {
        ticks++;
        wakeup((void *)&ticks);
    }
}
