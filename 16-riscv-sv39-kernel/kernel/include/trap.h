/* Trap frame and trap handling. */
#pragma once
#include "types.h"
#include "asm_offsets.h"

struct trapframe {
    u64 ra, sp, gp, tp, t0, t1, t2, s0, s1;
    u64 a0, a1, a2, a3, a4, a5, a6, a7;
    u64 s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
    u64 t3, t4, t5, t6;
    u64 sepc, sstatus, scause, stval;
    u64 pad;
    u64 rec_fp, rec_ra; /* frame record so backtraces cross the trap */
};
_Static_assert(offsetof(struct trapframe, sepc) == TF_SEPC, "tf.sepc");
_Static_assert(offsetof(struct trapframe, stval) == TF_STVAL, "tf.stval");
_Static_assert(offsetof(struct trapframe, rec_fp) == TF_REC_FP, "tf.rec_fp");
_Static_assert(sizeof(struct trapframe) == TF_SIZE, "tf size");

void trap_init_hart(void);
void kernel_trap(struct trapframe *tf);
const char *scause_name(u64 scause);

/* arch/probe.S: touch memory or run an instruction, returning -1 if it traps */
int probe_read64(u64 addr, u64 *out);
int probe_write64(u64 addr, u64 val);
int probe_exec(u64 addr);
int probe_illegal(void);
int probe_ebreak(void);
int probe_rdcycle(u64 *out);
/* C wrappers that disable interrupts around the probe */
int try_read(u64 addr, u64 *out);
int try_write(u64 addr, u64 val);
int try_exec(u64 addr);
extern bool trap_quiet_recover; /* print one-line reports for recovered faults */
extern u64 last_recovered_scause, last_recovered_stval;

/* core/timer.c */
void timer_init(u64 timebase_hz);
void timer_interrupt(void);
extern u64 timer_interval;
extern u64 timebase_hz;
/* Written by the timer interrupt, read by a thread: every field is volatile,
 * otherwise the compiler may drop "recording = true" as a dead store. */
struct latency_stats {
    volatile u64 n, sum, min, max;
    volatile u64 samples[256];
    volatile bool recording;
};
extern struct latency_stats timer_lat;

/* dev */
void uart_init(u64 base);
void uart_putc(char c);
void uart_enable_rx_irq(void);
void uart_intr(void);
void plic_init(u64 base, int hart);
void plic_enable(u32 irq);
u32 plic_claim(void);
void plic_complete(u32 irq);
extern u64 uart_irqs;
