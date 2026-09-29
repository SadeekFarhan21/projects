/* Per-hart state. tp always holds a pointer to the current hart's struct cpu. */
#pragma once
#include "types.h"
#include "param.h"
#include "asm_offsets.h"

struct thread;

struct cpu {
    struct thread *cur; /* running thread */
    u64 kstack_lo;      /* bounds of the running thread's stack, used by */
    u64 kstack_hi;      /* kernelvec's overflow check and by backtraces */
    u64 emerg_sp;       /* top of the emergency stack */
    u64 onfault;        /* if nonzero, exceptions resume here (probe helpers) */
    int hartid;
    int noff;           /* depth of push_off nesting */
    int intena;         /* were interrupts on before the outermost push_off? */
    struct thread *idle;
    /* updated by trap handlers behind the compiler's back: volatile */
    volatile u64 ntimer, nexternal, nexceptions, nrecovered;
};

_Static_assert(offsetof(struct cpu, cur) == CPU_CUR, "cpu.cur");
_Static_assert(offsetof(struct cpu, kstack_lo) == CPU_KSTACK_LO, "cpu.kstack_lo");
_Static_assert(offsetof(struct cpu, kstack_hi) == CPU_KSTACK_HI, "cpu.kstack_hi");
_Static_assert(offsetof(struct cpu, emerg_sp) == CPU_EMERG_SP, "cpu.emerg_sp");
_Static_assert(offsetof(struct cpu, onfault) == CPU_ONFAULT, "cpu.onfault");

extern struct cpu cpus[NCPU];

static inline struct cpu *mycpu(void) {
    struct cpu *c;
    asm volatile("mv %0, tp" : "=r"(c));
    return c;
}
