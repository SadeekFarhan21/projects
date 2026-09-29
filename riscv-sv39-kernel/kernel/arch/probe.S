/* Fault probes. Each helper stores a recovery address in cpu->onfault, does
 * one risky operation and clears onfault. If the operation traps, kernel_trap
 * sees onfault != 0, clears it and resumes at the recovery label, which
 * returns -1. Callers must have interrupts off (see try_* in trap.c). */
#include "asm_offsets.h"

    .section .text

    .globl probe_read64        /* int probe_read64(u64 addr, u64 *out) */
probe_read64:
    la t0, 1f
    sd t0, CPU_ONFAULT(tp)
    ld t1, 0(a0)
    sd zero, CPU_ONFAULT(tp)
    sd t1, 0(a1)
    li a0, 0
    ret
1:  li a0, -1
    ret

    .globl probe_write64       /* int probe_write64(u64 addr, u64 val) */
probe_write64:
    la t0, 1f
    sd t0, CPU_ONFAULT(tp)
    sd a1, 0(a0)
    sd zero, CPU_ONFAULT(tp)
    li a0, 0
    ret
1:  li a0, -1
    ret

    .globl probe_exec          /* int probe_exec(u64 addr): call addr */
probe_exec:
    mv t2, ra
    la t0, 1f
    sd t0, CPU_ONFAULT(tp)
    jalr ra, 0(a0)
    sd zero, CPU_ONFAULT(tp)
    mv ra, t2
    li a0, 0
    ret
1:  mv ra, t2
    li a0, -1
    ret

    .globl probe_illegal       /* int probe_illegal(void) */
probe_illegal:
    la t0, 1f
    sd t0, CPU_ONFAULT(tp)
    .4byte 0xc0001073          /* csrrw zero, cycle, zero: write to read-only CSR */
    sd zero, CPU_ONFAULT(tp)
    li a0, 0
    ret
1:  li a0, -1
    ret

    .globl probe_ebreak        /* int probe_ebreak(void) */
probe_ebreak:
    la t0, 1f
    sd t0, CPU_ONFAULT(tp)
    ebreak
    sd zero, CPU_ONFAULT(tp)
    li a0, 0
    ret
1:  li a0, -1
    ret

    .globl probe_rdcycle       /* int probe_rdcycle(u64 *out) */
probe_rdcycle:
    la t0, 1f
    sd t0, CPU_ONFAULT(tp)
    rdcycle t1
    sd zero, CPU_ONFAULT(tp)
    sd t1, 0(a0)
    li a0, 0
    ret
1:  li a0, -1
    ret
