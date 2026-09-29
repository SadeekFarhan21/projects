/* Supervisor trap vector (stvec, direct mode).
 * All traps in v0 come from S-mode, so the frame is pushed on the current
 * kernel stack. That is what lets the timer handler call yield(): the
 * interrupted thread's registers stay on its own stack while others run. */
#include "asm_offsets.h"

    .section .text
    .globl kernelvec
    .balign 4
kernelvec:
    /* Stack overflow check before touching memory: if sp is within one
     * frame of the bottom of the current stack, pushing the frame would
     * fault again forever. Switch to the per-hart emergency stack instead. */
    csrw sscratch, t0
    ld t0, CPU_KSTACK_LO(tp)
    addi t0, t0, TF_SIZE
    bltu sp, t0, kstack_overflow
    csrr t0, sscratch

    addi sp, sp, -TF_SIZE
    sd x1, 0(sp)
    /* x2 (sp) stored below once computed */
    sd x3, 16(sp)
    sd x4, 24(sp)
    sd x5, 32(sp)
    sd x6, 40(sp)
    sd x7, 48(sp)
    sd x8, 56(sp)
    sd x9, 64(sp)
    sd x10, 72(sp)
    sd x11, 80(sp)
    sd x12, 88(sp)
    sd x13, 96(sp)
    sd x14, 104(sp)
    sd x15, 112(sp)
    sd x16, 120(sp)
    sd x17, 128(sp)
    sd x18, 136(sp)
    sd x19, 144(sp)
    sd x20, 152(sp)
    sd x21, 160(sp)
    sd x22, 168(sp)
    sd x23, 176(sp)
    sd x24, 184(sp)
    sd x25, 192(sp)
    sd x26, 200(sp)
    sd x27, 208(sp)
    sd x28, 216(sp)
    sd x29, 224(sp)
    sd x30, 232(sp)
    sd x31, 240(sp)
    addi t0, sp, TF_SIZE
    sd t0, 8(sp)               /* the interrupted sp */
    csrr t0, sepc
    sd t0, TF_SEPC(sp)
    sd t0, TF_REC_RA(sp)       /* frame record: "return address" = sepc */
    csrr t0, sstatus
    sd t0, TF_SSTATUS(sp)
    csrr t0, scause
    sd t0, TF_SCAUSE(sp)
    csrr t0, stval
    sd t0, TF_STVAL(sp)
    sd s0, TF_REC_FP(sp)       /* frame record: previous fp */
    addi s0, sp, TF_SIZE       /* fp for backtraces: points just above the record */

    mv a0, sp
    call kernel_trap

    /* kernel_trap may have switched threads and come back. Interrupts must be
     * off before sstatus/sepc are reloaded. */
    csrci sstatus, 2
    ld t0, TF_SEPC(sp)
    csrw sepc, t0
    ld t0, TF_SSTATUS(sp)
    csrw sstatus, t0

    ld x1, 0(sp)
    ld x3, 16(sp)
    /* x4 (tp) is not restored: it names the hart, not the thread. */
    ld x5, 32(sp)
    ld x6, 40(sp)
    ld x7, 48(sp)
    ld x8, 56(sp)
    ld x9, 64(sp)
    ld x10, 72(sp)
    ld x11, 80(sp)
    ld x12, 88(sp)
    ld x13, 96(sp)
    ld x14, 104(sp)
    ld x15, 112(sp)
    ld x16, 120(sp)
    ld x17, 128(sp)
    ld x18, 136(sp)
    ld x19, 144(sp)
    ld x20, 152(sp)
    ld x21, 160(sp)
    ld x22, 168(sp)
    ld x23, 176(sp)
    ld x24, 184(sp)
    ld x25, 192(sp)
    ld x26, 200(sp)
    ld x27, 208(sp)
    ld x28, 216(sp)
    ld x29, 224(sp)
    ld x30, 232(sp)
    ld x31, 240(sp)
    addi sp, sp, TF_SIZE
    sret

kstack_overflow:
    mv t0, sp
    ld sp, CPU_EMERG_SP(tp)
    mv a3, t0                  /* faulting sp */
    csrr a0, sepc
    csrr a1, stval
    csrr a2, scause
    mv a4, s0                  /* fp of the overflowing code, for a backtrace */
    li s0, 0
    call kstack_overflow_panic
1:  j 1b
