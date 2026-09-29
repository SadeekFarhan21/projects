/* Kernel entry. OpenSBI jumps here in S-mode with
 *   a0 = hart id, a1 = physical address of the device tree blob,
 *   MMU off, interrupts off.
 * With the SBI HSM extension only the boot hart arrives here; the others stay
 * stopped inside OpenSBI. We still run a lottery so that firmware which
 * releases every hart (older SBI) parks the losers instead of corrupting BSS. */
#include "param.h"

#define SLOT_SIZE ((BOOT_STACK_PAGES + 1) * 4096)

    .section .text.boot, "ax"
    .globl _start
_start:
    csrw sie, zero
    csrw sip, zero

    li t0, NCPU
    bgeu a0, t0, park          /* hart id too large for our tables */

    la t0, boot_lottery
    li t1, 1
    amoswap.w.aq t1, t1, (t0)
    bnez t1, park              /* someone else is the boot hart */

    /* sp = top of this hart's boot stack slot */
    la sp, boot_stacks
    li t0, SLOT_SIZE
    addi t1, a0, 1
    mul t0, t0, t1
    add sp, sp, t0

    /* zero .bss (a0/a1 still hold hartid/dtb) */
    la t0, _bss_start
    la t1, _bss_end
1:  bgeu t0, t1, 2f
    sd zero, 0(t0)
    addi t0, t0, 8
    j 1b
2:
    li s0, 0                   /* terminate the frame-pointer chain */
    li ra, 0
    call kmain
park:
    wfi
    j park

    .section .data
    .balign 4
boot_lottery:                  /* in .data, not .bss, so clearing BSS cannot reset it */
    .word 0

    .section .bootstacks, "aw", @nobits
    .balign 4096
    .globl boot_stacks
boot_stacks:
    .space SLOT_SIZE * NCPU
