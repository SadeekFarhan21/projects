/* Compile-time kernel limits. Values chosen for a small teaching kernel. */
#pragma once

#define NCPU 8                /* max harts the data structures are sized for */
#define NTHREAD 32            /* max kernel threads (static table) */
#define PGSIZE 4096UL
#define PGSHIFT 12
#define KSTACK_PAGES 4        /* 16 KiB per kernel thread stack */
#define BOOT_STACK_PAGES 4    /* 16 KiB per hart boot stack */
#define EMERG_STACK_SIZE 4096 /* per-hart stack used only to report stack overflow */
#define HZ 100                /* timer interrupts per second */
#define QUANTUM_TICKS 1       /* ticks a thread runs before preemption */
#define KERNEL_BASE 0x80200000UL

/* Sv39: we only use the lower half of the 39-bit space (bit 38 clear). */
#define MAXVA (1UL << 38)
/* Kernel thread stacks live at high virtual addresses with a guard page each. */
#define KSTACK_REGION 0x3F00000000UL
#define KSTACK_SLOT_SIZE ((KSTACK_PAGES + 1) * PGSIZE)
#define KSTACK_LO(slot) (KSTACK_REGION + (u64)(slot) * KSTACK_SLOT_SIZE + PGSIZE)
#define KSTACK_HI(slot) (KSTACK_REGION + ((u64)(slot) + 1) * KSTACK_SLOT_SIZE)
/* Scratch VA window used by the page-table self-test. */
#define TEST_VA 0x2000000000UL

/* Exit codes reported to the host through the QEMU test device. */
#define EXIT_OK 0
#define EXIT_TESTFAIL 1
#define EXIT_PANIC 3
