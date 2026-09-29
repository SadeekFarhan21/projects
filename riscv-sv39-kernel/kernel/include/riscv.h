/* RISC-V privileged architecture helpers (S-mode). */
#pragma once
#include "types.h"

#define SSTATUS_SIE (1UL << 1)
#define SSTATUS_SPIE (1UL << 5)
#define SSTATUS_SPP (1UL << 8)
#define SSTATUS_FS (3UL << 13)
#define SSTATUS_SUM (1UL << 18)

#define SIE_SSIE (1UL << 1)
#define SIE_STIE (1UL << 5)
#define SIE_SEIE (1UL << 9)

#define SCAUSE_INTR (1UL << 63)
#define IRQ_S_SOFT 1
#define IRQ_S_TIMER 5
#define IRQ_S_EXT 9

#define EXC_INST_MISALIGNED 0
#define EXC_INST_ACCESS 1
#define EXC_ILLEGAL_INST 2
#define EXC_BREAKPOINT 3
#define EXC_LOAD_MISALIGNED 4
#define EXC_LOAD_ACCESS 5
#define EXC_STORE_MISALIGNED 6
#define EXC_STORE_ACCESS 7
#define EXC_ECALL_U 8
#define EXC_ECALL_S 9
#define EXC_INST_PAGE_FAULT 12
#define EXC_LOAD_PAGE_FAULT 13
#define EXC_STORE_PAGE_FAULT 15

#define SATP_SV39 (8UL << 60)
#define MAKE_SATP(pt) (SATP_SV39 | (((u64)(pt)) >> 12))

#define csr_read(csr)                                           \
    ({                                                          \
        u64 __v;                                                \
        asm volatile("csrr %0, " #csr : "=r"(__v)::"memory");  \
        __v;                                                    \
    })
#define csr_write(csr, val) asm volatile("csrw " #csr ", %0" ::"r"((u64)(val)) : "memory")
#define csr_set(csr, val) asm volatile("csrs " #csr ", %0" ::"r"((u64)(val)) : "memory")
#define csr_clear(csr, val) asm volatile("csrc " #csr ", %0" ::"r"((u64)(val)) : "memory")

static inline void intr_on(void) { csr_set(sstatus, SSTATUS_SIE); }
static inline void intr_off(void) { csr_clear(sstatus, SSTATUS_SIE); }
static inline bool intr_get(void) { return (csr_read(sstatus) & SSTATUS_SIE) != 0; }

static inline u64 rdtime(void) { u64 v; asm volatile("rdtime %0" : "=r"(v)); return v; }
static inline u64 rdcycle(void) { u64 v; asm volatile("rdcycle %0" : "=r"(v)); return v; }
static inline u64 rdinstret(void) { u64 v; asm volatile("rdinstret %0" : "=r"(v)); return v; }

static inline void sfence_vma_all(void) { asm volatile("sfence.vma zero, zero" ::: "memory"); }
static inline void sfence_vma(u64 va) { asm volatile("sfence.vma %0, zero" ::"r"(va) : "memory"); }
static inline void wfi(void) { asm volatile("wfi" ::: "memory"); }
static inline u64 read_fp(void) { return (u64)__builtin_frame_address(0); }

static inline void mmio_write8(u64 addr, u8 v) { *(volatile u8 *)addr = v; }
static inline u8 mmio_read8(u64 addr) { return *(volatile u8 *)addr; }
static inline void mmio_write32(u64 addr, u32 v) { *(volatile u32 *)addr = v; }
static inline u32 mmio_read32(u64 addr) { return *(volatile u32 *)addr; }
