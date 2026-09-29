/* Offsets shared between C structs and assembly. C code static_asserts them. */
#pragma once

/* struct cpu */
#define CPU_CUR 0
#define CPU_KSTACK_LO 8
#define CPU_KSTACK_HI 16
#define CPU_EMERG_SP 24
#define CPU_ONFAULT 32

/* struct trapframe: x1..x31 at (n-1)*8, then CSRs, then a frame record */
#define TF_SEPC 248
#define TF_SSTATUS 256
#define TF_SCAUSE 264
#define TF_STVAL 272
#define TF_REC_FP 288
#define TF_REC_RA 296
#define TF_SIZE 304
