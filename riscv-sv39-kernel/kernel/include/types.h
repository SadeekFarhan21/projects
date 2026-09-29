/* Basic fixed-width types for a freestanding LP64 kernel. */
#pragma once

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long u64;
typedef signed char i8;
typedef short i16;
typedef int i32;
typedef long i64;
typedef unsigned long size_t;
typedef long ssize_t;
typedef unsigned long uintptr_t;
typedef _Bool bool;

#define true 1
#define false 0
#define NULL ((void *)0)

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((u64)(a) - 1))
#define ALIGN_DOWN(x, a) ((x) & ~((u64)(a) - 1))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define offsetof(t, m) __builtin_offsetof(t, m)

#define NORETURN __attribute__((noreturn))
#define UNUSED __attribute__((unused))

typedef __builtin_va_list va_list;
#define va_start(ap, last) __builtin_va_start(ap, last)
#define va_arg(ap, t) __builtin_va_arg(ap, t)
#define va_end(ap) __builtin_va_end(ap)

/* Error codes returned as negative ints. */
#define E_NOMEM 1
#define E_INVAL 2
#define E_DOUBLEFREE 3
#define E_EXIST 4
#define E_NOENT 5
#define E_BADPTR 6
