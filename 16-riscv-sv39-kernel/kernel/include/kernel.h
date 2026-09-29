/* Core kernel services: printing, panic, strings, SBI. */
#pragma once
#include "types.h"
#include "param.h"

/* lib/printf.c */
int kprintf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
int kvprintf(const char *fmt, va_list ap);
int ksnprintf(char *buf, size_t n, const char *fmt, ...) __attribute__((format(printf, 3, 4)));

/* lib/string.c */
void *memset(void *dst, int c, size_t n);
void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);
int strcmp(const char *a, const char *b);
int strncmp(const char *a, const char *b, size_t n);
size_t strlcpy(char *dst, const char *src, size_t n);
const char *strstr(const char *hay, const char *needle);
bool bootarg_has(const char *args, const char *word);
bool bootarg_value(const char *args, const char *key, char *out, size_t n);

/* core/panic.c */
NORETURN void panic_at(const char *file, int line, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));
#define panic(...) panic_at(__FILE__, __LINE__, __VA_ARGS__)
#define assert(c)                                                  \
    do {                                                           \
        if (!(c)) panic("assertion failed: %s", #c);               \
    } while (0)
extern volatile int panicking;
void backtrace_from(u64 fp, u64 lo, u64 hi);
const char *ksym_lookup(u64 addr, u64 *off);
NORETURN void system_exit(int code);

/* dev/console.c */
void console_putc(char c);
void console_write(const char *s, size_t n);
void console_use_uart(void);
void console_input(char c);
int console_getline(char *buf, int max);
void console_lock_acquire(void);
void console_lock_release(void);
extern u64 console_rx_chars;

/* arch/sbi.c */
struct sbiret {
    long error;
    long value;
};
struct sbiret sbi_call(long ext, long fid, long a0, long a1, long a2);
void sbi_putchar(char c);
void sbi_set_timer(u64 stime);
long sbi_hart_status(u64 hartid);
void sbi_system_reset(u32 type, u32 reason);
void sbi_print_info(void);
long sbi_probe(long ext);
#define SBI_EXT_TIME 0x54494D45
#define SBI_EXT_HSM 0x48534D
#define SBI_EXT_SRST 0x53525354
#define SBI_EXT_DBCN 0x4442434E

/* boot information gathered from the device tree */
#define MAX_REGIONS 8
struct region {
    u64 base, size;
};
struct boot_info {
    u64 dtb_pa, dtb_size;
    struct region mem[MAX_REGIONS];
    int nmem;
    struct region rsv[MAX_REGIONS];
    int nrsv;
    u64 uart_base, uart_size;
    u32 uart_irq;
    u64 plic_base, plic_size;
    u32 plic_ndev;
    u64 test_base;
    u64 timebase_hz;
    int ncpus;
    char bootargs[256];
};
extern struct boot_info bootinfo;
int fdt_parse(u64 dtb_pa, struct boot_info *bi);
