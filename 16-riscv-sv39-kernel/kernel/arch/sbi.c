/* SBI calls into OpenSBI (M-mode). Calling convention: a7 = extension id,
 * a6 = function id, a0..a5 = args; returns error in a0 and value in a1. */
#include "kernel.h"

struct sbiret sbi_call(long ext, long fid, long arg0, long arg1, long arg2) {
    register long a0 asm("a0") = arg0;
    register long a1 asm("a1") = arg1;
    register long a2 asm("a2") = arg2;
    register long a6 asm("a6") = fid;
    register long a7 asm("a7") = ext;
    asm volatile("ecall" : "+r"(a0), "+r"(a1) : "r"(a2), "r"(a6), "r"(a7) : "memory");
    return (struct sbiret){a0, a1};
}

static int have_dbcn = -1;

long sbi_probe(long ext) { return sbi_call(0x10, 3, ext, 0, 0).value; }

void sbi_putchar(char c) {
    if (have_dbcn < 0) have_dbcn = sbi_probe(SBI_EXT_DBCN) != 0;
    if (have_dbcn)
        sbi_call(SBI_EXT_DBCN, 2, (u8)c, 0, 0); /* debug console write byte */
    else
        sbi_call(0x01, 0, c, 0, 0);             /* legacy console putchar */
}

void sbi_set_timer(u64 stime) { sbi_call(SBI_EXT_TIME, 0, (long)stime, 0, 0); }

long sbi_hart_status(u64 hartid) {
    struct sbiret r = sbi_call(SBI_EXT_HSM, 2, (long)hartid, 0, 0);
    return r.error ? r.error : r.value;
}

void sbi_system_reset(u32 type, u32 reason) {
    sbi_call(SBI_EXT_SRST, 0, type, reason, 0);
}

void sbi_print_info(void) {
    long spec = sbi_call(0x10, 0, 0, 0, 0).value;
    long impl = sbi_call(0x10, 1, 0, 0, 0).value;
    long ver = sbi_call(0x10, 2, 0, 0, 0).value;
    kprintf("sbi: spec v%ld.%ld, impl %s (id %ld) version 0x%lx\n", (spec >> 24) & 0x7f,
            spec & 0xffffff, impl == 1 ? "OpenSBI" : "unknown", impl, ver);
    kprintf("sbi: extensions TIME=%ld HSM=%ld SRST=%ld DBCN=%ld\n", sbi_probe(SBI_EXT_TIME),
            sbi_probe(SBI_EXT_HSM), sbi_probe(SBI_EXT_SRST), sbi_probe(SBI_EXT_DBCN));
}
