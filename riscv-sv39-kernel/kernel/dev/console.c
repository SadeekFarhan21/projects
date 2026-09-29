/* Console: early output goes through SBI, then switches to the UART driver
 * once the device tree has told us where the UART is. Input arrives from the
 * UART interrupt into a ring buffer; readers sleep until a line is ready. */
#include "kernel.h"
#include "spinlock.h"
#include "thread.h"
#include "trap.h"

static struct spinlock cons_lock = {.name = "console"};
static bool use_uart;
u64 console_rx_chars;

#define INBUF 128
static struct {
    struct spinlock lk;
    char buf[INBUF];
    u64 r, w; /* read and write indices, w - r = chars buffered */
    u64 lines; /* completed lines available */
} in = {.lk = {.name = "cons_in"}};

void console_use_uart(void) { use_uart = true; }

void console_putc(char c) {
    if (c == '\n') console_putc('\r');
    if (use_uart) uart_putc(c);
    else sbi_putchar(c);
}

void console_write(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) console_putc(s[i]);
}

/* Once we are panicking the lock may be held by the code that crashed. */
void console_lock_acquire(void) {
    if (!panicking) spin_lock(&cons_lock);
}
void console_lock_release(void) {
    if (!panicking && spin_holding(&cons_lock)) spin_unlock(&cons_lock);
}

/* Called from the UART interrupt handler for every received byte. */
void console_input(char c) {
    console_rx_chars++;
    spin_lock(&in.lk);
    if (c == '\r') c = '\n';
    if (c == 0x7f || c == '\b') {
        if (in.w > in.r && in.buf[(in.w - 1) % INBUF] != '\n') {
            in.w--;
            console_write("\b \b", 3); /* echo erase */
        }
    } else if (in.w - in.r < INBUF - 1) {
        in.buf[in.w++ % INBUF] = c;
        console_putc(c); /* echo */
        if (c == '\n') {
            in.lines++;
            wakeup(&in);
        }
    }
    spin_unlock(&in.lk);
}

/* Block until a full line is typed. Returns its length without the newline. */
int console_getline(char *buf, int max) {
    spin_lock(&in.lk);
    while (in.lines == 0) sleep_on(&in, &in.lk);
    int n = 0;
    while (in.r < in.w) {
        char c = in.buf[in.r++ % INBUF];
        if (c == '\n') break;
        if (n < max - 1) buf[n++] = c;
    }
    in.lines--;
    buf[n] = 0;
    spin_unlock(&in.lk);
    return n;
}
