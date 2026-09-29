/* NS16550A UART driver (QEMU virt). Output is polled; input is interrupt
 * driven through the PLIC. Register offsets assume reg-shift 0 (byte regs). */
#include "kernel.h"
#include "riscv.h"
#include "trap.h"

#define RBR 0 /* receive buffer (read) */
#define THR 0 /* transmit holding (write) */
#define IER 1 /* interrupt enable */
#define FCR 2 /* FIFO control (write) */
#define LCR 3 /* line control */
#define LSR 5 /* line status */
#define DLL 0 /* divisor latch low (DLAB=1) */
#define DLM 1 /* divisor latch high (DLAB=1) */

#define LSR_DR 0x01   /* data ready */
#define LSR_THRE 0x20 /* THR empty */

static u64 uart_base;
u64 uart_irqs;

void uart_init(u64 base) {
    uart_base = base;
    mmio_write8(base + IER, 0x00);
    mmio_write8(base + LCR, 0x80); /* DLAB on */
    mmio_write8(base + DLL, 0x03); /* 38400 baud; QEMU ignores it anyway */
    mmio_write8(base + DLM, 0x00);
    mmio_write8(base + LCR, 0x03); /* 8N1, DLAB off */
    mmio_write8(base + FCR, 0x07); /* enable and clear FIFOs */
}

void uart_enable_rx_irq(void) { mmio_write8(uart_base + IER, 0x01); }

void uart_putc(char c) {
    while (!(mmio_read8(uart_base + LSR) & LSR_THRE))
        ;
    mmio_write8(uart_base + THR, (u8)c);
}

void uart_intr(void) {
    uart_irqs++;
    while (mmio_read8(uart_base + LSR) & LSR_DR) console_input((char)mmio_read8(uart_base + RBR));
}
