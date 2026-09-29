#!/bin/sh
# Boot the demo, wait for the monitor prompt, type a few commands over the
# UART (they arrive through the PLIC as receive interrupts) and print the
# whole session to stdout.
set -u
QEMU=${QEMU:-qemu-system-riscv64}
KERNEL=${KERNEL:-build/kernel.elf}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkfifo "$TMP/in"
: > "$TMP/log"
(
    while ! grep -q "kmon>" "$TMP/log"; do sleep 0.3; done
    for c in help ps mem ticks stats pf bogus poweroff; do printf '%s\r' "$c"; sleep 0.5; done
    sleep 30
) > "$TMP/in" &
feeder=$!
sh tools/timeout.sh 90 "$QEMU" -machine virt -cpu rv64 -smp 2 -m 128M -nographic -bios default \
    -kernel "$KERNEL" -append "mode=demo" < "$TMP/in" > "$TMP/log" 2>&1
status=$?
kill "$feeder" 2>/dev/null
cat "$TMP/log"
echo "# qemu exit status: $status"
