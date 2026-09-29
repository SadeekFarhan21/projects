#!/bin/sh
# Headless kernel tests. Boots QEMU several times:
#   1. the self-test suite under plain TCG (what `make run` uses)
#   2. the same suite under -icount (deterministic virtual time)
#   3. four deliberate crashes, checking the report text and exit code 3
# Each boot has a timeout. Logs go to $OUT (build/test-logs by default,
# results/ when run through `make results`). Exit status 0 = all passed.
set -u
QEMU=${QEMU:-qemu-system-riscv64}
KERNEL=${KERNEL:-build/kernel.elf}
TIMEOUT=${TIMEOUT:-120}
SMP=${SMP:-2}
MEM=${MEM:-128M}
OUT=${OUT:-build/test-logs}
mkdir -p "$OUT"
BASE="-machine virt -cpu rv64 -smp $SMP -m $MEM -nographic -bios default -kernel $KERNEL"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
failures=0

# boot NAME EXPECTED_EXIT "EXTRA QEMU ARGS" "BOOTARGS" [INPUT] -- then patterns on stdin
boot() {
    name=$1 want=$2 extra=$3 args=$4 input=${5:-}
    log="$OUT/$name.txt"
    : > "$log"
    mkfifo "$TMP/in.$name"
    # Feed UART input only after the kernel has initialized the UART:
    # uart_init() resets the receive FIFO, so earlier bytes would be lost.
    (
        if [ -n "$input" ]; then
            while ! grep -q "=== kernel self-test" "$log" 2>/dev/null; do sleep 0.2; done
            printf '%s' "$input"
        fi
        sleep "$TIMEOUT"
    ) > "$TMP/in.$name" &
    feeder=$!
    start=$(date +%s)
    # shellcheck disable=SC2086
    sh tools/timeout.sh "$TIMEOUT" "$QEMU" $BASE $extra -append "$args" < "$TMP/in.$name" > "$log" 2>&1
    status=$?
    kill "$feeder" 2>/dev/null
    wait "$feeder" 2>/dev/null
    secs=$(( $(date +%s) - start ))
    ok=1
    [ "$status" = "$want" ] || ok=0
    missing=""
    while IFS= read -r pat; do
        [ -z "$pat" ] && continue
        grep -qF -- "$pat" "$log" || { ok=0; missing="$missing [$pat]"; }
    done
    if [ $ok = 1 ]; then
        printf 'PASS  %-22s exit=%s (%ss)  log: %s\n' "$name" "$status" "$secs" "$log"
    else
        printf 'FAIL  %-22s exit=%s want=%s (%ss)  log: %s\n' "$name" "$status" "$want" "$secs" "$log"
        [ -n "$missing" ] && printf '      missing output:%s\n' "$missing"
        [ "$status" = 124 ] || [ "$status" = 142 ] && echo "      (QEMU was killed by the ${TIMEOUT}s timeout)"
        failures=$((failures + 1))
    fi
}

echo "== self-test suite =="
boot selftest-tcg 0 "" "mode=test rxtest" "ping
" <<'P'
ALL TESTS PASSED
received "ping"
P
grep -E '^\[(PASS|FAIL)\]' "$OUT/selftest-tcg.txt"
boot selftest-icount 0 "-icount shift=0,sleep=off" "mode=test" <<'P'
ALL TESTS PASSED
P

echo "== crash reports (expect exit code 3) =="
boot crash-panic 3 "" "crash=panic" <<'P'
KERNEL PANIC at kernel/test/crash.c
deliberate panic
crash_leaf+
crash_middle+
crash_outer+
P
boot crash-pagefault 3 "" "crash=pagefault" <<'P'
store/AMO page fault (scause=15)
stval   : 0x0000000000000010
invalid: no mapping for this address
probably a NULL pointer
crash_middle+
P
boot crash-illegal 3 "" "crash=illegal" <<'P'
illegal instruction (scause=2)
instr   : 0xc0001073
crash_leaf+
P
boot crash-stackoverflow 3 "" "crash=stackoverflow" <<'P'
KERNEL STACK OVERFLOW
thread  : crasher
recurse+
same frame repeated
P

if [ $failures = 0 ]; then
    echo "make test: ALL PASSED"
else
    echo "make test: $failures FAILED"
    exit 1
fi
