#!/bin/bash
# Run the in-kernel benchmarks twice and keep the raw logs in results/:
#   bench-tcg-{1,2,3}.txt  plain TCG, three runs: virtual time follows the host
#                     clock, so the numbers include host scheduling noise
#   bench-icount.txt  -icount shift=0,sleep=off: virtual time advances exactly
#                     1 ns per guest instruction, so results are reproducible
#                     and "ns" there means "guest instructions"
# Neither is a real-hardware timing. The host load is recorded in each file.
set -u
QEMU=${QEMU:-qemu-system-riscv64}
KERNEL=${KERNEL:-build/kernel.elf}
TIMEOUT=${TIMEOUT:-300}
SMP=${SMP:-2}
MEM=${MEM:-128M}
mkdir -p results
BASE="-machine virt -cpu rv64 -smp $SMP -m $MEM -nographic -bios default -kernel $KERNEL"

run() { # file extra-args bootargs
    f=results/$1
    {
        echo "# $(date '+%Y-%m-%d %H:%M:%S') host: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), $(sysctl -n hw.ncpu 2>/dev/null) cpus"
        echo "# host load average before run: $(sysctl -n vm.loadavg 2>/dev/null || uptime)"
        echo "# qemu: $($QEMU --version | head -1)"
        echo "# command: $QEMU $BASE $2 -append \"$3\""
    } > "$f"
    # shellcheck disable=SC2086
    sh tools/timeout.sh "$TIMEOUT" "$QEMU" $BASE $2 -append "$3" < /dev/null >> "$f" 2>&1
    status=$?
    echo "# host load average after run: $(sysctl -n vm.loadavg 2>/dev/null)" >> "$f"
    echo "== $f (exit $status)"
    grep '^BENCH' "$f"
    return $status
}

for i in 1 2 3; do run bench-tcg-$i.txt "" "mode=bench" > /dev/null || exit 1; done
run bench-icount.txt "-icount shift=0,sleep=off" "mode=bench icount" || exit 1
# Determinism check: a second icount run must reproduce every per-op number.
run bench-icount-rerun.txt "-icount shift=0,sleep=off" "mode=bench icount" > /dev/null || exit 1
{
    echo "# raw diff of BENCH lines between bench-icount.txt and bench-icount-rerun.txt"
    diff <(grep '^BENCH' results/bench-icount.txt) <(grep '^BENCH' results/bench-icount-rerun.txt) && echo "(no differences)"
    a=$(grep -o '[0-9]* cycles/op' results/bench-icount.txt | tr '\n' ' ')
    b=$(grep -o '[0-9]* cycles/op' results/bench-icount-rerun.txt | tr '\n' ' ')
    if [ "$a" = "$b" ]; then
        echo "cycles/op columns identical in both runs: $a"
    else
        echo "cycles/op columns differ: run1 [$a] run2 [$b]"
    fi
} > results/icount-determinism.txt
cat results/icount-determinism.txt

# One table: deterministic icount cost next to the spread of the TCG runs.
{
    echo "# per-operation cost. icount: guest instructions (1 instruction = 1 virtual ns)."
    echo "# tcg: wall-clock ns per op inside QEMU, min and max over bench-tcg-1..3.txt"
    echo "# host load averages during the TCG runs:"
    grep -h 'load average before' results/bench-tcg-*.txt | sed 's/^/#   /'
    printf '%-30s %12s %14s %14s\n' benchmark icount_instr tcg_min_ns tcg_max_ns
    grep '^BENCH' results/bench-icount.txt | grep 'ns/op' | while IFS= read -r line; do
        name=$(echo "$line" | sed -E 's/^BENCH (.*[^ ]) +[0-9.]+ ns\/op.*/\1/')
        instr=$(echo "$line" | sed -E 's/.* ([0-9]+) cycles\/op.*/\1/')
        vals=$(grep -h "^BENCH $name " results/bench-tcg-*.txt | sed -E 's/.* ([0-9]+)\.[0-9]+ ns\/op.*/\1/' | sort -n)
        printf '%-30s %12s %14s %14s\n' "$name" "$instr" "$(echo "$vals" | head -1)" "$(echo "$vals" | tail -1)"
    done
    echo "# timer interrupt latency (deadline to C handler), ns"
    grep -h '^BENCH timer' results/bench-icount.txt | sed 's/^BENCH /icount: /'
    grep -h '^BENCH timer' results/bench-tcg-*.txt | sed 's/^BENCH /tcg:    /'
} > results/bench-summary.txt
cat results/bench-summary.txt
