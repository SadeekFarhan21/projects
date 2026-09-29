#!/usr/bin/env bash
# Mutation experiment: plant one deliberate bug at a time in the
# compiler and record which part of the test suite catches it.
# Each mutation is a single sed substitution; the original file is
# restored afterwards (also on error or Ctrl-C).
#
# Usage: tools/mutation.sh > results/mutation.txt
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(opam env --switch=. --set-switch)"
# planted bugs often turn loops into infinite loops: kill each test
# executable after 5 seconds (the runners also cap parallelism at 2)
export KITE_TEST_TIMEOUT=5

restore() { for f in lib/*.ml.orig; do [ -e "$f" ] && mv "$f" "${f%.orig}"; done; }
# Restore on exit, and on INT/TERM restore and then really exit: a bare
# `trap restore TERM` runs the handler and then carries on with the loop.
trap restore EXIT
trap 'restore; exit 143' INT TERM

# name | file | sed expression
MUTATIONS=(
  "fold >> as a logical shift|lib/opt.ml|s/Some (Int64.shift_right a (Int64.to_int b land 63))/Some (Int64.shift_right_logical a (Int64.to_int b land 63))/"
  "swap msub operands in remainder|lib/codegen.ml|s/out \"msub %s, x16, %s, %s\" rd rb ra/out \"msub %s, x16, %s, %s\" rd ra rb/"
  "linear scan frees a register one position early|lib/regalloc.ml|s/(fun (e', _, _) -> e' < s)/(fun (e', _, _) -> e' <= s)/"
  "live intervals ignore live-out (loop-carried values)|lib/regalloc.ml|s/Opt.IS.iter (fun t -> touch t bend) (Hashtbl.find live_out b.label))/ignore bend)/"
  "emit b.lt for <= comparisons|lib/codegen.ml|s/| Lt -> \"lt\" | Le -> \"le\"/| Lt -> \"lt\" | Le -> \"lt\"/"
  "bounds check uses signed compare|lib/lower.ml|s/emit fb (I.Bin (I.Ltu, c, idx, I.T len))/emit fb (I.Bin (I.Lt, c, idx, I.T len))/"
  "drop x+0 identity guard (x+c folds to x)|lib/opt.ml|s/| (Add | Sub | Or | Xor | Shl | Shr), x, C 0L -> Mov (d, x)/| (Add | Sub | Or | Xor | Shl | Shr), x, C _ -> Mov (d, x)/"
)

echo "Mutation experiment: one planted bug at a time"
echo "for each planted bug: differential program checks (275 total), 60-program fuzz"
echo "(180 native runs), and unit tests. A hang counts as a failure (5 s timeout)"
echo "and is also reported separately, because a heavily loaded machine can"
echo "push an innocent program past the limit."
echo
for m in "${MUTATIONS[@]}"; do
  IFS='|' read -r name file expr <<<"$m"
  cp "$file" "$file.orig"
  sed -i '' "$expr" "$file"
  if cmp -s "$file" "$file.orig"; then
    echo "$name: SED DID NOT APPLY"
    restore
    continue
  fi
  if ! dune build 2>/dev/null; then
    echo "$name: does not build"
    restore
    continue
  fi
  progs=$(nice ./_build/default/tests/run_programs.exe tests/programs tests/fail 2>&1 | grep -E '^passed' || true)
  fuzz=$(nice ./_build/default/tests/fuzz.exe 60 1 "${TMPDIR:-/tmp}/kite-mutation-fuzz" 2>&1 | grep -E '^fuzz:' | sed 's/.*levels, //' || true)
  unit=$( (cd _build/default/tests && ./test_unit.exe 2>&1) | grep -E 'Successful|failure' | head -1 || true)
  echo "$name"
  echo "    programs: $progs"
  echo "    fuzz:     $fuzz"
  echo "    unit:     $unit"
  restore
done
rm -rf "${TMPDIR:-/tmp}/kite-mutation-fuzz"
dune build 2>/dev/null
