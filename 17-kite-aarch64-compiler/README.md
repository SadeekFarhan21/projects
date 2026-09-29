# Kite: a compiler from scratch

Kite is a small statically typed language, and `kitec` is its compiler, written in OCaml from the lexer to the register allocator. It compiles `.kite` files to native AArch64 assembly for macOS, which Apple's `cc` assembles and links against a 130-line C runtime. A tree-walking interpreter over the typed AST serves as the reference implementation, and every test program must behave identically under the interpreter and under native code at three optimisation levels.

```kite
struct Account {
    owner: string,
    balance: int,
}

fn make_scaler(k: int) -> fn(int) -> int {
    return fn(x: int) -> int { return x * k; };
}

fn main() {
    let acct = Account { owner: "farhan", balance: 10 };
    acct.balance = acct.balance + 32;
    println(acct.owner + " has " + to_str(acct.balance));

    let triple = make_scaler(3);
    var total = 0;
    for i in 0..5 {
        total = total + triple(i);
    }
    println(total);                         // 30
    println(9223372036854775807 + 1);       // wraps: -9223372036854775808
}
```

The language has 64-bit integers with defined wrapping, booleans, immutable strings, heap-allocated arrays and structs (reference semantics), first-class functions and closures, `if`/`while`/`for`, recursion, and local type inference (`let x = ...` needs no annotation). The full definition, with an EBNF grammar, typing rules and evaluation semantics, is in [SPEC.md](SPEC.md). How the compiler works is in [DESIGN.md](DESIGN.md), and the story of building it is in [DEVLOG.md](DEVLOG.md).

## Pipeline

```
source -> lexer -> parser -> type checker -> lowering -> optimiser -> linear scan -> AArch64 asm -> cc
                     |            |           (3-address IR,  (fold, propagate,  (x19-x28)
                pretty-printer    +-> interpreter (test oracle)   basic blocks)     DCE, CFG, coalesce)
```

## Setup

Requirements: macOS on Apple silicon, Homebrew, and the Xcode command line tools (for `cc`).

```sh
tools/setup.sh                                   # idempotent: opam, local OCaml 5.2.1 switch, dune, alcotest
eval $(opam env --switch=. --set-switch)         # in every new shell
```

The setup script installs opam with Homebrew if needed, runs `opam init --bare --disable-sandboxing` once, creates a local switch in `_opam/`, and installs `dune` and `alcotest`. The parser is hand-written, so Menhir is not needed.

## Build

```sh
dune build                     # builds _build/default/bin/main.exe (the kitec driver)
alias kitec=$PWD/_build/default/bin/main.exe
```

## Compile and run a program

```sh
kitec build examples/tour.kite -o tour     # default -O2
./tour
kitec run examples/tour.kite               # the reference interpreter
```

Other commands:

```sh
kitec check  FILE                  # parse and type-check only
kitec fmt    FILE                  # pretty-print canonical source
kitec tokens FILE                  # dump the token stream with positions
kitec ir     FILE [-O0|-O1]        # dump the IR before or after optimisation
kitec asm    FILE [-O0|-O1|-O2]    # print the generated assembly
kitec build  FILE [-o EXE] [-O0|-O1|-O2] [--keep-asm]
```

The optimisation levels are:

| level | IR optimisation | registers |
|---|---|---|
| `-O0` | none | every temp in a stack slot |
| `-O1` | constant folding and propagation, CFG simplification, DCE, copy coalescing | every temp in a stack slot |
| `-O2` (default) | same as `-O1` | linear-scan allocation over x19..x28 |

Errors are reported with a position and a caret:

```
$ kitec build tests/fail/lambda_wrong_type.kite
tests/fail/lambda_wrong_type.kite:6:19: type error: argument 1: expected fn(int) -> int, found fn(int) -> bool
   |
 6 |     println(apply(fn(x: int) -> bool { return x > 0; }));
   |                   ^
```

A runtime error (bounds, division by zero, failed assert) prints `runtime error: ...` to stderr and exits with status 101.

## Test

```sh
dune test
```

This runs three things:

1. Unit tests (alcotest, `tests/test_unit.ml`): the lexer, parser precedence and associativity, the pretty-printer round trip over every test program, and the type checker's inference and error messages.
2. The differential runner (`tests/run_programs.ml`): every program in `tests/programs/` (60 of them) runs in the interpreter and as native executables at `-O0`, `-O1` and `-O2`; all four transcripts (stdout, and exit status plus stderr when non-trivial) must equal the checked-in `.out` file. Every file in `tests/fail/` (35 of them) must fail to compile with the exact diagnostic written in its first line.
3. A 60-program random differential test (`tests/fuzz.ml`): generated programs full of wrapping arithmetic, shifts, division, nested control flow, calls and closures must print the same thing in the interpreter and at every optimisation level.

For a bigger fuzzing run: `_build/default/tests/fuzz.exe 500 2026`.

## Benchmarks

```sh
python3 bench/run.py 10        # 10 timed runs per configuration, after a warm-up run
```

Four compute-heavy programs (`bench/*.kite`) are compiled at `-O0`, `-O1` and `-O2`, and their C twins (`bench/*.c`) with `cc -O0` and `cc -O2`. The harness checks that all five print the same answer, then records wall-clock times in `results/bench_raw.csv` and a summary in `results/bench_summary.txt`. It also writes static IR sizes before and after optimisation to `results/ir_counts.txt`.

BENCH_TABLE_PLACEHOLDER

## Milestones

| milestone | status |
|---|---|
| Lexer with positions and caret diagnostics | done |
| Parser (precedence climbing) and round-tripping pretty-printer | done |
| Name resolution and type checker with local inference | done |
| Closures (capture by value of immutable bindings) | done |
| Three-address IR in basic blocks with a CFG | done |
| Optimisations: constant folding and propagation, DCE, CFG simplification, copy coalescing | done |
| AArch64 code generation for macOS (Apple arm64 ABI) | done |
| Linear-scan register allocation | done |
| Reference interpreter, differential tests, fuzzer | done |
| Benchmarks against `cc -O0` and `cc -O2` | done |
| SSA construction, SCCP, GVN, LICM | later |
| Graph-colouring register allocation | later |
| Garbage collection | later |
| Generics or algebraic data types with pattern matching | later |
| RISC-V backend (to run on the project 16 OS) | later |
| REPL or language server | later |
| Self-hosting | later |

### Roadmap notes

- SSA would replace the "single definition dominates its uses" argument that global constant propagation currently relies on, and make SCCP, GVN and LICM straightforward. LICM matters most for the benchmarks: array lengths and `i * n` are recomputed inside inner loops today.
- Graph colouring (or a better linear scan with interval splitting) would let caller-saved registers hold values that do not live across calls, removing most of the callee-saved save/restore traffic.
- Garbage collection: every value is one word and object layouts are uniform, so a precise mark-sweep collector needs only stack maps saying which slots and registers hold pointers.
- ADTs with pattern matching would be the natural next language feature; they fit the one-word representation as a pointer to `[tag][fields...]`.
- A RISC-V backend mostly means a second `codegen` module: the IR has no AArch64 assumptions besides the 8-argument limit.
- Self-hosting would need at least ADTs, a growable buffer type and file I/O in the runtime.

## Known issues and limits

- Functions and closures take at most 8 parameters (no stack-passed arguments).
- Memory is never freed.
- Native stack overflow from very deep recursion is not detected: the process dies with a signal instead of a runtime error. The interpreter has a different depth limit. Tests stay at depth 10000.
- Strings are byte strings; there is no character type and `substr` is the only way to take them apart.
- A closure cannot call itself (it has no name); recursion goes through top-level functions.
- The pretty-printer drops comments.

## Layout

```
bin/main.ml          kitec command-line driver
lib/diag.ml          positions and error rendering
lib/lexer.ml         tokens
lib/parser.ml        recursive-descent parser
lib/ast.ml           AST and types
lib/pretty.ml        pretty-printer
lib/typecheck.ml     name resolution, type inference, captures
lib/interp.ml        reference interpreter
lib/ir.ml            three-address IR
lib/lower.ml         typed AST -> IR
lib/opt.ml           optimisation passes and liveness
lib/regalloc.ml      linear-scan register allocation
lib/codegen.ml       AArch64 assembly emission
lib/driver.ml        pipeline glue, calls cc
runtime/kite_rt.c    C runtime (allocation, printing, strings, errors)
tests/               unit tests, differential runner, fuzzer, programs/, fail/
bench/               benchmark programs in Kite and C, harness
results/             raw benchmark, fuzzing and mutation results
tools/setup.sh       toolchain setup
tools/mutation.sh    planted-bug experiment for the test suite
examples/tour.kite   a short tour of the language
```
