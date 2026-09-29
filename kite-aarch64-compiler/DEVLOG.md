# Building a compiler from scratch: Kite

### What I wanted to build

I wanted a compiler that goes all the way down: from text I designed to machine code that the M4 Pro in my laptop runs directly, with nothing borrowed in between. No LLVM, no parser generator, no existing IR. The point was to understand each layer well enough to write it, and to be able to show that it is correct rather than just that it runs.

The pieces I set out to build, in order, were a lexer with real error messages, a parser and a pretty-printer that round-trips, a type checker with local inference, a lowering to my own three-address IR, a few optimisation passes whose effect I could measure, an AArch64 backend that follows Apple's ABI, and a tree-walking interpreter to act as the test oracle. The source language is my own. It is called Kite (`.kite` files) and it is deliberately small but not a toy: 64-bit integers with defined overflow, booleans, strings, arrays, structs, first-class functions with closures, `if`/`while`/`for`, recursion, and `let x = ...` without annotations.

For the one optional feature I picked closures over generics and algebraic data types. First-class functions were already required, and without capture they are just function pointers. Closures also touch the most interesting parts of the compiler: capture analysis in the type checker, lambda lifting in the lowering, and an indirect calling convention in the backend.

### Theory

A few ideas carried the whole project.

Precedence climbing. A binary-operator grammar written as one rule per precedence level is correct but tedious. Precedence climbing parses `lhs (op rhs)*` in a loop and recurses with `min_prec = prec(op) + 1` for the right operand, which gives left associativity for free. A comparison is parsed once and then refused if another comparison follows, which makes them non-associative.

Local type inference without unification. If every function and closure parameter is annotated and there is no polymorphism, the type of every expression is determined by its children. Inference is then a bottom-up walk and `let x = e` just takes the type of `e`. The cost is that an empty array literal has no type, so Kite does not have one.

Closure conversion. A closure is a pair of code and environment. The compiler lifts every closure body into a top-level function that takes the environment as an extra argument, and turns the closure expression into an allocation that stores the code pointer and the captured values. Kite captures by value and forbids capturing `var` bindings, so copying a captured variable is indistinguishable from sharing it and no variable ever needs to be boxed.

Three-address code and dataflow. The IR is a control-flow graph of basic blocks whose instructions have at most one operator. Liveness is the classic backward dataflow problem: `in(b) = use(b) ∪ (out(b) - def(b))` and `out(b)` is the union of the successors' `in`, iterated to a fixed point. Dead code elimination and the register allocator both run on it.

Linear-scan register allocation (Poletto and Sarkar, 1999). Give each temp one interval from its first to its last position in a linear order of the code, widened to cover any block where it is live on entry or exit. Walk intervals by start, free registers whose interval has ended, and when none are free spill whichever interval ends furthest away. It is not optimal, but it is linear time and easy to get right.

The AArch64 procedure call standard, Apple flavour. Arguments in `x0` to `x7`, result in `x0`, `x19` to `x28` preserved by the callee, `x29` as frame pointer, `x30` as link register, `sp` 16-byte aligned at calls, and `x18` off limits on Apple platforms. Mach-O symbols get a leading underscore and addresses are built with `adrp` plus `@PAGEOFF`.

Differential testing. Two independent implementations of the same semantics, an interpreter and a compiler, should agree on every program. Every disagreement is a bug in one of them. Random program generation makes that comparison cheap to scale.

### Architecture

```
source -> lexer -> parser -> type checker -> lowering -> optimiser -> register allocation -> codegen -> cc
                                  |
                                  +-> interpreter (oracle)
```

Everything is OCaml 5.2 built with dune, about 2,700 lines for the compiler and interpreter plus about 600 for the tests, and a 130-line C runtime. The stages communicate through three data structures:

- The AST, which the parser builds and the type checker annotates in place (expression types, what each name refers to, the capture list of each closure, field indices). Every local gets a program-wide unique id, so nothing after the checker has to think about shadowing.
- The IR: temps, constants, `Mov`, `Bin`, `Un`, `Load`, `Store`, `Call` (direct, runtime or through a closure) and `Addr`, with `Jmp`, `Br`, `Ret` and `Unreachable` as block terminators. It is not SSA.
- Assembly text, which Apple's `cc` assembles and links with the runtime.

Runtime checks are explicit in the IR. An array access is a load of the length, an unsigned `ltu` compare (which also rejects negative indices), a branch to a failure block, and then the load. Division is guarded the same way. That puts the checks where the optimiser can see and remove them, and it makes `div` and `load` pure so that dead code elimination can drop them.

The backend has three modes. `-O0` lowers and emits with every temp in its own stack slot. `-O1` runs the optimiser first. `-O2` adds linear-scan allocation over `x19` to `x28`. I only hand out callee-saved registers, which means nothing needs saving around calls; the price is saving the used ones in the prologue.

Closures are called with the closure pointer in `x9`, not in `x0`. That keeps `x0` to `x7` identical for top-level functions and closures, so a top-level function used as a value can be wrapped in a static closure `{ code }` that points straight at the function with no adapter code.

DESIGN.md has the full data types, the frame diagram and the list of trade-offs.

### Implementation

I built the stages in order and kept each one testable before moving on.

The lexer is a single loop over the bytes that tracks line and column. Every error in the compiler goes through one exception, `Compile_error (phase, loc, message)`, and one renderer that prints `file:line:col: phase error: message` followed by the source line and a caret. The parser is hand-written recursive descent with a precedence table. The only context-sensitive rule is Rust's: `if x { ... }` cannot mean a struct literal `x { ... }`, so struct literals are banned in `if`, `while` and `for` headers unless parenthesised. The pretty-printer inserts parentheses only where precedence requires them, and a property test checks that printing and re-parsing every test program gives back the same tree.

The type checker collects all struct and function signatures before checking bodies, so declarations can come in any order and functions can be mutually recursive. It keeps a stack of enclosing lambdas and a nesting level per local; when a name resolves to a local from an outer level, the local is added to the capture list of every lambda in between. That handles `fn(a) -> fn(b) -> fn(c) -> a * 100 + b * 10 + c` where the innermost closure's use of `a` forces the middle closure to capture it too. The checker also enforces definite return, `break` only inside loops, immutability of `let`, and the no-`var`-capture rule.

The interpreter walks the typed AST with a hash table per call frame keyed by local id. It shares no code with the lowering, so a bug in the backend cannot hide by being present on both sides.

Lowering produces blocks with a small builder: `emit` appends an instruction, `finish` closes the current block with a terminator and opens the next. `&&`, `||` and `!` in conditions lower straight to branches, so `if a && b` never materialises a boolean. Code after `return`, `break` or `continue` goes into a fresh block with no predecessors, which the CFG cleanup deletes.

The optimiser has five parts that run in a loop until nothing changes: local constant and copy propagation with folding and algebraic identities, global propagation of temps that have a single constant definition, CFG simplification (constant branches, jump threading, block merging, unreachable block removal), liveness-based dead code elimination, and copy coalescing, which turns `s = a + b; x = s` into `x = a + b`. I added the last one after reading the `-O2` assembly for the `loops` benchmark: every `acc = ...` assignment ended in an extra `mov` because the allocator had to keep both `s` and `acc` alive.

The code generator maps each IR instruction to a short fixed sequence and has a handful of better forms: immediate adds, shifts and compares, `lsl` for multiplication by a power of two, `sdiv` plus `msub` for remainder, `xzr` for the constant 0, and a fused `cmp` plus `b.cond` when a comparison feeds the block's branch. Constants that do not fit 16 bits are built with `movz` and `movk`. Frames beyond the reach of scaled offsets (more than 4,095 slots) fall back to computing the address in `x17`; a generated test with a 2,500-statement function exercises that path at `-O0`.

The runtime is C: a bump allocator over 1 MB chunks, printing, string operations, array creation and the failure functions. It owns the real `main`, which buffers stdout, calls `_kf_main`, and flushes. Runtime errors flush stdout, print `runtime error: ...` to stderr and exit with 101. The C source is embedded in the compiler binary by a dune rule and compiled once into a cache directory.

Testing came in three layers. Alcotest unit tests cover the lexer, parser precedence, pretty-printer round trips and type checker messages. A differential runner compiles all 60 programs in `tests/programs/` at three levels, runs them, and compares stdout, stderr and exit status with the interpreter and a checked-in expected file, and checks that each of the 35 programs in `tests/fail/` produces exactly the diagnostic written in its first line. A fuzzer generates random well-typed programs heavy on wrapping arithmetic, shifts, division, nested loops, calls and closures, and compares the interpreter with all three native builds. I checked the expected outputs of the test programs by hand and against independent Python computations (primes below 5000, the Collatz record below 10000, FNV-1a hashes, 21! modulo 2^64, a quicksort of the same LCG sequence) rather than trusting the interpreter blindly.

PROBLEMS_AND_RESULTS_PLACEHOLDER
