# Design of kitec

This document describes how the Kite compiler is put together: the stages, the data types that flow between them, the machine-level conventions of the AArch64 backend, and the choices made along the way. The language itself is specified in [SPEC.md](SPEC.md).

## Pipeline

```
                 source.kite
                      |
                      v
   +------------------------------------+
   | Lexer          lib/lexer.ml        |  bytes -> tokens with line:col
   +------------------------------------+
                      |
                      v
   +------------------------------------+
   | Parser         lib/parser.ml       |  recursive descent + precedence climbing
   |                                    |  tokens -> AST            (lib/ast.ml)
   | Pretty-printer lib/pretty.ml       |  AST -> canonical source (round-trips)
   +------------------------------------+
                      |
                      v
   +------------------------------------+
   | Type checker   lib/typecheck.ml    |  name resolution, local inference,
   |                                    |  captures, definite return
   |                                    |  AST -> typed AST (annotations filled in)
   +------------------------------------+
            |                       |
            v                       v
   +------------------+   +------------------------------------+
   | Interpreter      |   | Lowering       lib/lower.ml        |
   | lib/interp.ml    |   | typed AST -> three-address IR      |
   | (test oracle)    |   |                (lib/ir.ml)         |
   +------------------+   +------------------------------------+
            |                       |
            |                       v   -O1, -O2
            |             +------------------------------------+
            |             | Optimiser      lib/opt.ml          |
            |             |  constant folding + propagation    |
            |             |  CFG simplification                |
            |             |  dead code elimination (liveness)  |
            |             |  copy coalescing                   |
            |             |  ... repeated to a fixed point     |
            |             +------------------------------------+
            |                       |
            |                       v   -O2
            |             +------------------------------------+
            |             | Register alloc lib/regalloc.ml     |
            |             | linear scan over x19-x28           |
            |             | (-O0/-O1: every temp in a slot)    |
            |             +------------------------------------+
            |                       |
            |                       v
            |             +------------------------------------+
            |             | Code generator lib/codegen.ml      |
            |             | IR -> AArch64 assembly (Mach-O)    |
            |             +------------------------------------+
            |                       |
            |                       v
            |             +------------------------------------+
            |             | cc (Apple clang as assembler and   |
            |             | linker) + runtime/kite_rt.c        |
            |             +------------------------------------+
            |                       |
            v                       v
     stdout/stderr/status  ==  stdout/stderr/status    (tests/run_programs.ml,
                                                         tests/fuzz.ml)
```

`lib/driver.ml` strings the stages together for the CLI (`bin/main.ml`), the test runner and the fuzzer. The runtime's C source is embedded into the compiler at build time (a dune rule turns `runtime/kite_rt.c` into `runtime_src.ml`); `kitec` compiles it once into a cache directory under `$TMPDIR` keyed by its digest, so the compiler binary is self-contained.

## Front end

### Lexer

A hand-written loop over the byte string. It tracks line and column, returns `Lexer.t = { tok; loc }` and raises `Diag.Compile_error (Lex, loc, msg)` on the first error. `Diag.render` turns any compile error into a header line plus the source line and a caret.

### Parser

Hand-written recursive descent, with precedence climbing for binary operators (a table in `Parser.binop_of`). I chose to hand-write it rather than use Menhir for three reasons: the grammar is small (about 300 lines of OCaml cover it), I wanted specific error messages ("comparison operators cannot be chained; use && to combine them", "expected ';', found 'println'") rather than state-number errors, and it avoids a build dependency. The one context-sensitive rule, the ban on bare struct literals in `if`/`while`/`for` headers, is a boolean `no_struct` flag threaded through the expression parser.

### AST

```ocaml
type ty = TInt | TBool | TStr | TUnit | TArray of ty | TStruct of string
        | TFun of ty list * ty | TUnknown

type var_res = Unresolved | RLocal of int | RFunc of string | RBuiltin of builtin

type expr = { desc : expr_desc; loc : loc; mutable ty : ty }
and expr_desc =
  | IntLit of int64 | BoolLit of bool | StrLit of string
  | Var of string * var_res ref
  | Binary of binop * expr * expr | Unary of unop * expr
  | Call of expr * expr list | Index of expr * expr
  | Field of expr * string * int ref            (* field index *)
  | ArrayLit of expr list | ArrayRepeat of expr * expr
  | StructLit of string * (string * expr) list
  | Lambda of lambda
and lambda = { lparams : param list; lret : ty_expr option; lbody : block;
               mutable lcaptures : (int * ty) list; mutable lparam_ids : int list;
               mutable lret_ty : ty; mutable lid : int }
and stmt_desc =
  | Let of bool * string * ty_expr option * expr * int ref   (* mutable?, ..., local id *)
  | Assign of expr * expr | ExprStmt of expr
  | If of expr * block * block option | While of expr * block
  | For of string * expr * expr * block * int ref
  | Return of expr option | Break | Continue | Block of block
```

The parser leaves the mutable fields empty; the checker fills them in. That makes the "typed AST" the same tree, which keeps the code short at the cost of some purity. Every local (including parameters and loop variables) gets a program-wide unique integer id, so later stages never deal with shadowing.

### Type checker

One recursive pass with an environment of scopes. Top-level structs and function signatures are collected first (four passes over the declarations: struct names, struct fields, function signatures, bodies), so declaration order does not matter. Inference is local and bottom-up: every expression's type is computed from its children, and `let` bindings take the type of their initialiser. No unification is needed because function and closure parameters are always annotated and there is no polymorphism; the price is that `[]` (an empty array literal with no element to infer from) is not allowed.

Captures are found during the same walk. The checker keeps a stack of enclosing lambdas and a nesting level for each local. When a name resolves to a local from a lower level, it is appended to the capture list of every lambda between that level and the current one, which handles closures nested in closures.

## Intermediate representation

```ocaml
type operand = T of temp | C of int64
type instr =
  | Mov   of temp * operand
  | Bin   of binop * temp * operand * operand   (* add sub mul div rem and or xor shl shr
                                                    lt le gt ge eq ne ltu *)
  | Un    of unop * temp * operand                (* neg, not *)
  | Load  of temp * operand * int                 (* d <- mem[base + off] *)
  | Store of operand * int * operand              (* mem[base + off] <- v *)
  | Call  of temp option * callee * operand list  (* Direct sym | Runtime sym | Indirect clo *)
  | Addr  of temp * string                        (* address of a data or code symbol *)
type term = Jmp of label | Br of operand * label * label | Ret of operand option | Unreachable
type block = { label; mutable instrs : instr list; mutable term : term }
type func  = { name; params : temp list; env : temp option; mutable blocks; mutable ntemps }
```

It is a conventional three-address code in basic blocks, not SSA. Every value is one 64-bit word. A source variable maps to one temp that may be assigned many times; expression intermediates get fresh temps. Everything that can fail at run time is explicit in the IR: an array access lowers to a load of the length, an unsigned compare `ltu i, len` (which also catches negative indices), a branch to a block that calls `kite_rt_oob`, and only then the address arithmetic and the load. Division lowers to a compare with zero and a branch to `kite_rt_divzero` before the `div`. Because of this, `div`, `rem` and `load` are pure in the IR and the optimiser may delete them when their result is dead.

Here is `fib` at `-O1` (`kitec ir bench/fib.kite -O1`):

```
func _kf_fib(t0) {
L0:
  t1 = lt t0, 2
  br t1, L1, L3
L1:
  ret t0
L3:
  t2 = sub t0, 1
  t3 = call _kf_fib(t2)
  t4 = sub t0, 2
  t5 = call _kf_fib(t4)
  t6 = add t3, t5
  ret t6
}
```

### Heap object layouts

All fields are 8-byte words; the runtime's bump allocator returns 16-byte aligned blocks.

```
string   [len][bytes ... NUL]       the pointer points at len
array    [len][e0][e1] ...          element i at offset 8 + 8*i
struct   [f0][f1] ...               declaration order
closure  [code][cap0][cap1] ...     code = address of the lifted function
```

A top-level function used as a value gets a static closure object in `__DATA` (`_kc_f: .quad _kf_f`), so there is exactly one representation of a function value and one way to call one.

### Optimisations

All passes are in `lib/opt.ml` and run in a loop until none of them changes anything (at most 20 rounds; over the test and benchmark programs, 123 of 145 functions settle in 2 rounds and none needs more than 4, see `results/opt_rounds.txt`).

1. Local constant and copy propagation with folding. Within a block, remember temps known to equal a constant or another temp, substitute them into later operands, and fold instructions whose operands are all constants, plus identities such as `x + 0`, `x * 1`, `x * 0`, `x & 0`. Folding uses exactly the interpreter's 64-bit operations (wrapping, shift amounts mod 64), and refuses to fold a division by zero.
2. Global single-definition constant propagation. If a temp has exactly one definition in the function and it is `t = c`, every use of `t` becomes `c`. This is only sound if the definition dominates all uses. For the IR that lowering produces it does: Kite has no uninitialised variables and scopes are lexical, so a variable's declaration comes before every use on every path. I rely on that invariant instead of computing dominators; it is noted as something to replace once the IR is SSA.
3. CFG simplification: branches on constants become jumps, jumps to empty forwarding blocks are threaded, a block whose only successor has one predecessor absorbs it, and blocks unreachable from the entry are deleted. This is what removes the out-of-bounds and division-by-zero failure paths once a check folds to true.
4. Dead code elimination. Backward liveness (a standard iterative dataflow over the CFG), then delete pure instructions whose result is not live.
5. Copy coalescing: `s = op a, b; d = s` with `s` dead afterwards becomes `d = op a, b`. Lowering emits this pair for every assignment to an existing variable.

## Backend: AArch64, Apple arm64 ABI

### Symbols and sections

Kite function `f` becomes `_kf_f`, closure bodies `_kl_<n>`, static closures `_kc_f`, string literals `_ks_<n>`, and C runtime functions keep the C name with the Mach-O underscore (`_kite_rt_alloc`). Code goes in `__TEXT,__text`, string literals in `__TEXT,__const`, static closures in `__DATA,__data` (they hold an absolute code address, which needs a relocation, so they cannot live in a read-only text section). Addresses are formed with `adrp reg, sym@PAGE` plus `add reg, reg, sym@PAGEOFF`. Local labels start with `L` so the assembler keeps them out of the symbol table. The file ends with `.subsections_via_symbols`, as Apple's toolchain expects.

### Calling convention

Kite code follows the Apple variant of AAPCS64 so that Kite and C can call each other:

- Arguments go in `x0` to `x7`, the result comes back in `x0`. The language limits functions and closures to 8 parameters, so arguments are never passed on the stack.
- `x19` to `x28`, `x29` (frame pointer) and `x30` (link register) are callee-saved. `sp` is 16-byte aligned at every call.
- `x18` is reserved by Apple and never touched.
- Closure calls pass the closure pointer in `x9`, a caller-saved scratch register that is not an argument register. A closure body copies `x9` into its environment temp in its prologue and reads captures from `[x9 + 8 + 8*i]`. Top-level functions simply ignore `x9`. Because the environment travels outside `x0..x7`, a top-level function and a closure with the same parameter types have identical argument registers, and a static closure for a top-level function can point straight at the function with no adapter.

An indirect call is therefore:

```
    ldr   x9, <closure>        // environment pointer
    ...   x0..x7 <- arguments
    ldr   x16, [x9]            // code pointer from the closure's first word
    blr   x16
```

### Stack frame

```
 higher addresses
 +-----------------------------+
 | caller's frame              |
 +-----------------------------+ <- sp at entry (16-byte aligned)
 | saved x30 (lr)              |  [x29 + 8]
 | saved x29 (caller's fp)     |  [x29]        <- x29
 +-----------------------------+
 | padding to 16 bytes         |
 | saved x19 .. x28 (only the  |  [sp + 8*(nslots + j)]
 |   ones the allocator used)  |
 +-----------------------------+
 | spill slot nslots-1         |
 | ...                         |
 | spill slot 1                |  [sp + 8]
 | spill slot 0                |  [sp + 0]     <- sp (16-byte aligned)
 +-----------------------------+
 lower addresses
```

Prologue and epilogue:

```
    stp   x29, x30, [sp, #-16]!
    mov   x29, sp
    sub   sp, sp, #frame           // frame = round_up_16(8 * (nslots + nsaved))
    str   x19, [sp, #8*nslots]     // one store per callee-saved register used
    ...                            // x9 -> env temp, x0..x7 -> parameter temps
  L<fn>_ret:
    ldr   x19, [sp, #8*nslots]
    mov   sp, x29
    ldp   x29, x30, [sp], #16
    ret
```

Slots are addressed from `sp` with a positive scaled offset, which reaches 32760 bytes. Beyond that (at `-O0` a function with more than about 4000 temps, which `tests/programs/big_function.kite` exercises) the address is built in `x17`. Frames of 4096 bytes or more use the `sub sp, sp, #n, lsl #12` form.

### Instruction selection

Each IR instruction maps to a short fixed sequence. Operands are fetched with a helper that returns a register: the temp's allocated register, a scratch register (`x10`, `x11`) loaded from its slot, or `xzr` for the constant 0. Results are computed into the temp's register or into `x12` and then stored. Constants that fit 16 bits use one `mov`; others are built from `movz` plus `movk` per non-zero halfword. Small cases get better forms: add or subtract of 0..4095 uses the immediate form, shifts by constants use the immediate form, multiplication by a power of two becomes `lsl`, and remainder is `sdiv` plus `msub`. A comparison immediately followed by a branch on its result is fused into `cmp` plus `b.cond`, and at `-O2` the `cset` is dropped when the boolean is not live afterwards. Blocks are emitted in IR order and a jump to the next block is omitted.

### Register allocation

At `-O0` and `-O1` every temp has its own stack slot, the simplest correct scheme. At `-O2`, `lib/regalloc.ml` runs linear scan (Poletto and Sarkar): blocks are laid out in order, every instruction gets a position, and each temp gets a single interval from its first to its last position, extended over every block where it is live-in or live-out. Intervals are scanned by start; a free register is taken from `x19..x28` if there is one, otherwise the interval that ends last is spilled to a slot. Only callee-saved registers are used, so no value ever needs saving around a call; the cost is saving and restoring the used ones in the prologue and epilogue, which is why `fib` saves three registers.

## Runtime

`runtime/kite_rt.c` is about 130 lines: a bump allocator over 1 MB `calloc` chunks, printing, string concatenation, equality, conversion and slicing, array creation, and the failure functions for bounds, division by zero and asserts. It owns the real `main`, which sets up a 64 KB stdout buffer, calls `_kf_main` and flushes. Runtime errors flush stdout, print to stderr and `exit(101)`.

## Testing architecture

- `tests/test_unit.ml` (alcotest): lexer tokens, positions and errors, caret rendering; parser precedence and associativity through an S-expression dump of the AST; a round-trip check over every test program (parse, print, parse again, compare trees, and check that printing is a fixed point); type inference results, capture lists and type error messages.
- `tests/run_programs.ml`: the differential runner. For each program it runs the interpreter in process, generates assembly at three levels, then assembles, links and runs the executables in parallel (`xargs -P 8`). All four transcripts (stdout, plus exit status and stderr when either is non-trivial) must equal the checked-in `.out` file. Compile-fail cases carry their expected diagnostic in the first line.
- `tests/fuzz.ml`: random well-typed programs full of wrapping arithmetic, shifts, division, nested control flow, calls and closures; interpreter output must match all three native builds.

## Trade-offs

### Chosen

- Closures over generics or algebraic data types as the one optional feature. The brief already asks for first-class functions, and without capture they are only function pointers; closures made higher-order code like `make_adder` and `compose` possible, and they exercise interesting parts of a compiler (capture analysis, lambda lifting, an indirect calling convention). Capturing by value, with capture of `var` forbidden, gives one simple rule with no boxing: since captured variables cannot change, copying them is indistinguishable from sharing them. Shared mutable state is still available through a captured struct or array.
- Uniform one-word values. Every Kite value fits in a 64-bit register (ints, bools as 0/1, pointers for everything else). The IR has no types and the backend has one register class, which kept both small.
- Reference semantics for arrays and structs, like Java. Value semantics would need copies on assignment and aggregate return conventions.
- Explicit checks in the IR, so that the optimiser sees and can remove them, and so that `div` and `load` are pure.
- Assembly text through `cc` instead of emitting Mach-O objects directly. Apple clang's integrated assembler handles relocations, the page/pageoff pairs and the linker, and a text `.s` file is easy to read when debugging.
- Callee-saved registers only in the allocator, which removes all caller-save logic around calls.
- The interpreter as oracle, with no code shared with the backend, so a bug in lowering or the optimiser cannot hide by being present on both sides.

### Rejected or deferred

- SSA. It makes the optimisations cleaner (and would replace the dominance assumption in single-definition constant propagation) but needs dominator trees, phi insertion and out-of-SSA copies. The non-SSA IR was enough for the v0 passes.
- Graph-colouring allocation. Linear scan was under 100 lines and already gives 2x to 10x over stack slots on the benchmarks.
- Passing more than 8 arguments on the stack. The restriction is in the type checker; lifting it means caller-side outgoing argument areas and callee-side incoming offsets.
- Garbage collection. A bump allocator that never frees is correct for short programs and makes allocation a pointer increment. The one-word uniform layout would make a precise collector feasible later if the backend records which slots hold pointers.
- Fixed-length array types (`[int; 3]`). Putting the length in the type would make every function that takes an array specific to one size. Length is a runtime property and bounds checks are dynamic.
- Menhir. See the parser section.
- Mutable captures (boxing captured variables). Forbidden instead; see above.
