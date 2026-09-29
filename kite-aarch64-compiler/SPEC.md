# The Kite language, version 0

Kite is a small, statically typed, imperative language with first-class functions and closures. Source files use the extension `.kite`. This document is the reference for the v0 compiler (`kitec`) and its tree-walking interpreter; where they could differ, this document decides, and the test suite checks that they do not.

```kite
struct Point {
    x: int,
    y: int,
}

fn make_adder(n: int) -> fn(int) -> int {
    return fn(x: int) -> int { return x + n; };
}

fn main() {
    let p = Point { x: 1, y: 2 };
    let add5 = make_adder(5);
    var total = 0;
    for i in 0..10 {
        total = total + add5(i) * p.y;
    }
    println("total = " + to_str(total));
}
```

## 1. Lexical structure

### Characters and whitespace

Source is read as bytes. Spaces, tabs, carriage returns and newlines separate tokens and are otherwise ignored. A comment starts with `//` and runs to the end of the line. Positions in diagnostics are 1-based line and column, where a column counts bytes.

### Identifiers and keywords

```ebnf
ident   = letter { letter | digit } ;
letter  = "a" ... "z" | "A" ... "Z" | "_" ;
digit   = "0" ... "9" ;
```

The keywords are `fn let var if else while for in return break continue struct true false`. They cannot be used as identifiers. The builtin function names `print println len to_str substr chr assert` and the type names `int bool string` are not keywords, but they cannot be redefined (section 4.9).

### Literals

```ebnf
int_lit    = dec_lit | hex_lit ;
dec_lit    = digit { digit | "_" } ;
hex_lit    = ("0x" | "0X") hex_digit { hex_digit | "_" } ;
string_lit = '"' { char | escape } '"' ;
escape     = "\n" | "\t" | "\r" | "\0" | "\\" | '\"' ;
```

Underscores in numbers are ignored. A decimal literal must be at most 9223372036854775807 (2^63 - 1); a literal cannot be negative, `-5` is the unary minus applied to `5`. To write the smallest integer, use `-9223372036854775807 - 1`. A hex literal may have up to 16 digits and denotes the 64-bit two's complement pattern, so `0xffffffffffffffff` is `-1`. A number immediately followed by a letter is an error. String literals cannot span lines; `char` is any byte except `"`, `\` and newline.

### Operators and punctuation

```
( ) { } [ ] , ; : . .. -> =
+ - * / % & | ^ << >> ! && || == != < <= > >=
```

The lexer always takes the longest match, so `<<=` is `<<` followed by `=`.

## 2. Grammar

The grammar below is complete. `{ x }` means zero or more, `[ x ]` means optional.

```ebnf
program     = { decl } EOF ;
decl        = fn_decl | struct_decl ;

struct_decl = "struct" ident "{" [ field { "," field } [ "," ] ] "}" ;
field       = ident ":" type ;

fn_decl     = "fn" ident "(" [ params ] ")" [ "->" type ] block ;
params      = param { "," param } [ "," ] ;
param       = ident ":" type ;

type        = ident                                  (* int, bool, string, or a struct name *)
            | "[" type "]"                           (* array *)
            | "fn" "(" [ type { "," type } [ "," ] ] ")" [ "->" type ] ;

block       = "{" { stmt } "}" ;
stmt        = ( "let" | "var" ) ident [ ":" type ] "=" expr ";"
            | if_stmt
            | "while" expr_ns block
            | "for" ident "in" expr_ns ".." expr_ns block
            | "return" [ expr ] ";"
            | "break" ";"
            | "continue" ";"
            | block
            | expr "=" expr ";"                      (* assignment *)
            | expr ";" ;                             (* expression statement *)
if_stmt     = "if" expr_ns block [ "else" ( block | if_stmt ) ] ;

expr        = or_expr ;
or_expr     = and_expr { "||" and_expr } ;
and_expr    = cmp_expr { "&&" cmp_expr } ;
cmp_expr    = bor_expr [ cmp_op bor_expr ] ;         (* non-associative *)
cmp_op      = "==" | "!=" | "<" | "<=" | ">" | ">=" ;
bor_expr    = bxor_expr { "|" bxor_expr } ;
bxor_expr   = band_expr { "^" band_expr } ;
band_expr   = shift_expr { "&" shift_expr } ;
shift_expr  = add_expr { ( "<<" | ">>" ) add_expr } ;
add_expr    = mul_expr { ( "+" | "-" ) mul_expr } ;
mul_expr    = unary { ( "*" | "/" | "%" ) unary } ;
unary       = ( "-" | "!" ) unary | postfix ;
postfix     = primary { "(" [ args ] ")" | "[" expr "]" | "." ident } ;
args        = expr { "," expr } [ "," ] ;
primary     = int_lit | string_lit | "true" | "false"
            | ident
            | ident "{" [ field_init { "," field_init } [ "," ] ] "}"   (* struct literal *)
            | "(" expr ")"
            | "[" expr { "," expr } [ "," ] "]"      (* array literal, at least one element *)
            | "[" expr ";" expr "]"                  (* array of n copies *)
            | "fn" "(" [ params ] ")" [ "->" type ] block ;             (* closure *)
field_init  = ident ":" expr ;
```

### Precedence

From loosest to tightest: `||`, `&&`, comparisons, `|`, `^`, `&`, shifts, `+ -`, `* / %`, unary `- !`, postfix call/index/field. All binary operators except comparisons are left-associative. Comparisons do not associate: `a < b < c` is a parse error with a hint to use `&&`. Unlike C, the bitwise operators bind tighter than comparisons, so `x & 1 == 0` means `(x & 1) == 0`.

### The struct-literal restriction

`expr_ns` is `expr` in which a struct literal may not appear unless it is inside parentheses, brackets or call arguments. It is used for the condition of `if` and `while` and for the bounds of `for`. Without it, `if x { ... }` is ambiguous: `x { ... }` could start a struct literal. Write `if p == (P { x: 1 }).x { ... }` when you need one.

### Assignment targets

The left side of `=` must be a variable, an index expression `a[i]`, or a field expression `e.f`. This is checked by the type checker, not the grammar.

## 3. Types

```
t ::= int | bool | string | [t] | S | fn(t1, ..., tn) -> t | unit
```

- `int` is a 64-bit two's complement integer.
- `bool` is `true` or `false`.
- `string` is an immutable byte string.
- `[t]` is an array of `t`. The length is fixed when the array is created and is not part of the type.
- `S` is a struct type introduced by `struct S { ... }`. Struct types are nominal.
- `fn(t1, ..., tn) -> t` is the type of functions and closures. `fn(t1, ..., tn)` without an arrow returns `unit`.
- `unit` has no syntax of its own. It is the result type of functions declared without `->` and of statements used as expressions. No variable, parameter, field or array element may have type `unit`.

Type equality is structural for arrays and functions and by name for structs.

## 4. Static semantics

A program is well typed if every declaration and statement satisfies the rules below. The checker reports the first violation with its line and column.

### 4.1 Declarations

- Struct names are unique and are not `int`, `bool`, `string` or a builtin name. Field names within a struct are unique. Field types may mention any struct, including the struct itself.
- Function names are unique and are not builtin names. All top-level functions are in scope in every function body, so functions may be mutually recursive in any order.
- A function or closure has at most 8 parameters, and parameter names within one parameter list are unique.
- There must be a function `main` with no parameters and no result.

### 4.2 Scopes and names

Each block, function body, closure body and `for` loop introduces a scope. `let x = e;` and `var x = e;` bring `x` into scope from the next statement to the end of the enclosing block; `e` itself is checked in the outer scope, so `let x = x + 1;` refers to the outer `x`. An inner declaration may shadow an outer one, including a top-level function. A name is looked up in the local scopes from innermost outwards, then among the top-level functions.

### 4.3 Local type inference

`let x = e;` gives `x` the type of `e`. With an annotation, `let x: t = e;` requires the type of `e` to equal `t`. Every expression's type is determined bottom-up from its parts, so no annotation is ever needed on a local; parameters and results of functions and closures must always be annotated.

### 4.4 Mutability

`let` bindings, parameters and `for` loop variables are immutable. `var` bindings may be assigned. Assigning to an array element or a struct field is allowed through any binding, since it changes the object, not the binding.

### 4.5 Expressions

Writing `e : t` for "e has type t":

- Integer literals are `int`, `true`/`false` are `bool`, string literals are `string`.
- `x` has the type of its binding. A top-level function name used as a value has its function type.
- `-e : int` if `e : int`. `!e : bool` if `e : bool`.
- `a + b : int` if both are `int`; `a + b : string` if both are `string` (concatenation).
- `- * / % & | ^ << >>` take two `int` and give `int`.
- `< <= > >=` take two `int` and give `bool`.
- `== !=` take two operands of the same type, which must be `int`, `bool` or `string`, and give `bool`. Arrays, structs and functions cannot be compared.
- `&& ||` take two `bool` and give `bool`.
- `f(a1, ..., an) : r` if `f : fn(t1, ..., tn) -> r` and each `ai : ti`.
- `a[i] : t` if `a : [t]` and `i : int`. Strings cannot be indexed; use `substr`.
- `e.f : t` if `e : S` and struct `S` has field `f : t`.
- `[e1, ..., en] : [t]` if every `ei : t` (n >= 1).
- `[e; n] : [t]` if `e : t` and `n : int`.
- `S { f1: e1, ..., fn: en } : S` if every field of `S` is given exactly once, in any order, with the declared type.
- `fn(x1: t1, ..., xn: tn) -> r { body } : fn(t1, ..., tn) -> r`, where the body is checked with the parameters in scope and `return` expecting `r`.

### 4.6 Closures and captures

A closure may refer to locals of enclosing functions and closures. Every such variable is captured. A captured variable must be immutable (`let`, a parameter, or a `for` variable); capturing a `var` is an error. Consequently the value seen inside the closure is always the value the variable had when the closure was created. A closure nested inside another closure that captures an outer variable makes the middle closure capture it too. A closure has no name, so it cannot call itself directly.

### 4.7 Statements

- `if` and `while` conditions must be `bool`; `for` bounds must be `int`.
- `break` and `continue` must be inside a `while` or `for` of the same function or closure body.
- In a function returning `r`, `return e;` requires `e : r`. In a function returning `unit`, only `return;` is allowed.
- An assignment `lhs = e;` requires `lhs` to be an assignable place of the same type as `e`.

### 4.8 Definite return

A function or closure with a non-unit result must return on every path. The check is syntactic: a block returns if one of its statements is `return`, a nested block that returns, or an `if` with an `else` where both branches return. A `while true` loop does not count, so a function ending in one needs a final `return`.

### 4.9 Builtins

The builtins are called like functions but are resolved by the checker and cannot be used as values or redefined.

| builtin | type | meaning |
|---|---|---|
| `print(x)` | `int`, `bool` or `string` -> unit | write `x` to stdout |
| `println(x)` / `println()` | same | write `x` then a newline |
| `len(x)` | `[t]` or `string` -> `int` | number of elements or bytes |
| `to_str(x)` | `int` or `bool` -> `string` | decimal or `true`/`false` |
| `substr(s, start, count)` | `string, int, int -> string` | bytes `start .. start+count` |
| `chr(b)` | `int -> string` | one-byte string, `0 <= b <= 255` |
| `assert(c)` | `bool -> unit` | runtime error if `c` is false |

## 5. Dynamic semantics

### 5.1 Programs

Running a program calls `main`. When `main` returns, the program exits with status 0. Output is written to stdout and is flushed before exit.

### 5.2 Evaluation order

Evaluation is strict and left to right: operands of binary operators, the callee before the arguments, arguments in order, array literal elements in order, and struct literal fields in the order written in the source (not the declaration order). In `a[i] = e;` the order is `a`, `i`, `e`, then the bounds check and the store. In `o.f = e;` it is `o` then `e`. `&&` and `||` evaluate their right operand only when needed. The bounds of a `for` loop are evaluated once, before the first iteration.

### 5.3 Integers

All integer arithmetic is modulo 2^64 with two's complement representation; overflow wraps and is never an error. `/` truncates toward zero and `%` gives a remainder with the sign of the dividend, so `a == (a / b) * b + a % b`. Division or remainder by zero is a runtime error. `(-2^63) / -1` wraps to `-2^63` and `(-2^63) % -1` is `0`. For `a << b` and `a >> b` only the low 6 bits of `b` are used (the shift amount is `b mod 64` in the range 0..63). `>>` is an arithmetic shift.

### 5.4 Values and references

Integers and booleans are values. Strings are immutable, so whether they are shared is not observable. Arrays, structs and closures are heap objects handled by reference: assigning, passing or returning one copies the reference, and a change made through one reference is visible through all others. `[e; n]` evaluates `e` once and stores that value `n` times, so `[[0; 2]; 2]` contains the same inner array twice. There is no null: every variable is initialised when declared, and every array element and field is initialised when the object is created. Memory is never reclaimed in v0.

### 5.5 Control flow

`if`, `while`, `for`, `break`, `continue` and `return` behave as in C. `for x in a..b { body }` runs `body` with `x` bound to `a, a+1, ..., b-1` in order; if `a >= b` it runs zero times. `continue` in a `for` moves to the next value of `x`.

### 5.6 Functions and closures

A call evaluates the callee and the arguments, binds the parameters, and runs the body until a `return` or the end of the body (which is only reachable in `unit` functions). Evaluating a closure expression creates a closure object holding the current values of its captured variables. Recursion has no language limit; running out of native stack is not detected (see the README's known issues).

### 5.7 Runtime errors

The following stop the program. Output already printed is kept, the message `runtime error: <description>` is written to stderr followed by a newline, and the exit status is 101.

| cause | message |
|---|---|
| index `i` outside `0 .. len-1` (read or write) | `index out of bounds: index i, length n` |
| `/` or `%` with a zero divisor | `division by zero` |
| `[e; n]` with `n < 0` | `negative array length: n` |
| `substr` outside the string | `substr out of bounds: start s, count c, length n` |
| `chr(b)` with `b` outside 0..255 | `chr argument out of range: b` |
| `assert(false)` | `assertion failed` |
