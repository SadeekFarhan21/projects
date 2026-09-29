(* Unit tests for the lexer, parser/pretty-printer and type checker. *)

open Kite
open Ast

(* ---------- helpers ---------- *)

let toks src = List.map (fun (t : Lexer.t) -> Lexer.show t.tok) (Lexer.tokenize src)

let err_of f =
  match f () with
  | _ -> "<no error>"
  | exception Diag.Compile_error (ph, loc, msg) -> Diag.header ph loc msg

(* A position-free S-expression rendering of the AST, so we can compare
   trees and check associativity/precedence explicitly. *)
let rec sx (e : expr) =
  match e.desc with
  | IntLit n -> Int64.to_string n
  | BoolLit b -> string_of_bool b
  | StrLit s -> Printf.sprintf "%S" s
  | Var (x, _) -> x
  | Binary (op, a, b) -> Printf.sprintf "(%s %s %s)" (Pretty.op_str op) (sx a) (sx b)
  | Unary (Neg, a) -> Printf.sprintf "(neg %s)" (sx a)
  | Unary (Not, a) -> Printf.sprintf "(not %s)" (sx a)
  | Call (f, args) -> Printf.sprintf "(call %s%s)" (sx f) (String.concat "" (List.map (fun a -> " " ^ sx a) args))
  | Index (a, i) -> Printf.sprintf "(index %s %s)" (sx a) (sx i)
  | Field (a, f, _) -> Printf.sprintf "(field %s %s)" (sx a) f
  | ArrayLit es -> Printf.sprintf "(array%s)" (String.concat "" (List.map (fun a -> " " ^ sx a) es))
  | ArrayRepeat (v, n) -> Printf.sprintf "(repeat %s %s)" (sx v) (sx n)
  | StructLit (n, fs) ->
    Printf.sprintf "(struct %s%s)" n (String.concat "" (List.map (fun (f, v) -> " " ^ f ^ "=" ^ sx v) fs))
  | Lambda l ->
    Printf.sprintf "(lambda (%s) %s %s)"
      (String.concat " " (List.map (fun p -> p.pname ^ ":" ^ Pretty.ty_expr p.pty) l.lparams))
      (match l.lret with None -> "unit" | Some t -> Pretty.ty_expr t)
      (sx_block l.lbody)

and sx_block b = "{" ^ String.concat " " (List.map sx_stmt b) ^ "}"

and sx_stmt s =
  match s.sdesc with
  | Let (m, x, t, e, _) ->
    Printf.sprintf "(%s %s%s %s)" (if m then "var" else "let") x
      (match t with None -> "" | Some t -> ":" ^ Pretty.ty_expr t) (sx e)
  | Assign (a, b) -> Printf.sprintf "(set %s %s)" (sx a) (sx b)
  | ExprStmt e -> sx e
  | If (c, t, e) ->
    Printf.sprintf "(if %s %s%s)" (sx c) (sx_block t) (match e with None -> "" | Some b -> " " ^ sx_block b)
  | While (c, b) -> Printf.sprintf "(while %s %s)" (sx c) (sx_block b)
  | For (x, a, b, body, _) -> Printf.sprintf "(for %s %s %s %s)" x (sx a) (sx b) (sx_block body)
  | Return None -> "(return)"
  | Return (Some e) -> Printf.sprintf "(return %s)" (sx e)
  | Break -> "break"
  | Continue -> "continue"
  | Block b -> sx_block b

let sx_program p =
  String.concat "\n"
    (List.map
       (function
         | DFn f ->
           Printf.sprintf "(fn %s (%s) %s %s)" f.fname
             (String.concat " " (List.map (fun p -> p.pname ^ ":" ^ Pretty.ty_expr p.pty) f.fparams))
             (match f.fret with None -> "unit" | Some t -> Pretty.ty_expr t)
             (sx_block f.fbody)
         | DStruct s ->
           Printf.sprintf "(struct %s %s)" s.sname
             (String.concat " " (List.map (fun (f, t, _) -> f ^ ":" ^ Pretty.ty_expr t) s.sfields)))
       p)

let pexpr s = sx (Parser.parse_expr_string s)

let check_str = Alcotest.(check string)

(* ---------- lexer ---------- *)

let lexer_tests =
  [
    ("simple statement", `Quick, fun () ->
        Alcotest.(check (list string)) "tokens" [ "let"; "x"; "="; "42"; ";"; "end of file" ] (toks "let x = 42;"));
    ("multi-char operators", `Quick, fun () ->
        Alcotest.(check (list string)) "tokens"
          [ "<="; ">="; "=="; "!="; "<<"; ">>"; "&&"; "||"; "->"; ".."; "<"; "="; "end of file" ]
          (toks "<= >= == != << >> && || -> .. < ="));
    ("keywords vs identifiers", `Quick, fun () ->
        Alcotest.(check (list string)) "tokens"
          [ "fn"; "fnord"; "let"; "letter"; "while"; "_x1"; "end of file" ]
          (toks "fn fnord let letter while _x1"));
    ("positions", `Quick, fun () ->
        let ts = Lexer.tokenize "fn\n  main  (\n)" in
        let pos = List.map (fun (t : Lexer.t) -> (t.loc.line, t.loc.col)) ts in
        Alcotest.(check (list (pair int int))) "positions" [ (1, 1); (2, 3); (2, 9); (3, 1); (3, 2) ] pos);
    ("comments are skipped", `Quick, fun () ->
        Alcotest.(check (list string)) "tokens" [ "a"; "b"; "end of file" ] (toks "a // comment\n// another\nb"));
    ("hex and underscores", `Quick, fun () ->
        Alcotest.(check (list string)) "tokens" [ "255"; "1000000"; "-1"; "end of file" ]
          (toks "0xff 1_000_000 0xffffffffffffffff"));
    ("range after integer", `Quick, fun () ->
        Alcotest.(check (list string)) "tokens" [ "0"; ".."; "10"; "end of file" ] (toks "0..10"));
    ("string escapes", `Quick, fun () ->
        Alcotest.(check (list string)) "tokens" [ "\"a\\nb\\t\\\"\\\\\""; "end of file" ] (toks {|"a\nb\t\"\\"|}));
    ("error: unterminated string", `Quick, fun () ->
        check_str "err" "1:9: lex error: unterminated string literal" (err_of (fun () -> Lexer.tokenize "let s = \"abc")));
    ("error: bad character", `Quick, fun () ->
        check_str "err" "2:3: lex error: unexpected character '@'" (err_of (fun () -> Lexer.tokenize "a\n  @")));
    ("error: literal too large", `Quick, fun () ->
        check_str "err" "1:1: lex error: integer literal 99999999999999999999 is too large (max 9223372036854775807)"
          (err_of (fun () -> Lexer.tokenize "99999999999999999999")));
    ("error: letter after number", `Quick, fun () ->
        check_str "err" "1:3: lex error: unexpected character 'a' after number" (err_of (fun () -> Lexer.tokenize "12abc")));
    ("caret rendering", `Quick, fun () ->
        let src = "fn main() {\n    let x = @;\n}\n" in
        let msg =
          match Lexer.tokenize src with
          | _ -> ""
          | exception Diag.Compile_error (ph, loc, m) -> Diag.render ~file:"t.kite" ~src ph loc m
        in
        check_str "render"
          "t.kite:2:13: lex error: unexpected character '@'\n   |\n 2 |     let x = @;\n   |             ^\n" msg);
  ]

(* ---------- parser ---------- *)

let parser_tests =
  let prec name src expected = (name, `Quick, fun () -> check_str src expected (pexpr src)) in
  [
    prec "mul binds tighter than add" "1 + 2 * 3" "(+ 1 (* 2 3))";
    prec "left associativity" "a - b - c" "(- (- a b) c)";
    prec "division left assoc" "a / b / c" "(/ (/ a b) c)";
    prec "and binds tighter than or" "a || b && c" "(|| a (&& b c))";
    prec "shift below add" "1 + 2 << 3" "(<< (+ 1 2) 3)";
    prec "bitwise above comparison" "a & b == c" "(== (& a b) c)";
    prec "bitwise or/xor/and order" "a | b ^ c & d" "(| a (^ b (& c d)))";
    prec "comparison below arithmetic" "a + 1 < b * 2" "(< (+ a 1) (* b 2))";
    prec "unary binds tighter than binary" "-a * b" "(* (neg a) b)";
    prec "postfix binds tighter than unary" "-a.b[1](2)" "(neg (call (index (field a b) 1) 2))";
    prec "not and and" "!a && b" "(&& (not a) b)";
    prec "parentheses" "(1 + 2) * 3" "(* (+ 1 2) 3)";
    prec "array literal and repeat" "[1, 2][0] + [0; n][1]" "(+ (index (array 1 2) 0) (index (repeat 0 n) 1))";
    prec "struct literal" "P { x: 1, y: a + b }.y" "(field (struct P x=1 y=(+ a b)) y)";
    prec "lambda call" "fn(x: int) -> int { return x; }(3)" "(call (lambda (x:int) int {(return x)}) 3)";
    ("chained comparison rejected", `Quick, fun () ->
        check_str "err" "1:7: parse error: comparison operators cannot be chained; use && to combine them"
          (err_of (fun () -> Parser.parse_expr_string "a < b < c")));
    ("struct literal not allowed in if condition", `Quick, fun () ->
        (* `if x { ... }` must parse the brace as the then-block *)
        let p = Parser.parse_program "fn main() { if x { y = 1; } }" in
        check_str "ast" "(fn main () unit {(if x {(set y 1)})})" (sx_program p));
    ("parenthesised struct literal allowed in condition", `Quick, fun () ->
        let p = Parser.parse_program "fn main() { if (P { a: 1 }).a == 1 { } }" in
        check_str "ast" "(fn main () unit {(if (== (field (struct P a=1) a) 1) {})})" (sx_program p));
    ("else if chains", `Quick, fun () ->
        let p = Parser.parse_program "fn main() { if a { } else if b { } else { c; } }" in
        check_str "ast" "(fn main () unit {(if a {} {(if b {} {c})})})" (sx_program p));
    ("function types", `Quick, fun () ->
        let p = Parser.parse_program "fn f(g: fn(int, [bool]) -> fn(int), h: [[int]]) {}" in
        check_str "ast" "(fn f (g:fn(int, [bool]) -> fn(int) h:[[int]]) unit {})" (sx_program p));
    ("error: missing semicolon", `Quick, fun () ->
        check_str "err" "1:23: parse error: expected ';', found '}'"
          (err_of (fun () -> Parser.parse_program "fn main() { let x = 1 }")));
    ("error: empty array literal", `Quick, fun () ->
        check_str "err" "1:1: parse error: empty array literal is not allowed; use [value; 0]"
          (err_of (fun () -> Parser.parse_expr_string "[]")));
    ("pretty-printer inserts needed parentheses", `Quick, fun () ->
        let e = Parser.parse_expr_string "(a + b) * (c - (d - e))" in
        check_str "printed" "(a + b) * (c - (d - e))" (Pretty.expr "" e));
    ("pretty-printer drops redundant parentheses", `Quick, fun () ->
        let e = Parser.parse_expr_string "((a * b)) + (c)" in
        check_str "printed" "a * b + c" (Pretty.expr "" e));
  ]

(* Round trip: for every test program, printing and re-parsing gives the
   same tree, and printing is a fixed point. *)
let roundtrip_tests =
  let dir = "programs" in
  let files =
    if Sys.file_exists dir then
      Sys.readdir dir |> Array.to_list |> List.filter (fun f -> Filename.check_suffix f ".kite") |> List.sort compare
    else []
  in
  List.map
    (fun f ->
      ( "roundtrip " ^ f,
        `Quick,
        fun () ->
          let src = Driver.read_file (Filename.concat dir f) in
          let p1 = Parser.parse_program src in
          let printed = Pretty.program p1 in
          let p2 = Parser.parse_program printed in
          check_str "same tree" (sx_program p1) (sx_program p2);
          check_str "printing is a fixed point" printed (Pretty.program p2) ))
    files

(* ---------- type checker ---------- *)

let prelude = "struct P { x: int, s: string }\nfn inc(x: int) -> int { return x + 1; }\n"

(* Type of [e] as inferred for `let v = e;` inside main. *)
let type_of e =
  let src = prelude ^ "fn main() { let a = [1, 2]; let v = " ^ e ^ "; }" in
  let p = Parser.parse_program src in
  Typecheck.check_program p;
  let main = List.find_map (function DFn f when f.fname = "main" -> Some f | _ -> None) p in
  match (Option.get main).fbody with
  | _ :: { sdesc = Let (_, _, _, init, _); _ } :: _ -> show_ty init.ty
  | _ -> assert false

let check_err src = err_of (fun () -> Typecheck.check_program (Parser.parse_program src))

let typecheck_tests =
  let infer name e expected = (name, `Quick, fun () -> check_str e expected (type_of e)) in
  [
    infer "int literal" "42" "int";
    infer "comparison" "1 < 2" "bool";
    infer "string concat" "\"a\" + \"b\"" "string";
    infer "array literal" "[true, false]" "[bool]";
    infer "nested arrays" "[[1], [2, 3]]" "[[int]]";
    infer "array repeat" "[\"x\"; 3]" "[string]";
    infer "indexing" "a[0]" "int";
    infer "struct literal" "P { x: 1, s: \"a\" }" "P";
    infer "field" "P { x: 1, s: \"a\" }.s" "string";
    infer "function value" "inc" "fn(int) -> int";
    infer "call" "inc(3)" "int";
    infer "lambda" "fn(x: int, y: bool) -> bool { return y; }" "fn(int, bool) -> bool";
    infer "unit lambda" "fn() {}" "fn()";
    infer "builtins" "len(to_str(3)) + len(a)" "int";
    infer "higher-order" "fn(f: fn(int) -> int) -> fn(int) -> int { return f; }(inc)" "fn(int) -> int";
    ("captures are computed", `Quick, fun () ->
        let src = "fn main() { let a = 1; let b = 2; let c = 3; let f = fn() -> int { return a + c + a; }; }" in
        let p = Parser.parse_program src in
        Typecheck.check_program p;
        let found = ref (-1) in
        (match p with
         | [ DFn { fbody; _ } ] ->
           List.iter
             (fun s -> match s.sdesc with Let (_, "f", _, { desc = Lambda l; _ }, _) -> found := List.length l.lcaptures | _ -> ())
             fbody
         | _ -> ());
        Alcotest.(check int) "captures a and c once each" 2 !found);
    ("nested captures propagate outward", `Quick, fun () ->
        let src = "fn main() { let a = 1; let f = fn() -> int { let g = fn() -> int { return a; }; return g(); }; }" in
        let p = Parser.parse_program src in
        Typecheck.check_program p;
        let outer = ref [] in
        (match p with
         | [ DFn { fbody; _ } ] ->
           List.iter
             (fun s -> match s.sdesc with Let (_, "f", _, { desc = Lambda l; _ }, _) -> outer := l.lcaptures | _ -> ())
             fbody
         | _ -> ());
        Alcotest.(check int) "outer lambda also captures a" 1 (List.length !outer));
    ("error: if condition", `Quick, fun () ->
        check_str "err" "1:16: type error: if condition: expected bool, found int"
          (check_err "fn main() { if 1 {} }"));
    ("error: mutable capture", `Quick, fun () ->
        check_str "err" "1:53: type error: closure cannot capture mutable variable `x` (declare it with `let`)"
          (check_err "fn main() { var x = 1; let f = fn() -> int { return x; }; }"));
    ("error: assign to immutable", `Quick, fun () ->
        check_str "err" "1:24: type error: cannot assign to `x`: it was declared with `let` (use `var`)"
          (check_err "fn main() { let x = 1; x = 2; }"));
    ("error: missing return", `Quick, fun () ->
        check_str "err" "1:1: type error: function `f` may reach the end without returning a value of type int"
          (check_err "fn f() -> int { if true { return 1; } } fn main() {}"));
    ("error: unknown struct", `Quick, fun () ->
        check_str "err" "1:21: type error: unknown struct `Q`"
          (check_err "fn main() { let q = Q { }; }"));
    ("error: redefine builtin", `Quick, fun () ->
        check_str "err" "1:13: type error: `len` is a builtin and cannot be redefined"
          (check_err "fn main() { let len = 3; }"));
    ("error: string equality with int", `Quick, fun () ->
        check_str "err" "1:25: type error: cannot compare string with int"
          (check_err "fn main() { let b = \"a\" == 1; }"));
    ("error: print a struct", `Quick, fun () ->
        check_str "err" "1:61: type error: println cannot print a value of type S"
          (check_err "struct S { a: int } fn main() { let s = S { a: 1 }; println(s); }"));
    ("ok: shadowing and function names", `Quick, fun () ->
        check_str "no error" "<no error>"
          (check_err "fn f() -> int { return 1; } fn main() { let f = 2; let g = f + 1; { let g = \"s\"; } }"));
  ]

let () =
  Alcotest.run "kite"
    [ ("lexer", lexer_tests); ("parser", parser_tests); ("roundtrip", roundtrip_tests);
      ("typecheck", typecheck_tests) ]
