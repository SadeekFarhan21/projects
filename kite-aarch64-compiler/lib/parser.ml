(* Hand-written recursive-descent parser with precedence climbing for
   binary operators. See SPEC.md for the grammar.

   The one real ambiguity in the grammar is `IDENT {`: in
   `if x { ... }` the brace opens the then-block, but in
   `let p = P { x: 1 };` it opens a struct literal. Like Rust, we
   forbid bare struct literals in the condition of `if`/`while` and the
   range of `for` (the [no_struct] flag); wrap them in parentheses there. *)

open Diag
open Lexer
open Ast

type st = { toks : Lexer.t array; mutable i : int }

let peek st = st.toks.(st.i).tok
let peek2 st = if st.i + 1 < Array.length st.toks then st.toks.(st.i + 1).tok else EOF
let loc st = st.toks.(st.i).loc
let advance st = if st.i < Array.length st.toks - 1 then st.i <- st.i + 1

let fail st fmt = error Parse (loc st) fmt

let expect st tok what =
  if peek st = tok then advance st
  else fail st "expected %s, found '%s'" what (show (peek st))

let ident st what =
  match peek st with
  | IDENT s -> advance st; s
  | t -> fail st "expected %s, found '%s'" what (show t)

(* comma-separated list up to [close], allowing a trailing comma *)
let comma_list st close f =
  let rec go acc =
    if peek st = close then List.rev acc
    else begin
      let x = f st in
      if peek st = COMMA then (advance st; go (x :: acc))
      else if peek st = close then List.rev (x :: acc)
      else fail st "expected ',' or '%s', found '%s'" (show close) (show (peek st))
    end
  in
  let r = go [] in
  expect st close ("'" ^ show close ^ "'");
  r

let rec parse_type st : ty_expr =
  let l = loc st in
  match peek st with
  | IDENT s -> advance st; TEName (s, l)
  | LBRACKET ->
    advance st;
    let t = parse_type st in
    expect st RBRACKET "']'";
    TEArray (t, l)
  | FN ->
    advance st;
    expect st LPAREN "'('";
    let ps = comma_list st RPAREN parse_type in
    let r = if peek st = ARROW then (advance st; Some (parse_type st)) else None in
    TEFn (ps, r, l)
  | t -> fail st "expected a type, found '%s'" (show t)

let parse_param st =
  let ploc = loc st in
  let pname = ident st "parameter name" in
  expect st COLON "':' and a parameter type";
  let pty = parse_type st in
  { pname; pty; ploc }

(* Binary operator table: token -> (op, precedence). Higher binds tighter. *)
let binop_of = function
  | OROR -> Some (Or, 1)
  | ANDAND -> Some (And, 2)
  | EQ -> Some (Eq, 3) | NE -> Some (Ne, 3)
  | LT -> Some (Lt, 3) | LE -> Some (Le, 3) | GT -> Some (Gt, 3) | GE -> Some (Ge, 3)
  | PIPE -> Some (BOr, 4)
  | CARET -> Some (BXor, 5)
  | AMP -> Some (BAnd, 6)
  | SHL -> Some (Shl, 7) | SHR -> Some (Shr, 7)
  | PLUS -> Some (Add, 8) | MINUS -> Some (Sub, 8)
  | STAR -> Some (Mul, 9) | SLASH -> Some (Div, 9) | PERCENT -> Some (Mod, 9)
  | _ -> None

let is_comparison = function Eq | Ne | Lt | Le | Gt | Ge -> true | _ -> false

let rec parse_expr ?(no_struct = false) st = parse_binary st no_struct 1

and parse_binary st ns min_prec =
  let lhs = ref (parse_unary st ns) in
  let continue_ = ref true in
  while !continue_ do
    match binop_of (peek st) with
    | Some (op, p) when p >= min_prec ->
      let l = loc st in
      advance st;
      let rhs = parse_binary st ns (p + 1) in
      (* comparisons are non-associative: a < b < c is rejected *)
      (if is_comparison op then
         match binop_of (peek st) with
         | Some (op2, _) when is_comparison op2 ->
           fail st "comparison operators cannot be chained; use && to combine them"
         | _ -> ());
      lhs := mk (Binary (op, !lhs, rhs)) l
    | _ -> continue_ := false
  done;
  !lhs

and parse_unary st ns =
  let l = loc st in
  match peek st with
  | MINUS -> advance st; mk (Unary (Neg, parse_unary st ns)) l
  | BANG -> advance st; mk (Unary (Not, parse_unary st ns)) l
  | _ -> parse_postfix st ns

and parse_postfix st ns =
  let e = ref (parse_primary st ns) in
  let continue_ = ref true in
  while !continue_ do
    let l = loc st in
    match peek st with
    | LPAREN ->
      advance st;
      let args = comma_list st RPAREN (fun st -> parse_expr st) in
      e := mk (Call (!e, args)) l
    | LBRACKET ->
      advance st;
      let idx = parse_expr st in
      expect st RBRACKET "']'";
      e := mk (Index (!e, idx)) l
    | DOT ->
      advance st;
      let l = loc st in
      let f = ident st "field name after '.'" in
      e := mk (Field (!e, f, ref (-1))) l
    | _ -> continue_ := false
  done;
  !e

and parse_primary st ns =
  let l = loc st in
  match peek st with
  | INT n -> advance st; mk (IntLit n) l
  | STR s -> advance st; mk (StrLit s) l
  | TRUE -> advance st; mk (BoolLit true) l
  | FALSE -> advance st; mk (BoolLit false) l
  | IDENT name when peek2 st = LBRACE && not ns ->
    advance st; advance st;
    let field st =
      let fl = loc st in
      let f = ident st "field name" in
      expect st COLON "':' after field name";
      let v = parse_expr st in
      ignore fl;
      (f, v)
    in
    let fields = comma_list st RBRACE field in
    mk (StructLit (name, fields)) l
  | IDENT name -> advance st; mk (Var (name, ref Unresolved)) l
  | LPAREN ->
    advance st;
    let e = parse_expr st in
    expect st RPAREN "')'";
    e
  | LBRACKET ->
    advance st;
    if peek st = RBRACKET then
      error Parse l "empty array literal is not allowed; use [value; 0]"
    else begin
      let first = parse_expr st in
      if peek st = SEMI then begin
        advance st;
        let count = parse_expr st in
        expect st RBRACKET "']'";
        mk (ArrayRepeat (first, count)) l
      end else if peek st = RBRACKET then (advance st; mk (ArrayLit [ first ]) l)
      else begin
        expect st COMMA "',' or ']' in array literal";
        let rest = comma_list st RBRACKET (fun st -> parse_expr st) in
        mk (ArrayLit (first :: rest)) l
      end
    end
  | FN ->
    advance st;
    expect st LPAREN "'(' after fn";
    let lparams = comma_list st RPAREN parse_param in
    let lret = if peek st = ARROW then (advance st; Some (parse_type st)) else None in
    let lbody = parse_block st in
    mk (Lambda { lparams; lret; lbody; lcaptures = []; lparam_ids = [];
                 lret_ty = TUnknown; lid = -1 }) l
  | t -> fail st "expected an expression, found '%s'" (show t)

and parse_block st : block =
  expect st LBRACE "'{'";
  let rec go acc =
    match peek st with
    | RBRACE -> advance st; List.rev acc
    | EOF -> fail st "unexpected end of file; missing '}'"
    | _ -> go (parse_stmt st :: acc)
  in
  go []

and parse_if st =
  let l = loc st in
  expect st IF "'if'";
  let cond = parse_expr ~no_struct:true st in
  let thn = parse_block st in
  let els =
    if peek st = ELSE then begin
      advance st;
      if peek st = IF then Some [ parse_if st ] else Some (parse_block st)
    end else None
  in
  { sdesc = If (cond, thn, els); sloc = l }

and parse_stmt st : stmt =
  let l = loc st in
  let s d = { sdesc = d; sloc = l } in
  match peek st with
  | LET | VAR ->
    let mut = peek st = VAR in
    advance st;
    let name = ident st "variable name" in
    let annot = if peek st = COLON then (advance st; Some (parse_type st)) else None in
    expect st ASSIGN "'=' (variables must be initialised)";
    let init = parse_expr st in
    expect st SEMI "';'";
    s (Let (mut, name, annot, init, ref (-1)))
  | IF -> parse_if st
  | WHILE ->
    advance st;
    let cond = parse_expr ~no_struct:true st in
    let body = parse_block st in
    s (While (cond, body))
  | FOR ->
    advance st;
    let name = ident st "loop variable" in
    expect st IN "'in'";
    let lo = parse_expr ~no_struct:true st in
    expect st DOTDOT "'..' in range";
    let hi = parse_expr ~no_struct:true st in
    let body = parse_block st in
    s (For (name, lo, hi, body, ref (-1)))
  | RETURN ->
    advance st;
    if peek st = SEMI then (advance st; s (Return None))
    else begin
      let e = parse_expr st in
      expect st SEMI "';'";
      s (Return (Some e))
    end
  | BREAK -> advance st; expect st SEMI "';'"; s Break
  | CONTINUE -> advance st; expect st SEMI "';'"; s Continue
  | LBRACE -> s (Block (parse_block st))
  | _ ->
    let e = parse_expr st in
    if peek st = ASSIGN then begin
      advance st;
      let rhs = parse_expr st in
      expect st SEMI "';'";
      s (Assign (e, rhs))
    end else begin
      expect st SEMI "';'";
      s (ExprStmt e)
    end

let parse_decl st : decl =
  let l = loc st in
  match peek st with
  | FN ->
    advance st;
    let fname = ident st "function name" in
    expect st LPAREN "'('";
    let fparams = comma_list st RPAREN parse_param in
    let fret = if peek st = ARROW then (advance st; Some (parse_type st)) else None in
    let fbody = parse_block st in
    DFn { fname; fparams; fret; fbody; floc = l; fparam_ids = []; fparam_tys = [];
          fret_ty = TUnknown }
  | STRUCT ->
    advance st;
    let sname = ident st "struct name" in
    expect st LBRACE "'{'";
    let field st =
      let fl = loc st in
      let f = ident st "field name" in
      expect st COLON "':' after field name";
      (f, parse_type st, fl)
    in
    let sfields = comma_list st RBRACE field in
    DStruct { sname; sfields; sdloc = l }
  | t -> fail st "expected 'fn' or 'struct' at top level, found '%s'" (show t)

let parse_program (src : string) : program =
  let st = { toks = Array.of_list (Lexer.tokenize src); i = 0 } in
  let rec go acc = if peek st = EOF then List.rev acc else go (parse_decl st :: acc) in
  go []

let parse_expr_string (src : string) : expr =
  let st = { toks = Array.of_list (Lexer.tokenize src); i = 0 } in
  let e = parse_expr st in
  if peek st <> EOF then fail st "unexpected '%s' after expression" (show (peek st));
  e
