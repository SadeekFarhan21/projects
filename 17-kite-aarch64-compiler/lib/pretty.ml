(* Pretty-printer. Emits canonical source that re-parses to the same
   AST (modulo positions): parse (print (parse s)) = parse s.
   Parentheses are inserted only where precedence requires them. *)

open Ast

let prec_of = function
  | Or -> 1 | And -> 2
  | Eq | Ne | Lt | Le | Gt | Ge -> 3
  | BOr -> 4 | BXor -> 5 | BAnd -> 6 | Shl | Shr -> 7
  | Add | Sub -> 8 | Mul | Div | Mod -> 9

let op_str = function
  | Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/" | Mod -> "%"
  | Lt -> "<" | Le -> "<=" | Gt -> ">" | Ge -> ">=" | Eq -> "==" | Ne -> "!="
  | And -> "&&" | Or -> "||" | BAnd -> "&" | BOr -> "|" | BXor -> "^"
  | Shl -> "<<" | Shr -> ">>"

let escape s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (function
      | '\n' -> Buffer.add_string b "\\n"
      | '\t' -> Buffer.add_string b "\\t"
      | '\r' -> Buffer.add_string b "\\r"
      | '\000' -> Buffer.add_string b "\\0"
      | '\\' -> Buffer.add_string b "\\\\"
      | '"' -> Buffer.add_string b "\\\""
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"';
  Buffer.contents b

let rec ty_expr = function
  | TEName (s, _) -> s
  | TEArray (t, _) -> "[" ^ ty_expr t ^ "]"
  | TEFn (ps, r, _) ->
    "fn(" ^ String.concat ", " (List.map ty_expr ps) ^ ")"
    ^ (match r with None -> "" | Some t -> " -> " ^ ty_expr t)

let param p = p.pname ^ ": " ^ ty_expr p.pty

(* [ns]: we are in a no-struct-literal context (if/while condition). *)
let rec expr ?(ns = false) ind (e : Ast.expr) : string = expr_p ns ind 0 e

(* [ctx] is the minimum precedence the surrounding context accepts. *)
and expr_p ns ind ctx e =
  match e.desc with
  | IntLit n ->
    (* A negative literal can only come from a hex literal >= 2^63
       (decimal literals are non-negative; `-5` is Neg 5). Printing it
       as "-..." would reparse as a negation, so print it in hex. *)
    if Int64.compare n 0L < 0 then Printf.sprintf "0x%Lx" n else Int64.to_string n
  | BoolLit b -> string_of_bool b
  | StrLit s -> escape s
  | Var (x, _) -> x
  | Binary (op, a, b) ->
    let p = prec_of op in
    let lp, rp = if p = 3 then (p + 1, p + 1) else (p, p + 1) in
    let s = expr_p ns ind lp a ^ " " ^ op_str op ^ " " ^ expr_p ns ind rp b in
    if p < ctx then "(" ^ s ^ ")" else s
  | Unary (op, a) ->
    let s = (match op with Neg -> "-" | Not -> "!") ^ expr_p ns ind 10 a in
    if 10 < ctx then "(" ^ s ^ ")" else s
  | Call (f, args) -> postfix ns ind f ^ "(" ^ String.concat ", " (List.map (expr ind) args) ^ ")"
  | Index (a, i) -> postfix ns ind a ^ "[" ^ expr ind i ^ "]"
  | Field (a, f, _) -> postfix ns ind a ^ "." ^ f
  | ArrayLit es -> "[" ^ String.concat ", " (List.map (expr ind) es) ^ "]"
  | ArrayRepeat (v, n) -> "[" ^ expr ind v ^ "; " ^ expr ind n ^ "]"
  | StructLit (name, fs) ->
    let s =
      name ^ " { "
      ^ String.concat ", " (List.map (fun (f, v) -> f ^ ": " ^ expr ind v) fs)
      ^ (if fs = [] then "}" else " }")
    in
    if ns then "(" ^ s ^ ")" else s
  | Lambda l ->
    let s =
      "fn(" ^ String.concat ", " (List.map param l.lparams) ^ ")"
      ^ (match l.lret with None -> "" | Some t -> " -> " ^ ty_expr t)
      ^ " " ^ block ind l.lbody
    in
    if ctx > 0 then "(" ^ s ^ ")" else s

(* The operand of a postfix operator must be a primary expression. *)
and postfix ns ind e =
  match e.desc with
  | Var _ | Call _ | Index _ | Field _ | StrLit _ | ArrayLit _ | ArrayRepeat _ -> expr_p ns ind 11 e
  | IntLit _ -> expr_p ns ind 11 e
  | BoolLit _ -> expr_p ns ind 11 e
  | StructLit _ when not ns -> expr_p ns ind 11 e
  | _ -> "(" ^ expr_p false ind 0 e ^ ")"

and block ind (b : block) =
  if b = [] then "{}"
  else
    let inner = ind ^ "    " in
    "{\n" ^ String.concat "" (List.map (fun s -> inner ^ stmt inner s ^ "\n") b) ^ ind ^ "}"

and stmt ind (s : stmt) =
  match s.sdesc with
  | Let (m, x, annot, init, _) ->
    (if m then "var " else "let ") ^ x
    ^ (match annot with None -> "" | Some t -> ": " ^ ty_expr t)
    ^ " = " ^ expr ind init ^ ";"
  | Assign (l, r) -> expr ind l ^ " = " ^ expr ind r ^ ";"
  | ExprStmt e ->
    (* A statement starting with `fn(` is a lambda expression, fine;
       a statement starting with `{` would be a block, but struct
       literals start with a name so there is no clash. *)
    expr ind e ^ ";"
  | If (c, t, e) -> if_stmt ind c t e
  | While (c, b) -> "while " ^ expr ~ns:true ind c ^ " " ^ block ind b
  | For (x, lo, hi, b, _) ->
    "for " ^ x ^ " in " ^ expr_p true ind 1 lo ^ ".." ^ expr_p true ind 1 hi ^ " " ^ block ind b
  | Return None -> "return;"
  | Return (Some e) -> "return " ^ expr ind e ^ ";"
  | Break -> "break;"
  | Continue -> "continue;"
  | Block b -> block ind b

and if_stmt ind c t e =
  "if " ^ expr ~ns:true ind c ^ " " ^ block ind t
  ^
  match e with
  | None -> ""
  | Some [ { sdesc = If (c2, t2, e2); _ } ] -> " else " ^ if_stmt ind c2 t2 e2
  | Some b -> " else " ^ block ind b

let decl = function
  | DFn f ->
    "fn " ^ f.fname ^ "(" ^ String.concat ", " (List.map param f.fparams) ^ ")"
    ^ (match f.fret with None -> "" | Some t -> " -> " ^ ty_expr t)
    ^ " " ^ block "" f.fbody
  | DStruct s ->
    "struct " ^ s.sname ^ " {\n"
    ^ String.concat "" (List.map (fun (f, t, _) -> "    " ^ f ^ ": " ^ ty_expr t ^ ",\n") s.sfields)
    ^ "}"

let program (p : program) = String.concat "\n\n" (List.map decl p) ^ "\n"
