(* Reference tree-walking interpreter over the typed AST.

   This is the test oracle: every test program is run here and as a
   native executable, and the two must agree byte for byte on stdout,
   stderr and exit status. It is deliberately simple and direct; it
   shares no code with the IR, optimiser or backend. *)

open Ast

type value =
  | VInt of int64
  | VBool of bool
  | VStr of string
  | VArr of value array
  | VStruct of value array
  | VFunc of string                            (* top-level function *)
  | VClo of lambda * (int * value) list       (* lambda + captured values *)
  | VUnit

exception Runtime_error of string
exception Break_exn
exception Continue_exn
exception Return_exn of value

type ctx = {
  funcs : (string, fn_decl) Hashtbl.t;
  out : Buffer.t;
}

let fail fmt = Printf.ksprintf (fun s -> raise (Runtime_error s)) fmt

let as_int = function VInt n -> n | _ -> assert false
let as_bool = function VBool b -> b | _ -> assert false
let as_str = function VStr s -> s | _ -> assert false

let show_value = function
  | VInt n -> Int64.to_string n
  | VBool b -> string_of_bool b
  | VStr s -> s
  | _ -> assert false

let arith op a b =
  match op with
  | Add -> Int64.add a b
  | Sub -> Int64.sub a b
  | Mul -> Int64.mul a b
  | Div -> if b = 0L then fail "division by zero" else Int64.div a b
  | Mod -> if b = 0L then fail "division by zero" else Int64.rem a b
  | BAnd -> Int64.logand a b
  | BOr -> Int64.logor a b
  | BXor -> Int64.logxor a b
  | Shl -> Int64.shift_left a (Int64.to_int b land 63)
  | Shr -> Int64.shift_right a (Int64.to_int b land 63)
  | _ -> assert false

let rec eval ctx (env : (int, value) Hashtbl.t) (e : expr) : value =
  match e.desc with
  | IntLit n -> VInt n
  | BoolLit b -> VBool b
  | StrLit s -> VStr s
  | Var (_, r) -> (
    match !r with
    | RLocal id -> Hashtbl.find env id
    | RFunc f -> VFunc f
    | _ -> assert false)
  | Binary (And, a, b) -> if as_bool (eval ctx env a) then eval ctx env b else VBool false
  | Binary (Or, a, b) -> if as_bool (eval ctx env a) then VBool true else eval ctx env b
  | Binary (op, a, b) -> (
    let va = eval ctx env a in
    let vb = eval ctx env b in
    match op, va, vb with
    | Add, VStr x, VStr y -> VStr (x ^ y)
    | (Eq | Ne), _, _ ->
      let eq =
        match va, vb with
        | VInt x, VInt y -> Int64.equal x y
        | VBool x, VBool y -> x = y
        | VStr x, VStr y -> String.equal x y
        | _ -> assert false
      in
      VBool (if op = Eq then eq else not eq)
    | Lt, VInt x, VInt y -> VBool (Int64.compare x y < 0)
    | Le, VInt x, VInt y -> VBool (Int64.compare x y <= 0)
    | Gt, VInt x, VInt y -> VBool (Int64.compare x y > 0)
    | Ge, VInt x, VInt y -> VBool (Int64.compare x y >= 0)
    | _, VInt x, VInt y -> VInt (arith op x y)
    | _ -> assert false)
  | Unary (Neg, a) -> VInt (Int64.neg (as_int (eval ctx env a)))
  | Unary (Not, a) -> VBool (not (as_bool (eval ctx env a)))
  | Call ({ desc = Var (_, { contents = RBuiltin b }); _ }, args) ->
    builtin ctx b (List.map (eval ctx env) args)
  | Call (f, args) ->
    let fv = eval ctx env f in
    let avs = List.map (eval ctx env) args in
    call ctx fv avs
  | Index (a, i) ->
    let arr = (match eval ctx env a with VArr x -> x | _ -> assert false) in
    let i = as_int (eval ctx env i) in
    check_index arr i;
    arr.(Int64.to_int i)
  | Field (a, _, idx) -> (
    match eval ctx env a with VStruct fs -> fs.(!idx) | _ -> assert false)
  | ArrayLit es -> VArr (Array.of_list (List.map (eval ctx env) es))
  | ArrayRepeat (v, n) ->
    let v = eval ctx env v in
    let n = as_int (eval ctx env n) in
    if Int64.compare n 0L < 0 then fail "negative array length: %Ld" n;
    VArr (Array.make (Int64.to_int n) v)
  | StructLit (_, fs) ->
    (* Fields are evaluated in source order and stored in declaration
       order. *)
    let decl_fields = struct_fields e.ty in
    let vals = List.map (fun (f, v) -> (f, eval ctx env v)) fs in
    VStruct (Array.of_list (List.map (fun f -> List.assoc f vals) decl_fields))
  | Lambda lam -> VClo (lam, List.map (fun (id, _) -> (id, Hashtbl.find env id)) lam.lcaptures)

and struct_fields ty =
  match ty with
  | TStruct s -> Hashtbl.find struct_table s
  | _ -> assert false

and struct_table : (string, string list) Hashtbl.t = Hashtbl.create 16

and check_index arr i =
  let len = Array.length arr in
  if Int64.compare i 0L < 0 || Int64.compare i (Int64.of_int len) >= 0 then
    fail "index out of bounds: index %Ld, length %d" i len

and call ctx fv avs =
  match fv with
  | VFunc name ->
    let f = Hashtbl.find ctx.funcs name in
    let env = Hashtbl.create 16 in
    List.iter2 (fun id v -> Hashtbl.replace env id v) f.fparam_ids avs;
    run_body ctx env f.fbody
  | VClo (lam, caps) ->
    let env = Hashtbl.create 16 in
    List.iter (fun (id, v) -> Hashtbl.replace env id v) caps;
    List.iter2 (fun id v -> Hashtbl.replace env id v) lam.lparam_ids avs;
    run_body ctx env lam.lbody
  | _ -> assert false

and run_body ctx env body =
  try exec_block ctx env body; VUnit with Return_exn v -> v

and builtin ctx b args =
  match b, args with
  | BPrint, [ v ] -> Buffer.add_string ctx.out (show_value v); VUnit
  | BPrintln, [ v ] -> Buffer.add_string ctx.out (show_value v); Buffer.add_char ctx.out '\n'; VUnit
  | BPrintln, [] -> Buffer.add_char ctx.out '\n'; VUnit
  | BLen, [ VArr a ] -> VInt (Int64.of_int (Array.length a))
  | BLen, [ VStr s ] -> VInt (Int64.of_int (String.length s))
  | BToStr, [ v ] -> VStr (show_value v)
  | BSubstr, [ VStr s; VInt st; VInt n ] ->
    let len = Int64.of_int (String.length s) in
    if Int64.compare st 0L < 0 || Int64.compare n 0L < 0 || Int64.compare st len > 0
       || Int64.compare n (Int64.sub len st) > 0
    then fail "substr out of bounds: start %Ld, count %Ld, length %Ld" st n len;
    VStr (String.sub s (Int64.to_int st) (Int64.to_int n))
  | BChr, [ VInt n ] ->
    if Int64.compare n 0L < 0 || Int64.compare n 255L > 0 then fail "chr argument out of range: %Ld" n;
    VStr (String.make 1 (Char.chr (Int64.to_int n)))
  | BAssert, [ VBool b ] -> if not b then fail "assertion failed"; VUnit
  | _ -> assert false

and exec_block ctx env b = List.iter (exec ctx env) b

and exec ctx env (s : stmt) =
  match s.sdesc with
  | Let (_, _, _, init, id) -> Hashtbl.replace env !id (eval ctx env init)
  | Assign (lhs, rhs) -> (
    match lhs.desc with
    | Var (_, { contents = RLocal id }) -> Hashtbl.replace env id (eval ctx env rhs)
    | Index (a, i) ->
      let arr = (match eval ctx env a with VArr x -> x | _ -> assert false) in
      let i = as_int (eval ctx env i) in
      let v = eval ctx env rhs in
      check_index arr i;
      arr.(Int64.to_int i) <- v
    | Field (a, _, idx) -> (
      match eval ctx env a with
      | VStruct fs -> fs.(!idx) <- eval ctx env rhs
      | _ -> assert false)
    | _ -> assert false)
  | ExprStmt e -> ignore (eval ctx env e)
  | If (c, t, e) ->
    if as_bool (eval ctx env c) then exec_block ctx env t
    else Option.iter (exec_block ctx env) e
  | While (c, b) -> (
    try
      while as_bool (eval ctx env c) do
        try exec_block ctx env b with Continue_exn -> ()
      done
    with Break_exn -> ())
  | For (_, lo, hi, b, id) -> (
    let lo = as_int (eval ctx env lo) in
    let hi = as_int (eval ctx env hi) in
    let i = ref lo in
    try
      while Int64.compare !i hi < 0 do
        Hashtbl.replace env !id (VInt !i);
        (try exec_block ctx env b with Continue_exn -> ());
        i := Int64.add !i 1L
      done
    with Break_exn -> ())
  | Return None -> raise (Return_exn VUnit)
  | Return (Some e) -> raise (Return_exn (eval ctx env e))
  | Break -> raise Break_exn
  | Continue -> raise Continue_exn
  | Block b -> exec_block ctx env b

(* Run a type-checked program. Returns (stdout, exit status, stderr). *)
let run (p : program) : string * int * string =
  let funcs = Hashtbl.create 16 in
  Hashtbl.reset struct_table;
  List.iter
    (function
      | DFn f -> Hashtbl.replace funcs f.fname f
      | DStruct s -> Hashtbl.replace struct_table s.sname (List.map (fun (f, _, _) -> f) s.sfields))
    p;
  let ctx = { funcs; out = Buffer.create 4096 } in
  try
    ignore (call ctx (VFunc "main") []);
    (Buffer.contents ctx.out, 0, "")
  with Runtime_error msg -> (Buffer.contents ctx.out, 101, "runtime error: " ^ msg ^ "\n")
