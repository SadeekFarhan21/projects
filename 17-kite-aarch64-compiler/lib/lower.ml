(* Lowering from the typed AST to the three-address IR.

   Each source local (by unique id) gets one temp. Expressions are
   flattened into temps left to right; control flow (if, while, for,
   &&, ||, bounds checks, division-by-zero checks) becomes explicit
   blocks and branches. Lambdas are lifted to separate IR functions
   that read their captures from the closure environment. *)

open Ast
module I = Ir

let fn_sym name = "_kf_" ^ name
let lambda_sym n = "_kl_" ^ string_of_int n
let closure_sym name = "_kc_" ^ name

(* ---------- per-program state ---------- *)

type prog_state = {
  mutable strings : (string * string) list;
  mutable nstr : int;
  mutable closures : string list;
  mutable pending : lambda list;  (* lambdas still to be lowered *)
  structs : (string, string list) Hashtbl.t;
}

(* ---------- per-function builder ---------- *)

type fb = {
  ps : prog_state;
  mutable ntemps : int;
  mutable nlabels : int;
  mutable done_blocks : I.block list;          (* reversed *)
  mutable cur_label : I.label;
  mutable cur : I.instr list;                  (* reversed *)
  vars : (int, I.temp) Hashtbl.t;
  mutable loops : (I.label * I.label) list;    (* (break, continue) *)
}

let new_temp fb = let t = fb.ntemps in fb.ntemps <- t + 1; t
let new_label fb = let l = fb.nlabels in fb.nlabels <- l + 1; l
let emit fb i = fb.cur <- i :: fb.cur

(* Close the current block with [t] and start a new one labelled [next]. *)
let finish fb t next =
  fb.done_blocks <- { I.label = fb.cur_label; instrs = List.rev fb.cur; term = t } :: fb.done_blocks;
  fb.cur_label <- next;
  fb.cur <- []

(* After a return/break/continue, code that follows in the same source
   block is unreachable; we still need somewhere to put it. *)
let finish_dead fb t = finish fb t (new_label fb)

let var_temp fb id =
  match Hashtbl.find_opt fb.vars id with
  | Some t -> t
  | None ->
    let t = new_temp fb in
    Hashtbl.replace fb.vars id t;
    t

let string_sym ps s =
  match List.find_opt (fun (_, c) -> c = s) ps.strings with
  | Some (sym, _) -> sym
  | None ->
    let sym = "_ks_" ^ string_of_int ps.nstr in
    ps.nstr <- ps.nstr + 1;
    ps.strings <- (sym, s) :: ps.strings;
    sym

let rt name = I.Runtime name

let binop_ir = function
  | Add -> I.Add | Sub -> I.Sub | Mul -> I.Mul | Div -> I.Div | Mod -> I.Rem
  | BAnd -> I.And | BOr -> I.Or | BXor -> I.Xor | Shl -> I.Shl | Shr -> I.Shr
  | Lt -> I.Lt | Le -> I.Le | Gt -> I.Gt | Ge -> I.Ge | Eq -> I.Eq | Ne -> I.Ne
  | And | Or -> assert false

(* Emit a runtime check: if [cond] is false, call [fail_fn args] which
   never returns. *)
let check fb cond fail_fn args =
  let ok = new_label fb and bad = new_label fb in
  finish fb (I.Br (cond, ok, bad)) bad;
  emit fb (I.Call (None, rt fail_fn, args));
  finish fb I.Unreachable ok

(* Bounds-checked address computation: returns (base, index) after the
   check; element i lives at base + 8 + 8*i. *)
let bounds_check fb arr idx =
  let len = new_temp fb in
  emit fb (I.Load (len, arr, 0));
  let c = new_temp fb in
  emit fb (I.Bin (I.Ltu, c, idx, I.T len));
  check fb (I.T c) "kite_rt_oob" [ idx; I.T len ]

let elem_addr fb arr idx =
  let off = new_temp fb in
  emit fb (I.Bin (I.Shl, off, idx, I.C 3L));
  let p = new_temp fb in
  emit fb (I.Bin (I.Add, p, arr, I.T off));
  I.T p

let rec lower_expr fb (e : expr) : I.operand =
  match e.desc with
  | IntLit n -> I.C n
  | BoolLit b -> I.C (if b then 1L else 0L)
  | StrLit s ->
    let t = new_temp fb in
    emit fb (I.Addr (t, string_sym fb.ps s));
    I.T t
  | Var (_, r) -> (
    match !r with
    | RLocal id -> I.T (var_temp fb id)
    | RFunc f ->
      if not (List.mem f fb.ps.closures) then fb.ps.closures <- f :: fb.ps.closures;
      let t = new_temp fb in
      emit fb (I.Addr (t, closure_sym f));
      I.T t
    | _ -> assert false)
  | Binary ((And | Or | Lt | Le | Gt | Ge), _, _) | Unary (Not, _) -> lower_bool fb e
  | Binary (((Eq | Ne) as op), a, b) when a.ty = TStr ->
    let va = lower_expr fb a in
    let vb = lower_expr fb b in
    let t = new_temp fb in
    emit fb (I.Call (Some t, rt "kite_rt_str_eq", [ va; vb ]));
    if op = Eq then I.T t
    else begin
      let t2 = new_temp fb in
      emit fb (I.Un (I.Not, t2, I.T t));
      I.T t2
    end
  | Binary (Add, a, b) when e.ty = TStr ->
    let va = lower_expr fb a in
    let vb = lower_expr fb b in
    let t = new_temp fb in
    emit fb (I.Call (Some t, rt "kite_rt_str_concat", [ va; vb ]));
    I.T t
  | Binary (((Div | Mod) as op), a, b) ->
    let va = lower_expr fb a in
    let vb = lower_expr fb b in
    let nz = new_temp fb in
    emit fb (I.Bin (I.Ne, nz, vb, I.C 0L));
    check fb (I.T nz) "kite_rt_divzero" [];
    let t = new_temp fb in
    emit fb (I.Bin (binop_ir op, t, va, vb));
    I.T t
  | Binary (op, a, b) ->
    let va = lower_expr fb a in
    let vb = lower_expr fb b in
    let t = new_temp fb in
    emit fb (I.Bin (binop_ir op, t, va, vb));
    I.T t
  | Unary (Neg, a) ->
    let va = lower_expr fb a in
    let t = new_temp fb in
    emit fb (I.Un (I.Neg, t, va));
    I.T t
  | Call ({ desc = Var (_, { contents = RBuiltin b }); _ }, args) -> lower_builtin fb b args
  | Call ({ desc = Var (_, { contents = RFunc f }); _ }, args) ->
    let vs = List.map (lower_expr fb) args in
    call_result fb e.ty (I.Direct (fn_sym f)) vs
  | Call (f, args) ->
    let vf = lower_expr fb f in
    let vs = List.map (lower_expr fb) args in
    call_result fb e.ty (I.Indirect vf) vs
  | Index (a, i) ->
    let va = lower_expr fb a in
    let vi = lower_expr fb i in
    bounds_check fb va vi;
    let p = elem_addr fb va vi in
    let t = new_temp fb in
    emit fb (I.Load (t, p, 8));
    I.T t
  | Field (a, _, idx) ->
    let va = lower_expr fb a in
    let t = new_temp fb in
    emit fb (I.Load (t, va, 8 * !idx));
    I.T t
  | ArrayLit es ->
    let vs = List.map (lower_expr fb) es in
    let n = List.length vs in
    let t = new_temp fb in
    emit fb (I.Call (Some t, rt "kite_rt_alloc", [ I.C (Int64.of_int (8 * (n + 1))) ]));
    emit fb (I.Store (I.T t, 0, I.C (Int64.of_int n)));
    List.iteri (fun i v -> emit fb (I.Store (I.T t, 8 * (i + 1), v))) vs;
    I.T t
  | ArrayRepeat (v, n) ->
    let vv = lower_expr fb v in
    let vn = lower_expr fb n in
    let t = new_temp fb in
    emit fb (I.Call (Some t, rt "kite_rt_array_new", [ vn; vv ]));
    I.T t
  | StructLit (name, fs) ->
    let order = Hashtbl.find fb.ps.structs name in
    let vals = List.map (fun (f, v) -> (f, lower_expr fb v)) fs in
    let t = new_temp fb in
    let n = List.length order in
    emit fb (I.Call (Some t, rt "kite_rt_alloc", [ I.C (Int64.of_int (8 * max 1 n)) ]));
    List.iteri (fun i f -> emit fb (I.Store (I.T t, 8 * i, List.assoc f vals))) order;
    I.T t
  | Lambda lam ->
    fb.ps.pending <- lam :: fb.ps.pending;
    let n = List.length lam.lcaptures in
    let t = new_temp fb in
    emit fb (I.Call (Some t, rt "kite_rt_alloc", [ I.C (Int64.of_int (8 * (n + 1))) ]));
    let code = new_temp fb in
    emit fb (I.Addr (code, lambda_sym lam.lid));
    emit fb (I.Store (I.T t, 0, I.T code));
    List.iteri
      (fun i (id, _) -> emit fb (I.Store (I.T t, 8 * (i + 1), I.T (var_temp fb id))))
      lam.lcaptures;
    I.T t

and call_result fb ty callee args =
  if ty = TUnit then (emit fb (I.Call (None, callee, args)); I.C 0L)
  else begin
    let t = new_temp fb in
    emit fb (I.Call (Some t, callee, args));
    I.T t
  end

(* A boolean-valued expression materialised as 0/1. Comparisons become
   a single Bin; && / || / ! go through lower_cond. *)
and lower_bool fb e =
  match e.desc with
  | Binary (((Lt | Le | Gt | Ge) as op), a, b) ->
    let va = lower_expr fb a in
    let vb = lower_expr fb b in
    let t = new_temp fb in
    emit fb (I.Bin (binop_ir op, t, va, vb));
    I.T t
  | Unary (Not, a) ->
    let va = lower_expr fb a in
    let t = new_temp fb in
    emit fb (I.Un (I.Not, t, va));
    I.T t
  | _ ->
    let t = new_temp fb in
    let lt = new_label fb and lf = new_label fb and join = new_label fb in
    lower_cond fb e lt lf;
    fb.cur_label <- lt;
    emit fb (I.Mov (t, I.C 1L));
    finish fb (I.Jmp join) lf;
    emit fb (I.Mov (t, I.C 0L));
    finish fb (I.Jmp join) join;
    I.T t

(* Branch to [lt] if [e] is true, else to [lf]. Closes the current
   block; the caller must start a new one. Short-circuits && and ||. *)
and lower_cond fb (e : expr) lt lf =
  match e.desc with
  | Binary (And, a, b) ->
    let mid = new_label fb in
    lower_cond fb a mid lf;
    fb.cur_label <- mid;
    lower_cond fb b lt lf
  | Binary (Or, a, b) ->
    let mid = new_label fb in
    lower_cond fb a lt mid;
    fb.cur_label <- mid;
    lower_cond fb b lt lf
  | Unary (Not, a) -> lower_cond fb a lf lt
  | BoolLit true -> finish fb (I.Jmp lt) (new_label fb)
  | BoolLit false -> finish fb (I.Jmp lf) (new_label fb)
  | _ ->
    let v = lower_expr fb e in
    finish fb (I.Br (v, lt, lf)) (new_label fb)

and lower_builtin fb b args =
  let vs () = List.map (lower_expr fb) args in
  let ret name vs =
    let t = new_temp fb in
    emit fb (I.Call (Some t, rt name, vs));
    I.T t
  in
  match b, args with
  | (BPrint | BPrintln), [ a ] ->
    let v = lower_expr fb a in
    let f =
      match a.ty with
      | TInt -> "kite_rt_print_int"
      | TBool -> "kite_rt_print_bool"
      | TStr -> "kite_rt_print_str"
      | _ -> assert false
    in
    emit fb (I.Call (None, rt f, [ v ]));
    if b = BPrintln then emit fb (I.Call (None, rt "kite_rt_print_nl", []));
    I.C 0L
  | BPrintln, [] -> emit fb (I.Call (None, rt "kite_rt_print_nl", [])); I.C 0L
  | BLen, [ a ] ->
    let v = lower_expr fb a in
    let t = new_temp fb in
    emit fb (I.Load (t, v, 0));
    I.T t
  | BToStr, [ a ] ->
    ret (if a.ty = TInt then "kite_rt_int_to_str" else "kite_rt_bool_to_str") (vs ())
  | BSubstr, _ -> ret "kite_rt_substr" (vs ())
  | BChr, _ -> ret "kite_rt_chr" (vs ())
  | BAssert, [ a ] ->
    let v = lower_expr fb a in
    check fb v "kite_rt_assert_fail" [];
    I.C 0L
  | _ -> assert false

and lower_block fb (b : block) = List.iter (lower_stmt fb) b

and lower_stmt fb (s : stmt) =
  match s.sdesc with
  | Let (_, _, _, init, id) ->
    let v = lower_expr fb init in
    emit fb (I.Mov (var_temp fb !id, v))
  | Assign (lhs, rhs) -> (
    match lhs.desc with
    | Var (_, { contents = RLocal id }) ->
      let v = lower_expr fb rhs in
      emit fb (I.Mov (var_temp fb id, v))
    | Index (a, i) ->
      let va = lower_expr fb a in
      let vi = lower_expr fb i in
      let v = lower_expr fb rhs in
      bounds_check fb va vi;
      let p = elem_addr fb va vi in
      emit fb (I.Store (p, 8, v))
    | Field (a, _, idx) ->
      let va = lower_expr fb a in
      let v = lower_expr fb rhs in
      emit fb (I.Store (va, 8 * !idx, v))
    | _ -> assert false)
  | ExprStmt e -> ignore (lower_expr fb e)
  | If (c, t, e) ->
    let lt = new_label fb and lf = new_label fb and join = new_label fb in
    lower_cond fb c lt lf;
    fb.cur_label <- lt;
    lower_block fb t;
    finish fb (I.Jmp join) lf;
    Option.iter (lower_block fb) e;
    finish fb (I.Jmp join) join
  | While (c, body) ->
    let head = new_label fb and bodyl = new_label fb and exit = new_label fb in
    finish fb (I.Jmp head) head;
    lower_cond fb c bodyl exit;
    fb.cur_label <- bodyl;
    fb.loops <- (exit, head) :: fb.loops;
    lower_block fb body;
    fb.loops <- List.tl fb.loops;
    finish fb (I.Jmp head) exit
  | For (_, lo, hi, body, id) ->
    let i = var_temp fb !id in
    let vlo = lower_expr fb lo in
    let vhi = lower_expr fb hi in
    let hi_t = new_temp fb in
    emit fb (I.Mov (i, vlo));
    emit fb (I.Mov (hi_t, vhi));
    let head = new_label fb and bodyl = new_label fb and step = new_label fb
    and exit = new_label fb in
    finish fb (I.Jmp head) head;
    let c = new_temp fb in
    emit fb (I.Bin (I.Lt, c, I.T i, I.T hi_t));
    finish fb (I.Br (I.T c, bodyl, exit)) bodyl;
    fb.loops <- (exit, step) :: fb.loops;
    lower_block fb body;
    fb.loops <- List.tl fb.loops;
    finish fb (I.Jmp step) step;
    emit fb (I.Bin (I.Add, i, I.T i, I.C 1L));
    finish fb (I.Jmp head) exit
  | Return None -> finish_dead fb (I.Ret None)
  | Return (Some e) ->
    let v = lower_expr fb e in
    finish_dead fb (I.Ret (Some v))
  | Break -> finish_dead fb (I.Jmp (fst (List.hd fb.loops)))
  | Continue -> finish_dead fb (I.Jmp (snd (List.hd fb.loops)))
  | Block b -> lower_block fb b

let new_fb ps =
  { ps; ntemps = 0; nlabels = 1; done_blocks = []; cur_label = 0; cur = [];
    vars = Hashtbl.create 32; loops = [] }

let finish_func fb name params env ret_ty =
  (* Falling off the end: unit functions return; for others the type
     checker proved this point unreachable. *)
  finish fb (if ret_ty = TUnit then I.Ret None else I.Unreachable) (-1);
  { I.name; params; env; blocks = List.rev fb.done_blocks; ntemps = fb.ntemps }

let lower_fn ps (f : fn_decl) : I.func =
  let fb = new_fb ps in
  let params = List.map (var_temp fb) f.fparam_ids in
  lower_block fb f.fbody;
  finish_func fb (fn_sym f.fname) params None f.fret_ty

let lower_lambda ps (lam : lambda) : I.func =
  let fb = new_fb ps in
  let env = new_temp fb in
  let params = List.map (var_temp fb) lam.lparam_ids in
  List.iteri
    (fun i (id, _) -> emit fb (I.Load (var_temp fb id, I.T env, 8 * (i + 1))))
    lam.lcaptures;
  lower_block fb lam.lbody;
  finish_func fb (lambda_sym lam.lid) params (Some env) lam.lret_ty

let lower_program (p : program) : I.program =
  let ps = { strings = []; nstr = 0; closures = []; pending = []; structs = Hashtbl.create 16 } in
  List.iter
    (function
      | DStruct s -> Hashtbl.replace ps.structs s.sname (List.map (fun (f, _, _) -> f) s.sfields)
      | DFn _ -> ())
    p;
  let funcs = List.filter_map (function DFn f -> Some (lower_fn ps f) | _ -> None) p in
  let rec drain acc =
    match ps.pending with
    | [] -> List.rev acc
    | lam :: rest ->
      ps.pending <- rest;
      drain (lower_lambda ps lam :: acc)
  in
  let lambdas = drain [] in
  { I.funcs = funcs @ lambdas; strings = List.rev ps.strings; closures = List.rev ps.closures }
