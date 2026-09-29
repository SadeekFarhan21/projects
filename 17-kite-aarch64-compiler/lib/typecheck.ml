(* Name resolution and type checking.

   One pass over the AST that
   - resolves every identifier to a local (unique id), a top-level
     function, or a builtin;
   - infers the type of every expression bottom-up (local type
     inference: `let x = e;` takes the type of e, annotations optional);
   - computes the capture list of every lambda;
   - enforces the static rules in SPEC.md (mutability, capture of
     immutable bindings only, definite return, break inside loops...).

   It mutates the annotation fields of the AST in place, so the result
   is the "typed AST" consumed by the interpreter and the lowering. *)

open Ast
open Diag

let err loc fmt = error Type loc fmt

(* Where an expression starts in the source. Binary and postfix nodes
   carry the position of their operator, which is what we want when the
   operator itself is at fault; when the whole expression is blamed we
   point at its first character instead. *)
let rec start (e : expr) =
  match e.desc with
  | Binary (_, a, _) | Index (a, _) | Field (a, _, _) | Call (a, _) -> start a
  | _ -> e.loc

type local = { id : int; lty : ty; mut : bool; level : int }

type fn_sig = { params : ty list; ret : ty }

type env = {
  structs : (string, (string * ty) list) Hashtbl.t;
  funcs : (string, fn_sig) Hashtbl.t;
  mutable scopes : (string * local) list list;  (* innermost first *)
  mutable level : int;                          (* lambda nesting depth *)
  mutable lambdas : lambda list;                (* enclosing lambdas, innermost first *)
  mutable ret : ty list;                        (* expected return type stack *)
  mutable loops : int;                          (* loop depth in current function *)
}

let next_id = ref 0
let fresh () = incr next_id; !next_id
let lambda_counter = ref 0

(* Types of every local id; used by later stages. *)
let local_types : (int, ty) Hashtbl.t = Hashtbl.create 64

let rec resolve_ty env (t : ty_expr) : ty =
  match t with
  | TEName ("int", _) -> TInt
  | TEName ("bool", _) -> TBool
  | TEName ("string", _) -> TStr
  | TEName (s, l) ->
    if Hashtbl.mem env.structs s then TStruct s else err l "unknown type `%s`" s
  | TEArray (t, _) -> TArray (resolve_ty env t)
  | TEFn (ps, r, _) ->
    TFun (List.map (resolve_ty env) ps, match r with None -> TUnit | Some t -> resolve_ty env t)

let push_scope env = env.scopes <- [] :: env.scopes
let pop_scope env = env.scopes <- List.tl env.scopes

let check_not_builtin loc name =
  if List.mem_assoc name builtins then err loc "`%s` is a builtin and cannot be redefined" name

let declare env loc name ty mut =
  check_not_builtin loc name;
  if ty = TUnit then err loc "cannot bind `%s` to a value of type unit" name;
  let id = fresh () in
  Hashtbl.replace local_types id ty;
  (match env.scopes with
   | s :: rest -> env.scopes <- ((name, { id; lty = ty; mut; level = env.level }) :: s) :: rest
   | [] -> assert false);
  id

let lookup_local env name =
  let rec go = function
    | [] -> None
    | s :: rest -> (match List.assoc_opt name s with Some l -> Some l | None -> go rest)
  in
  go env.scopes

(* Record that the lambdas between [l.level] and the current level
   capture local [l]. *)
let note_capture env loc name (l : local) =
  if l.level < env.level then begin
    if l.mut then
      err loc "closure cannot capture mutable variable `%s` (declare it with `let`)" name;
    List.iteri
      (fun i lam ->
        (* lambda at index i has level env.level - i *)
        if env.level - i > l.level && not (List.mem_assoc l.id lam.lcaptures) then
          lam.lcaptures <- lam.lcaptures @ [ (l.id, l.lty) ])
      env.lambdas
  end

let expect_ty loc ~what expected actual =
  if expected <> actual then
    err loc "%s: expected %s, found %s" what (show_ty expected) (show_ty actual)

let is_eq_type = function TInt | TBool | TStr -> true | _ -> false

let rec check_expr env (e : expr) : ty =
  let t = infer env e in
  e.ty <- t;
  t

and infer env (e : expr) : ty =
  match e.desc with
  | IntLit _ -> TInt
  | BoolLit _ -> TBool
  | StrLit _ -> TStr
  | Var (name, res) -> (
    match lookup_local env name with
    | Some l ->
      note_capture env e.loc name l;
      res := RLocal l.id;
      l.lty
    | None -> (
      match Hashtbl.find_opt env.funcs name with
      | Some s ->
        res := RFunc name;
        TFun (s.params, s.ret)
      | None ->
        if List.mem_assoc name builtins then
          err e.loc "builtin `%s` can only be called, not used as a value" name
        else err e.loc "unknown variable `%s`" name))
  | Binary (op, a, b) -> (
    let ta = check_expr env a in
    let tb = check_expr env b in
    let both t what =
      if ta <> t then err (start a) "left operand of %s: expected %s, found %s" what (show_ty t) (show_ty ta);
      if tb <> t then err (start b) "right operand of %s: expected %s, found %s" what (show_ty t) (show_ty tb)
    in
    match op with
    | Add ->
      if ta = TStr && tb = TStr then TStr
      else if ta = TInt && tb = TInt then TInt
      else err e.loc "operator + needs two ints or two strings, found %s and %s" (show_ty ta) (show_ty tb)
    | Sub | Mul | Div | Mod | BAnd | BOr | BXor | Shl | Shr ->
      both TInt ("'" ^ Pretty.op_str op ^ "'"); TInt
    | Lt | Le | Gt | Ge -> both TInt ("'" ^ Pretty.op_str op ^ "'"); TBool
    | Eq | Ne ->
      if ta <> tb then
        err e.loc "cannot compare %s with %s" (show_ty ta) (show_ty tb);
      if not (is_eq_type ta) then
        err e.loc "values of type %s cannot be compared with %s" (show_ty ta) (Pretty.op_str op);
      TBool
    | And | Or -> both TBool ("'" ^ Pretty.op_str op ^ "'"); TBool)
  | Unary (Neg, a) ->
    expect_ty (start a) ~what:"operand of unary -" TInt (check_expr env a); TInt
  | Unary (Not, a) ->
    expect_ty (start a) ~what:"operand of !" TBool (check_expr env a); TBool
  | Call ({ desc = Var (name, res); loc; _ } as f, args)
    when lookup_local env name = None && not (Hashtbl.mem env.funcs name)
         && List.mem_assoc name builtins ->
    let b = List.assoc name builtins in
    res := RBuiltin b;
    f.ty <- TUnknown;
    check_builtin env e.loc name b args
  | Call (f, args) -> (
    let tf = check_expr env f in
    match tf with
    | TFun (ps, r) ->
      if List.length ps <> List.length args then
        err (start e) "function expects %d argument%s but %d %s given" (List.length ps)
          (if List.length ps = 1 then "" else "s")
          (List.length args) (if List.length args = 1 then "was" else "were");
      List.iteri
        (fun i (p, a) ->
          let ta = check_expr env a in
          if ta <> p then
            err (start a) "argument %d: expected %s, found %s" (i + 1) (show_ty p) (show_ty ta))
        (List.combine ps args);
      r
    | t -> err f.loc "cannot call a value of type %s" (show_ty t))
  | Index (a, i) -> (
    let ta = check_expr env a in
    expect_ty (start i) ~what:"array index" TInt (check_expr env i);
    match ta with
    | TArray t -> t
    | TStr -> err e.loc "strings cannot be indexed with []; use substr(s, i, 1)"
    | t -> err a.loc "cannot index a value of type %s" (show_ty t))
  | Field (a, f, idx) -> (
    match check_expr env a with
    | TStruct s ->
      let fields = Hashtbl.find env.structs s in
      let rec find i = function
        | [] -> err e.loc "struct %s has no field `%s`" s f
        | (n, t) :: rest -> if n = f then (idx := i; t) else find (i + 1) rest
      in
      find 0 fields
    | t -> err a.loc "cannot access field `%s` on a value of type %s" f (show_ty t))
  | ArrayLit es -> (
    match es with
    | [] -> err e.loc "empty array literal"
    | first :: rest ->
      let t = check_expr env first in
      if t = TUnit then err first.loc "array elements cannot have type unit";
      List.iter
        (fun x ->
          let tx = check_expr env x in
          if tx <> t then
            err (start x) "array element: expected %s, found %s" (show_ty t) (show_ty tx))
        rest;
      TArray t)
  | ArrayRepeat (v, n) ->
    let t = check_expr env v in
    if t = TUnit then err v.loc "array elements cannot have type unit";
    expect_ty (start n) ~what:"array length" TInt (check_expr env n);
    TArray t
  | StructLit (name, fs) ->
    let decl =
      match Hashtbl.find_opt env.structs name with
      | Some d -> d
      | None -> err e.loc "unknown struct `%s`" name
    in
    List.iter
      (fun (f, v) ->
        match List.assoc_opt f decl with
        | None -> err v.loc "struct %s has no field `%s`" name f
        | Some t ->
          let tv = check_expr env v in
          if tv <> t then
            err (start v) "field `%s` of %s: expected %s, found %s" f name (show_ty t) (show_ty tv))
      fs;
    let seen = Hashtbl.create 8 in
    List.iter
      (fun (f, v) ->
        if Hashtbl.mem seen f then err v.loc "field `%s` given twice" f;
        Hashtbl.add seen f ())
      fs;
    List.iter
      (fun (f, _) -> if not (Hashtbl.mem seen f) then err e.loc "missing field `%s` in %s literal" f name)
      decl;
    TStruct name
  | Lambda lam ->
    if List.length lam.lparams > 8 then err e.loc "functions may take at most 8 parameters";
    incr lambda_counter;
    lam.lid <- !lambda_counter;
    lam.lcaptures <- [];
    let ptys = List.map (fun p -> resolve_ty env p.pty) lam.lparams in
    let rt = match lam.lret with None -> TUnit | Some t -> resolve_ty env t in
    lam.lret_ty <- rt;
    let saved_loops = env.loops in
    env.level <- env.level + 1;
    env.lambdas <- lam :: env.lambdas;
    env.ret <- rt :: env.ret;
    env.loops <- 0;
    push_scope env;
    lam.lparam_ids <- List.map2 (fun p t -> declare env p.ploc p.pname t false) lam.lparams ptys;
    check_params_unique lam.lparams;
    check_block env lam.lbody;
    if rt <> TUnit && not (block_returns lam.lbody) then
      err e.loc "closure may reach the end without returning a value of type %s" (show_ty rt);
    pop_scope env;
    env.loops <- saved_loops;
    env.ret <- List.tl env.ret;
    env.lambdas <- List.tl env.lambdas;
    env.level <- env.level - 1;
    (* Captures of this lambda that come from even further out must also
       be captured by the enclosing lambda; note_capture handled that at
       each use site. *)
    TFun (ptys, rt)

and check_builtin env loc name b args =
  let arity n =
    if List.length args <> n then
      err loc "%s expects %d argument%s, got %d" name n (if n = 1 then "" else "s") (List.length args)
  in
  let ts () = List.map (check_expr env) args in
  match b with
  | BPrint | BPrintln -> (
    match args with
    | [] when b = BPrintln -> TUnit
    | [ a ] ->
      let t = check_expr env a in
      if not (is_eq_type t) then err (start a) "%s cannot print a value of type %s" name (show_ty t);
      TUnit
    | _ -> err loc "%s expects 1 argument, got %d" name (List.length args))
  | BLen -> (
    arity 1;
    match ts () with
    | [ (TArray _ | TStr) ] -> TInt
    | [ t ] -> err (List.hd args).loc "len expects an array or string, found %s" (show_ty t)
    | _ -> assert false)
  | BToStr -> (
    arity 1;
    match ts () with
    | [ (TInt | TBool) ] -> TStr
    | [ t ] -> err (List.hd args).loc "to_str expects int or bool, found %s" (show_ty t)
    | _ -> assert false)
  | BSubstr ->
    arity 3;
    List.iteri
      (fun i (a, want) -> expect_ty (start a) ~what:(Printf.sprintf "substr argument %d" (i + 1)) want (check_expr env a))
      (List.combine args [ TStr; TInt; TInt ]);
    TStr
  | BChr ->
    arity 1;
    expect_ty (start (List.hd args)) ~what:"chr argument" TInt (check_expr env (List.hd args));
    TStr
  | BAssert ->
    arity 1;
    expect_ty (start (List.hd args)) ~what:"assert argument" TBool (check_expr env (List.hd args));
    TUnit

and check_params_unique ps =
  let seen = Hashtbl.create 8 in
  List.iter
    (fun p ->
      if Hashtbl.mem seen p.pname then err p.ploc "duplicate parameter `%s`" p.pname;
      Hashtbl.add seen p.pname ())
    ps

and check_block env (b : block) =
  push_scope env;
  List.iter (check_stmt env) b;
  pop_scope env

and check_stmt env (s : stmt) =
  match s.sdesc with
  | Let (mut, name, annot, init, idr) ->
    let t = check_expr env init in
    (match annot with
     | Some a ->
       let at = resolve_ty env a in
       if at <> t then
         err (start init) "`%s` is declared as %s but initialised with %s" name (show_ty at) (show_ty t)
     | None -> ());
    if t = TUnit then err (start init) "cannot bind `%s` to a value of type unit" name;
    idr := declare env s.sloc name t mut
  | Assign (lhs, rhs) ->
    let tl =
      match lhs.desc with
      | Var (name, res) -> (
        match lookup_local env name with
        | Some l ->
          if l.level < env.level then
            err lhs.loc "cannot assign to captured variable `%s` inside a closure" name;
          if not l.mut then
            err lhs.loc "cannot assign to `%s`: it was declared with `let` (use `var`)" name;
          res := RLocal l.id;
          lhs.ty <- l.lty;
          l.lty
        | None ->
          if Hashtbl.mem env.funcs name then err lhs.loc "cannot assign to function `%s`" name
          else err lhs.loc "unknown variable `%s`" name)
      | Index _ | Field _ -> check_expr env lhs
      | _ -> err lhs.loc "invalid assignment target"
    in
    let tr = check_expr env rhs in
    if tl <> tr then err (start rhs) "cannot assign %s to a place of type %s" (show_ty tr) (show_ty tl)
  | ExprStmt e -> ignore (check_expr env e)
  | If (c, t, e) ->
    expect_ty (start c) ~what:"if condition" TBool (check_expr env c);
    check_block env t;
    Option.iter (check_block env) e
  | While (c, b) ->
    expect_ty (start c) ~what:"while condition" TBool (check_expr env c);
    env.loops <- env.loops + 1;
    check_block env b;
    env.loops <- env.loops - 1
  | For (x, lo, hi, b, idr) ->
    expect_ty (start lo) ~what:"range start" TInt (check_expr env lo);
    expect_ty (start hi) ~what:"range end" TInt (check_expr env hi);
    push_scope env;
    idr := declare env s.sloc x TInt false;
    env.loops <- env.loops + 1;
    check_block env b;
    env.loops <- env.loops - 1;
    pop_scope env
  | Return eo -> (
    let rt = List.hd env.ret in
    match eo with
    | None -> if rt <> TUnit then err s.sloc "missing return value of type %s" (show_ty rt)
    | Some e ->
      let t = check_expr env e in
      if rt = TUnit then err (start e) "this function returns no value, but a value of type %s was returned" (show_ty t);
      if t <> rt then err (start e) "return type mismatch: expected %s, found %s" (show_ty rt) (show_ty t))
  | Break -> if env.loops = 0 then err s.sloc "`break` outside of a loop"
  | Continue -> if env.loops = 0 then err s.sloc "`continue` outside of a loop"
  | Block b -> check_block env b

(* Does every path through the block end in a return? *)
and block_returns (b : block) =
  List.exists
    (fun s ->
      match s.sdesc with
      | Return _ -> true
      | If (_, t, Some e) -> block_returns t && block_returns e
      | Block b -> block_returns b
      | _ -> false)
    b

let check_program (p : program) : unit =
  next_id := 0;
  lambda_counter := 0;
  Hashtbl.reset local_types;
  let env =
    { structs = Hashtbl.create 16; funcs = Hashtbl.create 16; scopes = [ [] ]; level = 0;
      lambdas = []; ret = []; loops = 0 }
  in
  (* Pass 1: struct names (so fields may mention any struct). *)
  List.iter
    (function
      | DStruct s ->
        check_not_builtin s.sdloc s.sname;
        if List.mem s.sname [ "int"; "bool"; "string" ] then
          err s.sdloc "`%s` is a builtin type name" s.sname;
        if Hashtbl.mem env.structs s.sname then err s.sdloc "struct `%s` is defined twice" s.sname;
        Hashtbl.replace env.structs s.sname []
      | DFn _ -> ())
    p;
  (* Pass 2: struct fields. *)
  List.iter
    (function
      | DStruct s ->
        let seen = Hashtbl.create 8 in
        let fields =
          List.map
            (fun (f, t, l) ->
              if Hashtbl.mem seen f then err l "duplicate field `%s` in struct %s" f s.sname;
              Hashtbl.add seen f ();
              (f, resolve_ty env t))
            s.sfields
        in
        Hashtbl.replace env.structs s.sname fields
      | DFn _ -> ())
    p;
  (* Pass 3: function signatures (so functions may be mutually recursive). *)
  List.iter
    (function
      | DFn f ->
        check_not_builtin f.floc f.fname;
        if Hashtbl.mem env.funcs f.fname then err f.floc "function `%s` is defined twice" f.fname;
        if List.length f.fparams > 8 then err f.floc "functions may take at most 8 parameters";
        let ps = List.map (fun p -> resolve_ty env p.pty) f.fparams in
        let r = match f.fret with None -> TUnit | Some t -> resolve_ty env t in
        f.fparam_tys <- ps;
        f.fret_ty <- r;
        Hashtbl.replace env.funcs f.fname { params = ps; ret = r }
      | DStruct _ -> ())
    p;
  (* Pass 4: bodies. *)
  List.iter
    (function
      | DFn f ->
        env.ret <- [ f.fret_ty ];
        env.loops <- 0;
        env.scopes <- [ [] ];
        check_params_unique f.fparams;
        f.fparam_ids <- List.map2 (fun p t -> declare env p.ploc p.pname t false) f.fparams f.fparam_tys;
        check_block env f.fbody;
        if f.fret_ty <> TUnit && not (block_returns f.fbody) then
          err f.floc "function `%s` may reach the end without returning a value of type %s" f.fname
            (show_ty f.fret_ty)
      | DStruct _ -> ())
    p;
  match Hashtbl.find_opt env.funcs "main" with
  | None -> err { line = 1; col = 1 } "program has no `main` function"
  | Some { params = []; ret = TUnit } -> ()
  | Some _ ->
    let loc = List.find_map (function DFn f when f.fname = "main" -> Some f.floc | _ -> None) p in
    err (Option.get loc) "`main` must take no parameters and return nothing"
