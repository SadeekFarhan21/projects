(* IR optimisation passes.

   1. Constant folding and propagation
      - local: within a block, track temps known to hold a constant or
        a copy of another temp, substitute them into later operands and
        fold instructions whose operands are all constant (plus a few
        algebraic identities such as x*1, x+0, x*0);
      - global: a temp with exactly one definition, where that
        definition is `t = const`, can be replaced by the constant
        everywhere. This is sound because Kite requires every variable
        to be initialised at its declaration and scopes are lexical, so
        the single definition dominates every use.
   2. Dead code elimination: backward liveness over the CFG; remove pure
      instructions whose result is not live.
   3. CFG simplification: fold branches on constants, thread jumps
      through empty blocks, merge straight-line block pairs, and drop
      unreachable blocks.
   4. Copy coalescing: `s = e; d = s` with s dead becomes `d = e`.

   [optimize] runs them in a loop until nothing changes. *)

open Ir

(* ---------- evaluation shared with folding ---------- *)

let fold_bin op a b : int64 option =
  let bool x = Some (if x then 1L else 0L) in
  match op with
  | Add -> Some (Int64.add a b)
  | Sub -> Some (Int64.sub a b)
  | Mul -> Some (Int64.mul a b)
  | Div -> if b = 0L then None else Some (Int64.div a b)
  | Rem -> if b = 0L then None else Some (Int64.rem a b)
  | And -> Some (Int64.logand a b)
  | Or -> Some (Int64.logor a b)
  | Xor -> Some (Int64.logxor a b)
  | Shl -> Some (Int64.shift_left a (Int64.to_int b land 63))
  | Shr -> Some (Int64.shift_right a (Int64.to_int b land 63))
  | Lt -> bool (Int64.compare a b < 0)
  | Le -> bool (Int64.compare a b <= 0)
  | Gt -> bool (Int64.compare a b > 0)
  | Ge -> bool (Int64.compare a b >= 0)
  | Eq -> bool (Int64.equal a b)
  | Ne -> bool (not (Int64.equal a b))
  | Ltu -> bool (Int64.unsigned_compare a b < 0)

(* Algebraic simplification of d = a op b; returns a replacement. *)
let simplify_bin op d a b : instr =
  match op, a, b with
  | _, C x, C y -> (match fold_bin op x y with Some v -> Mov (d, C v) | None -> Bin (op, d, a, b))
  | (Add | Sub | Or | Xor | Shl | Shr), x, C 0L -> Mov (d, x)
  | (Add | Or | Xor), C 0L, x -> Mov (d, x)
  | (Mul | Div), x, C 1L -> Mov (d, x)
  | Mul, C 1L, x -> Mov (d, x)
  | (Mul | And), _, C 0L | (Mul | And), C 0L, _ -> Mov (d, C 0L)
  | Rem, _, C 1L -> Mov (d, C 0L)
  | (Add | Mul | And | Or | Xor | Eq | Ne), C _, T _ -> Bin (op, d, b, a)  (* constant on the right *)
  | _ -> Bin (op, d, a, b)

(* ---------- local constant and copy propagation ---------- *)

let changed = ref false

let local_prop (f : func) =
  List.iter
    (fun blk ->
      (* value of a temp as an operand, if known *)
      let env : (temp, operand) Hashtbl.t = Hashtbl.create 16 in
      let subst o = match o with T t -> (match Hashtbl.find_opt env t with Some v -> v | None -> o) | C _ -> o in
      (* when temp d is redefined, forget facts about d and copies of d *)
      let kill d =
        Hashtbl.remove env d;
        let stale = Hashtbl.fold (fun k v acc -> if v = T d then k :: acc else acc) env [] in
        List.iter (Hashtbl.remove env) stale
      in
      let record d v = match v with
        | C _ -> Hashtbl.replace env d v
        | T s when s <> d -> Hashtbl.replace env d v
        | _ -> ()
      in
      let rewrite i =
        let i' =
          match i with
          | Mov (d, a) -> Mov (d, subst a)
          | Bin (op, d, a, b) -> simplify_bin op d (subst a) (subst b)
          | Un (Neg, d, a) -> (match subst a with C c -> Mov (d, C (Int64.neg c)) | a -> Un (Neg, d, a))
          | Un (Not, d, a) -> (match subst a with C c -> Mov (d, C (Int64.logxor c 1L)) | a -> Un (Not, d, a))
          | Load (d, b, o) -> Load (d, subst b, o)
          | Store (b, o, v) -> Store (subst b, o, subst v)
          | Call (d, c, args) ->
            let c = match c with Indirect o -> Indirect (subst o) | c -> c in
            Call (d, c, List.map subst args)
          | Addr _ -> i
        in
        if i' <> i then changed := true;
        (match def_of i' with Some d -> kill d | None -> ());
        (match i' with Mov (d, v) -> record d v | _ -> ());
        i'
      in
      blk.instrs <- List.map rewrite blk.instrs;
      let t' =
        match blk.term with
        | Br (o, a, b) -> Br (subst o, a, b)
        | Ret (Some o) -> Ret (Some (subst o))
        | t -> t
      in
      if t' <> blk.term then (changed := true; blk.term <- t'))
    f.blocks

(* ---------- global single-definition constant propagation ---------- *)

let global_const_prop (f : func) =
  let ndefs = Hashtbl.create 64 and const = Hashtbl.create 64 in
  let bump t = Hashtbl.replace ndefs t (1 + Option.value ~default:0 (Hashtbl.find_opt ndefs t)) in
  List.iter bump f.params;
  Option.iter bump f.env;
  List.iter
    (fun blk ->
      List.iter
        (fun i ->
          match def_of i with
          | Some d ->
            bump d;
            (match i with Mov (_, C c) -> Hashtbl.replace const d c | _ -> ())
          | None -> ())
        blk.instrs)
    f.blocks;
  let known t = if Hashtbl.find_opt ndefs t = Some 1 then Hashtbl.find_opt const t else None in
  let s o = match o with T t -> (match known t with Some c -> C c | None -> o) | C _ -> o in
  List.iter
    (fun blk ->
      let instrs =
        List.map
          (function
            | Mov (d, a) -> Mov (d, s a)
            | Bin (op, d, a, b) -> Bin (op, d, s a, s b)
            | Un (op, d, a) -> Un (op, d, s a)
            | Load (d, b, o) -> Load (d, s b, o)
            | Store (b, o, v) -> Store (s b, o, s v)
            | Call (d, c, args) ->
              Call (d, (match c with Indirect o -> Indirect (s o) | c -> c), List.map s args)
            | Addr _ as i -> i)
          blk.instrs
      in
      if instrs <> blk.instrs then (changed := true; blk.instrs <- instrs);
      let t' =
        match blk.term with
        | Br (o, a, b) -> Br (s o, a, b)
        | Ret (Some o) -> Ret (Some (s o))
        | t -> t
      in
      if t' <> blk.term then (changed := true; blk.term <- t'))
    f.blocks

(* ---------- liveness ---------- *)

module IS = Set.Make (Int)

let block_map (f : func) =
  let h = Hashtbl.create 32 in
  List.iter (fun b -> Hashtbl.replace h b.label b) f.blocks;
  h

(* Returns live-in and live-out sets per block label. *)
let liveness (f : func) =
  let live_in = Hashtbl.create 32 and live_out = Hashtbl.create 32 in
  let use_def =
    List.map
      (fun b ->
        (* backward: gen = uses before def *)
        let gen = ref (IS.of_list (term_uses b.term)) and kill = ref IS.empty in
        List.iter
          (fun i ->
            (match def_of i with Some d -> gen := IS.remove d !gen; kill := IS.add d !kill | None -> ());
            List.iter (fun u -> gen := IS.add u !gen) (uses_of i))
          (List.rev b.instrs);
        (b, !gen, !kill))
      f.blocks
  in
  List.iter (fun b -> Hashtbl.replace live_in b.label IS.empty; Hashtbl.replace live_out b.label IS.empty) f.blocks;
  let rec iterate () =
    let ch = ref false in
    List.iter
      (fun (b, gen, kill) ->
        let out =
          List.fold_left
            (fun acc s -> IS.union acc (Option.value ~default:IS.empty (Hashtbl.find_opt live_in s)))
            IS.empty (successors b.term)
        in
        let inn = IS.union gen (IS.diff out kill) in
        if not (IS.equal out (Hashtbl.find live_out b.label)) then (ch := true; Hashtbl.replace live_out b.label out);
        if not (IS.equal inn (Hashtbl.find live_in b.label)) then (ch := true; Hashtbl.replace live_in b.label inn))
      (List.rev use_def);
    if !ch then iterate ()
  in
  iterate ();
  (live_in, live_out)

(* ---------- dead code elimination ---------- *)

let dce (f : func) =
  let _, live_out = liveness f in
  List.iter
    (fun b ->
      let live = ref (IS.union (Hashtbl.find live_out b.label) (IS.of_list (term_uses b.term))) in
      let kept =
        List.fold_left
          (fun acc i ->
            let d = def_of i in
            let dead = match d with Some d -> is_pure i && not (IS.mem d !live) | None -> false in
            if dead then (changed := true; acc)
            else begin
              (match d with Some d -> live := IS.remove d !live | None -> ());
              List.iter (fun u -> live := IS.add u !live) (uses_of i);
              i :: acc
            end)
          [] (List.rev b.instrs)
      in
      (* a self-move is also dead *)
      let kept = List.filter (function Mov (d, T s) when d = s -> changed := true; false | _ -> true) kept in
      b.instrs <- kept)
    f.blocks

(* ---------- copy coalescing ---------- *)

(* `s = op ...; d = s` with s dead afterwards becomes `d = op ...`.
   Lowering produces this shape for every `x = expr;` assignment, and
   without it the register allocator has to keep both s and d alive. *)
let with_def i d =
  match i with
  | Mov (_, a) -> Some (Mov (d, a))
  | Bin (op, _, a, b) -> Some (Bin (op, d, a, b))
  | Un (op, _, a) -> Some (Un (op, d, a))
  | Load (_, b, o) -> Some (Load (d, b, o))
  | Addr (_, s) -> Some (Addr (d, s))
  | Call (Some _, c, args) -> Some (Call (Some d, c, args))
  | Call (None, _, _) | Store _ -> None

let coalesce (f : func) =
  let _, live_out = liveness f in
  List.iter
    (fun b ->
      let arr = Array.of_list b.instrs in
      let n = Array.length arr in
      (* live_after.(k) = temps live just after instruction k *)
      let live_after = Array.make n IS.empty in
      let live = ref (IS.union (Hashtbl.find live_out b.label) (IS.of_list (term_uses b.term))) in
      for k = n - 1 downto 0 do
        live_after.(k) <- !live;
        (match def_of arr.(k) with Some d -> live := IS.remove d !live | None -> ());
        List.iter (fun u -> live := IS.add u !live) (uses_of arr.(k))
      done;
      let out = ref [] and k = ref 0 in
      while !k < n do
        (if !k + 1 < n then
           match arr.(!k + 1), def_of arr.(!k) with
           | Mov (d, T s), Some s' when s = s' && d <> s && not (IS.mem s live_after.(!k + 1)) -> (
             match with_def arr.(!k) d with
             | Some i ->
               changed := true;
               out := i :: !out;
               k := !k + 2
             | None -> out := arr.(!k) :: !out; incr k)
           | _ -> out := arr.(!k) :: !out; incr k
         else (out := arr.(!k) :: !out; incr k))
      done;
      b.instrs <- List.rev !out)
    f.blocks

(* ---------- CFG simplification ---------- *)

let simplify_cfg (f : func) =
  let bm = block_map f in
  (* fold constant / trivial branches *)
  List.iter
    (fun b ->
      match b.term with
      | Br (C c, l1, l2) -> changed := true; b.term <- Jmp (if c <> 0L then l1 else l2)
      | Br (_, l1, l2) when l1 = l2 -> changed := true; b.term <- Jmp l1
      | _ -> ())
    f.blocks;
  (* jump threading through empty blocks that just jump *)
  let rec target l seen =
    match Hashtbl.find_opt bm l with
    | Some { instrs = []; term = Jmp l'; _ } when not (List.mem l' seen) -> target l' (l :: seen)
    | _ -> l
  in
  let entry = (List.hd f.blocks).label in
  List.iter
    (fun b ->
      let t' =
        match b.term with
        | Jmp l -> Jmp (target l [])
        | Br (o, l1, l2) -> Br (o, target l1 [], target l2 [])
        | t -> t
      in
      if t' <> b.term then (changed := true; b.term <- t'))
    f.blocks;
  (* remove unreachable blocks *)
  let reach = Hashtbl.create 32 in
  let rec visit l =
    if not (Hashtbl.mem reach l) then begin
      Hashtbl.replace reach l ();
      match Hashtbl.find_opt bm l with Some b -> List.iter visit (successors b.term) | None -> ()
    end
  in
  visit entry;
  let before = List.length f.blocks in
  f.blocks <- List.filter (fun b -> Hashtbl.mem reach b.label) f.blocks;
  if List.length f.blocks <> before then changed := true;
  (* merge b -> c when b ends in Jmp c and c has exactly one predecessor *)
  let preds = Hashtbl.create 32 in
  List.iter
    (fun b -> List.iter (fun s -> Hashtbl.replace preds s (1 + Option.value ~default:0 (Hashtbl.find_opt preds s))) (successors b.term))
    f.blocks;
  let bm = block_map f in
  let removed = Hashtbl.create 8 in
  List.iter
    (fun b ->
      if not (Hashtbl.mem removed b.label) then begin
        let rec absorb () =
          match b.term with
          | Jmp c when c <> b.label && c <> entry && Hashtbl.find_opt preds c = Some 1
                       && not (Hashtbl.mem removed c) ->
            let cb = Hashtbl.find bm c in
            b.instrs <- b.instrs @ cb.instrs;
            b.term <- cb.term;
            Hashtbl.replace removed c ();
            changed := true;
            absorb ()
          | _ -> ()
        in
        absorb ()
      end)
    f.blocks;
  f.blocks <- List.filter (fun b -> not (Hashtbl.mem removed b.label)) f.blocks

(* ---------- driver ---------- *)

type stats = { mutable rounds : int }

let optimize_func (f : func) =
  let rounds = ref 0 in
  let continue_ = ref true in
  while !continue_ && !rounds < 20 do
    changed := false;
    local_prop f;
    global_const_prop f;
    simplify_cfg f;
    dce f;
    coalesce f;
    incr rounds;
    continue_ := !changed
  done;
  if Sys.getenv_opt "KITE_OPT_STATS" <> None then
    Printf.eprintf "opt: %s settled after %d rounds\n" f.name !rounds

let optimize (p : program) : program =
  List.iter optimize_func p.funcs;
  p
