(* Three-address intermediate representation.

   A function is a list of basic blocks; the first block is the entry.
   Every value is a 64-bit word held in a virtual register ("temp").
   Temps are not in SSA form: a source variable maps to one temp that
   may be assigned many times. Memory is accessed only through
   Load/Store with a base operand and a constant byte offset. *)

type temp = int
type label = int

type operand = T of temp | C of int64

type binop =
  | Add | Sub | Mul | Div | Rem
  | And | Or | Xor | Shl | Shr
  | Lt | Le | Gt | Ge | Eq | Ne
  | Ltu  (* unsigned less-than, used for bounds checks *)

type unop = Neg | Not  (* Not is logical: 0 <-> 1 *)

type callee =
  | Direct of string    (* a Kite function symbol *)
  | Runtime of string   (* a C runtime function *)
  | Indirect of operand (* a closure pointer *)

type instr =
  | Mov of temp * operand
  | Bin of binop * temp * operand * operand
  | Un of unop * temp * operand
  | Load of temp * operand * int            (* d <- mem[base + off] *)
  | Store of operand * int * operand        (* mem[base + off] <- v *)
  | Call of temp option * callee * operand list
  | Addr of temp * string                   (* d <- &symbol *)

type term =
  | Jmp of label
  | Br of operand * label * label           (* if op <> 0 then l1 else l2 *)
  | Ret of operand option
  | Unreachable

type block = { label : label; mutable instrs : instr list; mutable term : term }

type func = {
  name : string;          (* assembly symbol *)
  params : temp list;
  env : temp option;      (* closure environment pointer, for lambdas *)
  mutable blocks : block list;
  mutable ntemps : int;
}

type program = {
  funcs : func list;
  strings : (string * string) list;  (* symbol, contents *)
  closures : string list;            (* functions that need a static closure *)
}

(* ---------- helpers used by the optimiser and backend ---------- *)

let def_of = function
  | Mov (d, _) | Bin (_, d, _, _) | Un (_, d, _) | Load (d, _, _) | Addr (d, _) -> Some d
  | Call (d, _, _) -> d
  | Store _ -> None

let op_temps = function T t -> [ t ] | C _ -> []

let uses_of = function
  | Mov (_, a) | Un (_, _, a) | Load (_, a, _) -> op_temps a
  | Bin (_, _, a, b) -> op_temps a @ op_temps b
  | Store (a, _, b) -> op_temps a @ op_temps b
  | Call (_, c, args) ->
    (match c with Indirect o -> op_temps o | _ -> []) @ List.concat_map op_temps args
  | Addr _ -> []

let term_uses = function
  | Br (o, _, _) -> op_temps o
  | Ret (Some o) -> op_temps o
  | _ -> []

let successors = function
  | Jmp l -> [ l ]
  | Br (_, a, b) -> if a = b then [ a ] else [ a; b ]
  | Ret _ | Unreachable -> []

let is_pure = function
  | Mov _ | Bin _ | Un _ | Load _ | Addr _ -> true
  | Store _ | Call _ -> false

let instr_count (f : func) =
  List.fold_left (fun acc b -> acc + List.length b.instrs + 1) 0 f.blocks

let program_instr_count (p : program) =
  List.fold_left (fun acc f -> acc + instr_count f) 0 p.funcs

(* ---------- printing ---------- *)

let show_op = function T t -> "t" ^ string_of_int t | C c -> Int64.to_string c

let show_binop = function
  | Add -> "add" | Sub -> "sub" | Mul -> "mul" | Div -> "div" | Rem -> "rem"
  | And -> "and" | Or -> "or" | Xor -> "xor" | Shl -> "shl" | Shr -> "shr"
  | Lt -> "lt" | Le -> "le" | Gt -> "gt" | Ge -> "ge" | Eq -> "eq" | Ne -> "ne"
  | Ltu -> "ltu"

let show_callee = function
  | Direct s -> s
  | Runtime s -> s
  | Indirect o -> "*" ^ show_op o

let show_instr = function
  | Mov (d, a) -> Printf.sprintf "t%d = %s" d (show_op a)
  | Bin (op, d, a, b) -> Printf.sprintf "t%d = %s %s, %s" d (show_binop op) (show_op a) (show_op b)
  | Un (Neg, d, a) -> Printf.sprintf "t%d = neg %s" d (show_op a)
  | Un (Not, d, a) -> Printf.sprintf "t%d = not %s" d (show_op a)
  | Load (d, b, o) -> Printf.sprintf "t%d = load [%s + %d]" d (show_op b) o
  | Store (b, o, v) -> Printf.sprintf "store [%s + %d], %s" (show_op b) o (show_op v)
  | Call (d, c, args) ->
    Printf.sprintf "%scall %s(%s)"
      (match d with Some d -> Printf.sprintf "t%d = " d | None -> "")
      (show_callee c)
      (String.concat ", " (List.map show_op args))
  | Addr (d, s) -> Printf.sprintf "t%d = addr %s" d s

let show_term = function
  | Jmp l -> Printf.sprintf "jmp L%d" l
  | Br (o, a, b) -> Printf.sprintf "br %s, L%d, L%d" (show_op o) a b
  | Ret None -> "ret"
  | Ret (Some o) -> "ret " ^ show_op o
  | Unreachable -> "unreachable"

let show_func (f : func) =
  let b = Buffer.create 1024 in
  Buffer.add_string b
    (Printf.sprintf "func %s(%s)%s {\n" f.name
       (String.concat ", " (List.map (fun t -> "t" ^ string_of_int t) f.params))
       (match f.env with Some e -> Printf.sprintf " env t%d" e | None -> ""));
  List.iter
    (fun blk ->
      Buffer.add_string b (Printf.sprintf "L%d:\n" blk.label);
      List.iter (fun i -> Buffer.add_string b ("  " ^ show_instr i ^ "\n")) blk.instrs;
      Buffer.add_string b ("  " ^ show_term blk.term ^ "\n"))
    f.blocks;
  Buffer.add_string b "}\n";
  Buffer.contents b

let show_program (p : program) = String.concat "\n" (List.map show_func p.funcs)
