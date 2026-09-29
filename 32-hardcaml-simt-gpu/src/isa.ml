(* The instruction set of the GPU.

   Every instruction is 16 bits wide. The opcode lives in bits [15:12].

     NOP                 0000 ---- ---- ----
     BRnzp nzp, target   0001 nzp- tttt tttt   pc <- target if (NZP & nzp) != 0
     CMP  rs, rt         0010 ---- ssss tttt   NZP <- compare(rs, rt), unsigned
     ADD  rd, rs, rt     0011 dddd ssss tttt   rd <- rs + rt  (mod 256)
     SUB  rd, rs, rt     0100 dddd ssss tttt   rd <- rs - rt  (mod 256)
     MUL  rd, rs, rt     0101 dddd ssss tttt   rd <- rs * rt  (low 8 bits)
     DIV  rd, rs, rt     0110 dddd ssss tttt   rd <- rs / rt  (unsigned, x/0 = 255)
     LDR  rd, rs         0111 dddd ssss ----   rd <- mem[rs]
     STR  rs, rt         1000 ---- ssss tttt   mem[rs] <- rt
     CONST rd, imm       1001 dddd iiii iiii   rd <- imm
     RET                 1111 ---- ---- ----   the block is finished

   Registers R0..R12 are general purpose. R13 = %blockIdx, R14 = %blockDim and
   R15 = %threadIdx are read only; writes to them are dropped. *)

type reg = int

type t =
  | Nop
  | Br of { n : bool; z : bool; p : bool; target : int }
  | Cmp of reg * reg
  | Add of reg * reg * reg
  | Sub of reg * reg * reg
  | Mul of reg * reg * reg
  | Div of reg * reg * reg
  | Ldr of reg * reg
  | Str of reg * reg
  | Const of reg * int
  | Ret

let op_nop = 0x0
let op_br = 0x1
let op_cmp = 0x2
let op_add = 0x3
let op_sub = 0x4
let op_mul = 0x5
let op_div = 0x6
let op_ldr = 0x7
let op_str = 0x8
let op_const = 0x9
let op_ret = 0xf

let num_gp_regs = 13
let reg_block_idx = 13
let reg_block_dim = 14
let reg_thread_idx = 15
let threads_per_block = 4

let check_reg r = if r < 0 || r > 15 then invalid_arg (Printf.sprintf "register R%d" r)
let check_byte v = if v < 0 || v > 255 then invalid_arg (Printf.sprintf "immediate %d" v)

let encode i =
  let r3 op a b c =
    check_reg a;
    check_reg b;
    check_reg c;
    (op lsl 12) lor (a lsl 8) lor (b lsl 4) lor c
  in
  match i with
  | Nop -> 0
  | Br { n; z; p; target } ->
    check_byte target;
    let b x = if x then 1 else 0 in
    (op_br lsl 12) lor (b n lsl 11) lor (b z lsl 10) lor (b p lsl 9) lor target
  | Cmp (s, t) -> r3 op_cmp 0 s t
  | Add (d, s, t) -> r3 op_add d s t
  | Sub (d, s, t) -> r3 op_sub d s t
  | Mul (d, s, t) -> r3 op_mul d s t
  | Div (d, s, t) -> r3 op_div d s t
  | Ldr (d, s) -> r3 op_ldr d s 0
  | Str (s, t) -> r3 op_str 0 s t
  | Const (d, v) ->
    check_reg d;
    check_byte v;
    (op_const lsl 12) lor (d lsl 8) lor v
  | Ret -> op_ret lsl 12

let decode w =
  let op = (w lsr 12) land 0xf
  and d = (w lsr 8) land 0xf
  and s = (w lsr 4) land 0xf
  and t = w land 0xf
  and imm = w land 0xff in
  match op with
  | 0x1 ->
    Br
      { n = w land 0x800 <> 0; z = w land 0x400 <> 0; p = w land 0x200 <> 0; target = imm }
  | 0x2 -> Cmp (s, t)
  | 0x3 -> Add (d, s, t)
  | 0x4 -> Sub (d, s, t)
  | 0x5 -> Mul (d, s, t)
  | 0x6 -> Div (d, s, t)
  | 0x7 -> Ldr (d, s)
  | 0x8 -> Str (s, t)
  | 0x9 -> Const (d, imm)
  | 0xf -> Ret
  | _ -> Nop

let reg_name r =
  match r with
  | 13 -> "%blockIdx"
  | 14 -> "%blockDim"
  | 15 -> "%threadIdx"
  | r -> Printf.sprintf "R%d" r

let to_string i =
  let r = reg_name in
  match i with
  | Nop -> "NOP"
  | Br { n; z; p; target } ->
    Printf.sprintf "BR%s%s%s %d" (if n then "n" else "") (if z then "z" else "")
      (if p then "p" else "") target
  | Cmp (s, t) -> Printf.sprintf "CMP %s, %s" (r s) (r t)
  | Add (d, s, t) -> Printf.sprintf "ADD %s, %s, %s" (r d) (r s) (r t)
  | Sub (d, s, t) -> Printf.sprintf "SUB %s, %s, %s" (r d) (r s) (r t)
  | Mul (d, s, t) -> Printf.sprintf "MUL %s, %s, %s" (r d) (r s) (r t)
  | Div (d, s, t) -> Printf.sprintf "DIV %s, %s, %s" (r d) (r s) (r t)
  | Ldr (d, s) -> Printf.sprintf "LDR %s, %s" (r d) (r s)
  | Str (s, t) -> Printf.sprintf "STR %s, %s" (r s) (r t)
  | Const (d, v) -> Printf.sprintf "CONST %s, #%d" (r d) v
  | Ret -> "RET"

(* Arithmetic shared by the interpreter and the tests. The hardware computes
   the same functions with gates; these are the specification. *)
let div8 a b = if b = 0 then 255 else a / b

let nzp_of_compare a b = if a < b then 4 else if a = b then 2 else 1
