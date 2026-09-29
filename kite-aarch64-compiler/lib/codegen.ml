(* AArch64 code generation for macOS (Mach-O, Apple arm64 ABI).

   Conventions (see DESIGN.md for the frame diagram):
   - Symbols: Kite function f -> _kf_f, lambda n -> _kl_n, static
     closure for f -> _kc_f, string literal -> _ks_n, C runtime
     function g -> _g.
   - Arguments in x0..x7 (at most 8), result in x0.
   - Closure calls pass the closure pointer in x9 (a caller-saved
     temporary register that no argument uses), so top-level functions
     and lambdas share the same argument registers.
   - Frame: stp x29, x30 then sub sp; spill slots at [sp + 8k];
     callee-saved registers used by the allocator are saved above the
     slots. sp stays 16-byte aligned at every call.
   - Scratch: x10-x12 for operands, x16 for the indirect call target,
     x17 for out-of-range stack addresses. x18 is never touched
     (reserved by Apple).

   Each temp lives either in a stack slot or in a callee-saved register
   (x19-x28) chosen by the register allocator. With no allocator, every
   temp gets a slot: the classic "stack machine in registers" v0. *)

open Ir

type location = Reg of string | Slot of int  (* slot index *)

type alloc = {
  loc_of : temp -> location;
  nslots : int;
  saved : string list;        (* callee-saved registers to preserve *)
  live_out : (label, temp list) Hashtbl.t option;  (* for compare/branch fusion *)
}

let all_slots (f : func) =
  { loc_of = (fun t -> Slot t); nslots = f.ntemps; saved = []; live_out = None }

let buf = Buffer.create 65536
let out fmt = Printf.ksprintf (fun s -> Buffer.add_string buf "\t"; Buffer.add_string buf s; Buffer.add_char buf '\n') fmt
let label s = Buffer.add_string buf (s ^ ":\n")

(* Load a 64-bit constant into register [r]. *)
let mov_imm r (c : int64) =
  if Int64.compare c 0L >= 0 && Int64.compare c 65535L <= 0 then out "mov %s, #%Ld" r c
  else if Int64.compare c (-65536L) >= 0 && Int64.compare c 0L < 0 then out "mov %s, #%Ld" r c
  else begin
    let first = ref true in
    for k = 0 to 3 do
      let chunk = Int64.logand (Int64.shift_right_logical c (16 * k)) 0xFFFFL in
      if chunk <> 0L then begin
        if !first then (out "movz %s, #%Ld, lsl #%d" r chunk (16 * k); first := false)
        else out "movk %s, #%Ld, lsl #%d" r chunk (16 * k)
      end
    done;
    if !first then out "mov %s, #0" r
  end

(* Address operand for stack slot [k]; may clobber x17. *)
let slot_addr k =
  let off = 8 * k in
  if off <= 32760 then Printf.sprintf "[sp, #%d]" off
  else begin
    mov_imm "x17" (Int64.of_int off);
    out "add x17, sp, x17";
    "[x17]"
  end

type ctx = { a : alloc; fname : string; epilogue : string }

(* Return a register holding operand [o], using [scratch] if needed. *)
let src ?(zr = true) ctx o scratch =
  match o with
  | C 0L when zr -> "xzr"
  | C c -> mov_imm scratch c; scratch
  | T t -> (
    match ctx.a.loc_of t with
    | Reg r -> r
    | Slot k -> out "ldr %s, %s" scratch (slot_addr k); scratch)

(* Move operand into a specific register. *)
let src_into ctx o r =
  let s = src ctx o r in
  if s <> r then out "mov %s, %s" r s

(* Register to compute temp [t] into; call [commit] afterwards. *)
let dst ctx t scratch = match ctx.a.loc_of t with Reg r -> r | Slot _ -> scratch

let commit ctx t r =
  match ctx.a.loc_of t with
  | Reg r' -> if r' <> r then out "mov %s, %s" r' r
  | Slot k -> out "str %s, %s" r (slot_addr k)

let cond_of = function
  | Lt -> "lt" | Le -> "le" | Gt -> "gt" | Ge -> "ge" | Eq -> "eq" | Ne -> "ne" | Ltu -> "lo"
  | _ -> assert false

let invert = function
  | "lt" -> "ge" | "ge" -> "lt" | "le" -> "gt" | "gt" -> "le" | "eq" -> "ne" | "ne" -> "eq"
  | "lo" -> "hs" | "hs" -> "lo" | c -> failwith ("invert " ^ c)

let is_cmp = function Lt | Le | Gt | Ge | Eq | Ne | Ltu -> true | _ -> false

let is_pow2 c = Int64.compare c 0L > 0 && Int64.logand c (Int64.sub c 1L) = 0L
let log2 c = let rec go k = if Int64.shift_left 1L k = c then k else go (k + 1) in go 0

let emit_cmp ctx a b =
  let ra = src ~zr:false ctx a "x10" in
  match b with
  | C c when Int64.compare c 0L >= 0 && Int64.compare c 4095L <= 0 -> out "cmp %s, #%Ld" ra c
  | _ ->
    let rb = src ctx b "x11" in
    out "cmp %s, %s" ra rb

let emit_bin ctx op d a b =
  let rd = dst ctx d "x12" in
  (match op, b with
   | (Add | Sub), C c when Int64.compare c 0L >= 0 && Int64.compare c 4095L <= 0 ->
     let ra = src ~zr:false ctx a "x10" in
     out "%s %s, %s, #%Ld" (if op = Add then "add" else "sub") rd ra c
   | (Shl | Shr), C c ->
     let ra = src ctx a "x10" in
     out "%s %s, %s, #%Ld" (if op = Shl then "lsl" else "asr") rd ra (Int64.logand c 63L)
   | Mul, C c when is_pow2 c ->
     let ra = src ctx a "x10" in
     out "lsl %s, %s, #%d" rd ra (log2 c)
   | _ when is_cmp op ->
     emit_cmp ctx a b;
     out "cset %s, %s" rd (cond_of op)
   | _ ->
     let ra = src ctx a "x10" in
     let rb = src ctx b "x11" in
     (match op with
      | Add -> out "add %s, %s, %s" rd ra rb
      | Sub -> out "sub %s, %s, %s" rd ra rb
      | Mul -> out "mul %s, %s, %s" rd ra rb
      | Div -> out "sdiv %s, %s, %s" rd ra rb
      | Rem ->
        out "sdiv x16, %s, %s" ra rb;
        out "msub %s, x16, %s, %s" rd rb ra
      | And -> out "and %s, %s, %s" rd ra rb
      | Or -> out "orr %s, %s, %s" rd ra rb
      | Xor -> out "eor %s, %s, %s" rd ra rb
      | Shl -> out "lsl %s, %s, %s" rd ra rb
      | Shr -> out "asr %s, %s, %s" rd ra rb
      | _ -> assert false));
  commit ctx d rd

let arg_regs = [| "x0"; "x1"; "x2"; "x3"; "x4"; "x5"; "x6"; "x7" |]

let emit_instr ctx = function
  | Mov (d, a) ->
    let rd = dst ctx d "x10" in
    src_into ctx a rd;
    commit ctx d rd
  | Bin (op, d, a, b) -> emit_bin ctx op d a b
  | Un (Neg, d, a) ->
    let rd = dst ctx d "x12" in
    let ra = src ctx a "x10" in
    out "neg %s, %s" rd ra;
    commit ctx d rd
  | Un (Not, d, a) ->
    let rd = dst ctx d "x12" in
    let ra = src ctx a "x10" in
    out "eor %s, %s, #1" rd ra;
    commit ctx d rd
  | Load (d, base, off) ->
    let rd = dst ctx d "x12" in
    let rb = src ~zr:false ctx base "x10" in
    out "ldr %s, [%s, #%d]" rd rb off;
    commit ctx d rd
  | Store (base, off, v) ->
    let rb = src ~zr:false ctx base "x10" in
    let rv = src ctx v "x11" in
    out "str %s, [%s, #%d]" rv rb off
  | Addr (d, sym) ->
    let rd = dst ctx d "x12" in
    out "adrp %s, %s@PAGE" rd sym;
    out "add %s, %s, %s@PAGEOFF" rd rd sym;
    commit ctx d rd
  | Call (d, callee, args) ->
    (* Temps live in slots or callee-saved registers, never in x0-x7, so
       filling argument registers in order cannot clobber a source. *)
    (match callee with Indirect f -> src_into ctx f "x9" | _ -> ());
    List.iteri (fun i a -> src_into ctx a arg_regs.(i)) args;
    (match callee with
     | Direct s -> out "bl %s" s
     | Runtime s -> out "bl _%s" s
     | Indirect _ ->
       out "ldr x16, [x9]";
       out "blr x16");
    Option.iter (fun d -> commit ctx d "x0") d

let block_label ctx l = Printf.sprintf "L%s_%d" ctx.fname l

let emit_func (f : func) (a : alloc) =
  let fname = String.sub f.name 1 (String.length f.name - 1) in
  let ctx = { a; fname; epilogue = Printf.sprintf "L%s_ret" fname } in
  let nsaved = List.length a.saved in
  let frame = ((8 * (a.nslots + nsaved)) + 15) / 16 * 16 in
  Buffer.add_string buf "\n\t.p2align 2\n";
  out ".globl %s" f.name;
  label f.name;
  out "stp x29, x30, [sp, #-16]!";
  out "mov x29, sp";
  let sub_sp n =
    if n > 0 then begin
      if n >= 4096 then out "sub sp, sp, #%d, lsl #12" (n lsr 12);
      if n land 4095 > 0 then out "sub sp, sp, #%d" (n land 4095)
    end
  in
  sub_sp frame;
  List.iteri (fun j r -> out "str %s, %s" r (slot_addr (a.nslots + j))) a.saved;
  Option.iter (fun e -> commit ctx e "x9") f.env;
  List.iteri (fun i p -> commit ctx p arg_regs.(i)) f.params;
  let blocks = Array.of_list f.blocks in
  let nb = Array.length blocks in
  Array.iteri
    (fun bi blk ->
      let next = if bi + 1 < nb then Some blocks.(bi + 1).label else None in
      if bi > 0 || true then label (block_label ctx blk.label);
      (* Compare/branch fusion: if the block ends with t = cmp a, b and
         the terminator branches on t, branch on the flags directly. *)
      let instrs, fused =
        match List.rev blk.instrs, blk.term with
        | Bin (op, t, x, y) :: rest, Br (T t', _, _) when t = t' && is_cmp op ->
          let dead =
            match a.live_out with
            | Some lo -> not (List.mem t (Hashtbl.find lo blk.label))
            | None -> false
          in
          (List.rev rest, Some (op, t, x, y, dead))
        | _ -> (blk.instrs, None)
      in
      List.iter (emit_instr ctx) instrs;
      let jump l = if Some l <> next then out "b %s" (block_label ctx l) in
      match blk.term, fused with
      | Br (_, l1, l2), Some (op, t, x, y, dead) ->
        emit_cmp ctx x y;
        if not dead then begin
          let rd = dst ctx t "x12" in
          out "cset %s, %s" rd (cond_of op);
          commit ctx t rd
        end;
        let c = cond_of op in
        if Some l1 = next then out "b.%s %s" (invert c) (block_label ctx l2)
        else (out "b.%s %s" c (block_label ctx l1); jump l2)
      | Jmp l, _ -> jump l
      | Br (o, l1, l2), None -> (
        match o with
        | C c -> jump (if c <> 0L then l1 else l2)
        | _ ->
          let r = src ctx o "x10" in
          if Some l1 = next then out "cbz %s, %s" r (block_label ctx l2)
          else (out "cbnz %s, %s" r (block_label ctx l1); jump l2))
      | Ret o, _ ->
        Option.iter (fun o -> src_into ctx o "x0") o;
        if bi + 1 < nb then out "b %s" ctx.epilogue
      | Unreachable, _ -> out "brk #1")
    blocks;
  label ctx.epilogue;
  List.iteri (fun j r -> out "ldr %s, %s" r (slot_addr (a.nslots + j))) a.saved;
  out "mov sp, x29";
  out "ldp x29, x30, [sp], #16";
  out "ret"

let emit_program (p : program) (alloc_for : func -> alloc) : string =
  Buffer.clear buf;
  Buffer.add_string buf "// generated by kitec\n\t.section __TEXT,__text,regular,pure_instructions\n";
  List.iter (fun f -> emit_func f (alloc_for f)) p.funcs;
  if p.strings <> [] then begin
    Buffer.add_string buf "\n\t.section __TEXT,__const\n";
    List.iter
      (fun (sym, s) ->
        Buffer.add_string buf "\t.p2align 3\n";
        label sym;
        out ".quad %d" (String.length s);
        let bytes = List.init (String.length s) (fun i -> string_of_int (Char.code s.[i])) in
        let rec chunks l =
          match l with
          | [] -> ()
          | _ ->
            let hd = List.filteri (fun i _ -> i < 16) l in
            let tl = List.filteri (fun i _ -> i >= 16) l in
            out ".byte %s" (String.concat ", " hd);
            chunks tl
        in
        chunks bytes;
        out ".byte 0")
      p.strings
  end;
  if p.closures <> [] then begin
    Buffer.add_string buf "\n\t.section __DATA,__data\n";
    List.iter
      (fun f ->
        Buffer.add_string buf "\t.p2align 3\n";
        label (Lower.closure_sym f);
        out ".quad %s" (Lower.fn_sym f))
      p.closures
  end;
  Buffer.add_string buf "\n.subsections_via_symbols\n";
  Buffer.contents buf
