(* Per thread arithmetic. Pure combinational logic. *)
open Hardcaml
open Signal

(* Unsigned restoring division, one subtract stage per quotient bit. A zero
   divisor makes every trial subtraction succeed, so x / 0 = all ones, which
   matches Isa.div8. *)
let divide a b =
  let w = width a in
  let b' = uresize b (w + 1) in
  let rec go i rem qbits =
    if i < 0 then concat_msb (List.rev qbits)
    else
      let shifted = lsbs rem @: bit a i in
      let ge = shifted >=: b' in
      let rem = mux2 ge (shifted -: b') shifted in
      go (i - 1) rem (ge :: qbits)
  in
  go (w - 1) (zero (w + 1)) []

(* NZP flags of an unsigned compare, packed as [N; Z; P] (N is the msb). *)
let compare_nzp a b = concat_msb [ a <: b; a ==: b; a >: b ]

(* Result written to rd for the register writing ALU opcodes. *)
let result ~op ~rs ~rt ~imm =
  let w = width rs in
  let prod = sel_bottom (rs *: rt) w in
  mux op
    (List.init 16 (fun code ->
       if code = Isa.op_add then rs +: rt
       else if code = Isa.op_sub then rs -: rt
       else if code = Isa.op_mul then prod
       else if code = Isa.op_div then divide rs rt
       else if code = Isa.op_const then imm
       else zero w))
