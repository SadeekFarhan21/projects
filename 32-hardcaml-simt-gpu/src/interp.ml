(* ISA level reference interpreter. This is the specification the hardware is
   checked against. It models the same lockstep semantics as the core:

   - Threads are launched in blocks of 4. Block b owns threads 4b..4b+3, and a
     thread is active only if its global index is below the launch count.
   - All active threads of a block execute the same instruction together.
   - Branch direction is decided by thread 0 of the block (always active).
     v0 has no divergence support, so a branch where active threads disagree is
     counted as divergent and every thread follows thread 0.
   - Within one STR, stores are applied in thread index order, so the highest
     active thread wins on an address collision.
   - Registers are zeroed and NZP is set to Z at the start of every block. *)

type result =
  { mem : int array
  ; instrs : int (* block level instructions retired, RET included *)
  ; divergent : int
  ; mem_ops : int (* per thread memory requests *)
  }

exception Step_limit

let run ?(max_steps = 1_000_000) ~threads ~(code : int array) (mem0 : int array) : result =
  let mem = Array.copy mem0 in
  let nblocks = (threads + 3) / 4 in
  let instrs = ref 0 and divergent = ref 0 and mem_ops = ref 0 in
  let fetch pc = if pc < Array.length code then Isa.decode code.(pc) else Isa.Nop in
  for b = 0 to nblocks - 1 do
    let regs = Array.init 4 (fun _ -> Array.make 16 0) in
    let nzp = Array.make 4 2 (* Z, so BRnzp is taken before any CMP *) in
    let active = Array.init 4 (fun t -> (b * 4) + t < threads) in
    let rd t r =
      match r with
      | 13 -> b land 0xff
      | 14 -> 4
      | 15 -> t
      | r -> regs.(t).(r)
    in
    let wr t r v = if r < Isa.num_gp_regs then regs.(t).(r) <- v land 0xff in
    let each f =
      for t = 0 to 3 do
        if active.(t) then f t
      done
    in
    let pc = ref 0 and running = ref true in
    while !running do
      if !instrs >= max_steps then raise Step_limit;
      incr instrs;
      let next = ref ((!pc + 1) land 0xff) in
      (match fetch !pc with
      | Isa.Nop -> ()
      | Isa.Ret -> running := false
      | Isa.Const (d, v) -> each (fun t -> wr t d v)
      | Isa.Add (d, s, u) -> each (fun t -> wr t d (rd t s + rd t u))
      | Isa.Sub (d, s, u) -> each (fun t -> wr t d (rd t s - rd t u))
      | Isa.Mul (d, s, u) -> each (fun t -> wr t d (rd t s * rd t u))
      | Isa.Div (d, s, u) -> each (fun t -> wr t d (Isa.div8 (rd t s) (rd t u)))
      | Isa.Cmp (s, u) -> each (fun t -> nzp.(t) <- Isa.nzp_of_compare (rd t s) (rd t u))
      | Isa.Ldr (d, s) ->
        (* all addresses are read before any register is written *)
        let v = Array.init 4 (fun t -> if active.(t) then mem.(rd t s) else 0) in
        each (fun t ->
            incr mem_ops;
            wr t d v.(t))
      | Isa.Str (s, u) ->
        each (fun t ->
            incr mem_ops;
            mem.(rd t s) <- rd t u)
      | Isa.Br { n; z; p; target } ->
        let mask = (if n then 4 else 0) lor (if z then 2 else 0) lor if p then 1 else 0 in
        let taken t = nzp.(t) land mask <> 0 in
        let t0 = taken 0 in
        let div = ref false in
        each (fun t -> if taken t <> t0 then div := true);
        if !div then incr divergent;
        if t0 then next := target);
      pc := !next
    done
  done;
  { mem; instrs = !instrs; divergent = !divergent; mem_ops = !mem_ops }
