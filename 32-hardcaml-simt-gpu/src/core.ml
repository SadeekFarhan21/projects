(* The compute core: one PC shared by four threads, a multi cycle
   fetch / decode / execute / wait / writeback state machine, per thread
   register files and NZP flags, and one load store unit (LSU) per thread.

   The core also contains the block dispatcher: it walks block indices
   0, 1, ... until every launched thread has run, resetting the register files
   between blocks. *)
open Hardcaml
open Signal

let threads = Isa.threads_per_block

module State = struct
  type t =
    | Idle
    | Launch
    | Fetch
    | Decode
    | Execute
    | Wait
    | Writeback
    | Done
  [@@deriving sexp_of, compare, enumerate]
end

(* LSU states. *)
let lsu_idle = 0
let lsu_pending = 1
let lsu_inflight = 2
let lsu_done = 3

type t =
  { imem_addr : Signal.t
  ; lsu : Mem_ctrl.lsu_request array
  ; done_ : Signal.t
  ; busy : Signal.t
  ; state : Signal.t
  ; pc : Signal.t
  ; cycles : Signal.t
  ; stall_cycles : Signal.t
  ; instrs : Signal.t
  ; mem_reqs : Signal.t
  ; divergent : Signal.t
  ; queue_full_cycles : Signal.t
  }

let create ~spec ~start ~thread_count ~imem_rdata ~(grant : Signal.t array)
    ~(resp : Mem_ctrl.response) ~queue_full_stall =
  let open Always in
  let sm = State_machine.create (module State) spec ~enable:vdd in
  let reg8 name = Variable.reg spec ~width:8 |> fun v -> ignore (v.value -- name); v in
  let pc = reg8 "pc" in
  let block_idx = reg8 "block_idx" in
  let launch_count = reg8 "launch_count" in
  let instr = Variable.reg spec ~width:16 in
  ignore (instr.value -- "instr");
  (* Decoded fields, straight from the instruction register. *)
  let iv = instr.value in
  let op = select iv 15 12 -- "op"
  and rd = select iv 11 8
  and rs = select iv 7 4
  and rt = select iv 3 0
  and imm = select iv 7 0
  and br_mask = select iv 11 9 in
  let is c = op ==:. c in
  let is_mem = is Isa.op_ldr |: is Isa.op_str in
  let writes_alu =
    is Isa.op_add |: is Isa.op_sub |: is Isa.op_mul |: is Isa.op_div |: is Isa.op_const
  in
  (* Per thread state. *)
  let active = Array.init threads (fun t -> Variable.reg spec ~width:1 |> fun v -> ignore (v.value -- Printf.sprintf "active%d" t); v) in
  let regs =
    Array.init threads (fun t ->
      Array.init Isa.num_gp_regs (fun k ->
        let v = Variable.reg spec ~width:8 in
        ignore (v.value -- Printf.sprintf "t%d_r%d" t k);
        v))
  in
  let nzp = Array.init threads (fun t -> Variable.reg spec ~width:3 |> fun v -> ignore (v.value -- Printf.sprintf "t%d_nzp" t); v) in
  let rs_val = Array.init threads (fun t -> reg8 (Printf.sprintf "t%d_rs" t)) in
  let rt_val = Array.init threads (fun t -> reg8 (Printf.sprintf "t%d_rt" t)) in
  let result = Array.init threads (fun t -> reg8 (Printf.sprintf "t%d_result" t)) in
  let cmp_flags = Array.init threads (fun _ -> Variable.reg spec ~width:3) in
  let lsu_state = Array.init threads (fun t -> Variable.reg spec ~width:2 |> fun v -> ignore (v.value -- Printf.sprintf "t%d_lsu" t); v) in
  let lsu_rdata = Array.init threads (fun t -> reg8 (Printf.sprintf "t%d_lsu_rdata" t)) in
  (* Performance counters. *)
  let cnt w = Variable.reg spec ~width:w in
  let cycles = cnt 32 and stall_cycles = cnt 32 and instrs = cnt 32 and mem_reqs = cnt 32 in
  let divergent = cnt 16 and queue_full_cycles = cnt 32 in
  let counters = [ cycles; stall_cycles; instrs; mem_reqs; divergent; queue_full_cycles ] in
  (* Register read ports: R13..R15 are the special index registers. *)
  let read_reg t idx =
    mux idx
      (List.init 16 (fun k ->
         if k < Isa.num_gp_regs then regs.(t).(k).value
         else if k = Isa.reg_block_idx then block_idx.value
         else if k = Isa.reg_block_dim then of_int ~width:8 threads
         else of_int ~width:8 t))
  in
  (* Launch bookkeeping. A thread is active if 4 * block + t < launch count. *)
  let global_idx t = block_idx.value @: of_int ~width:2 t in
  let thread_active t = global_idx t <: uresize launch_count.value 10 in
  let last_block =
    (uresize block_idx.value 9 +:. 1) @: zero 2 >=: uresize launch_count.value 11
  in
  (* Branch resolution: thread 0 decides, disagreement is counted. *)
  let taken t = (nzp.(t).value &: br_mask) <>:. 0 in
  let diverged =
    List.fold_left ( |: ) gnd
      (List.init threads (fun t -> active.(t).value &: (taken t ^: taken 0)))
  in
  let all_lsu_done =
    List.fold_left ( &: ) vdd
      (List.init threads (fun t -> ~:(active.(t).value) |: (lsu_state.(t).value ==:. lsu_done)))
  in
  let num_active =
    List.fold_left ( +: ) (zero 32) (List.init threads (fun t -> uresize active.(t).value 32))
  in
  let start_kernel =
    [ block_idx <--. 0
    ; launch_count <-- thread_count
    ; proc (List.map (fun c -> c <--. 0) counters)
    ; if_ (thread_count ==:. 0) [ sm.set_next Done ] [ sm.set_next Launch ]
    ]
  in
  let per_thread f = proc (List.init threads f) in
  compile
    [ (* Cycle accounting: every cycle of a running kernel counts. *)
      when_ (~:(sm.is Idle) &: ~:(sm.is Done)) [ cycles <-- cycles.value +:. 1 ]
    ; when_ (sm.is Wait) [ stall_cycles <-- stall_cycles.value +:. 1 ]
    ; when_ queue_full_stall [ queue_full_cycles <-- queue_full_cycles.value +:. 1 ]
    ; (* LSU progress, driven by the memory controller in any state. *)
      per_thread (fun t ->
        proc
          [ when_ grant.(t) [ lsu_state.(t) <--. lsu_inflight ]
          ; when_ (resp.valid &: (resp.tid ==:. t))
              [ lsu_state.(t) <--. lsu_done; lsu_rdata.(t) <-- resp.rdata ]
          ])
    ; sm.switch
        [ (State.Idle, [ when_ start start_kernel ])
        ; (State.Done, [ when_ start start_kernel ])
        ; ( State.Launch
          , [ pc <--. 0
            ; per_thread (fun t ->
                proc
                  ([ active.(t) <-- thread_active t; nzp.(t) <--. 0b010; lsu_state.(t) <--. lsu_idle ]
                   @ List.init Isa.num_gp_regs (fun k -> regs.(t).(k) <--. 0)))
            ; sm.set_next Fetch
            ] )
        ; (State.Fetch, [ instr <-- imem_rdata; sm.set_next Decode ])
        ; ( State.Decode
          , [ per_thread (fun t ->
                proc [ rs_val.(t) <-- read_reg t rs; rt_val.(t) <-- read_reg t rt ])
            ; sm.set_next Execute
            ] )
        ; ( State.Execute
          , [ per_thread (fun t ->
                proc
                  [ result.(t)
                    <-- Alu.result ~op ~rs:rs_val.(t).value ~rt:rt_val.(t).value ~imm
                  ; cmp_flags.(t) <-- Alu.compare_nzp rs_val.(t).value rt_val.(t).value
                  ; when_ (is_mem &: active.(t).value) [ lsu_state.(t) <--. lsu_pending ]
                  ])
            ; if_ (is Isa.op_ret)
                [ instrs <-- instrs.value +:. 1
                ; if_ last_block [ sm.set_next Done ]
                    [ block_idx <-- block_idx.value +:. 1; sm.set_next Launch ]
                ]
                [ if_ is_mem
                    [ mem_reqs <-- mem_reqs.value +: num_active; sm.set_next Wait ]
                    [ sm.set_next Writeback ]
                ]
            ] )
        ; (State.Wait, [ when_ all_lsu_done [ sm.set_next Writeback ] ])
        ; ( State.Writeback
          , [ per_thread (fun t ->
                let value = mux2 (is Isa.op_ldr) lsu_rdata.(t).value result.(t).value in
                let we = active.(t).value &: (writes_alu |: is Isa.op_ldr) in
                proc
                  (List.init Isa.num_gp_regs (fun k ->
                     when_ (we &: (rd ==:. k)) [ regs.(t).(k) <-- value ])
                   @ [ when_ (active.(t).value &: is Isa.op_cmp)
                         [ nzp.(t) <-- cmp_flags.(t).value ]
                     ; lsu_state.(t) <--. lsu_idle
                     ]))
            ; if_ (is Isa.op_br &: taken 0) [ pc <-- imm ] [ pc <-- pc.value +:. 1 ]
            ; when_ (is Isa.op_br &: diverged) [ divergent <-- divergent.value +:. 1 ]
            ; instrs <-- instrs.value +:. 1
            ; sm.set_next Fetch
            ] )
        ]
    ];
  let lsu =
    Array.init threads (fun t ->
      { Mem_ctrl.pending = lsu_state.(t).value ==:. lsu_pending
      ; addr = rs_val.(t).value
      ; we = is Isa.op_str
      ; wdata = rt_val.(t).value
      })
  in
  { imem_addr = pc.value
  ; lsu
  ; done_ = sm.is Done
  ; busy = ~:(sm.is Idle) &: ~:(sm.is Done)
  ; state = sm.current
  ; pc = pc.value
  ; cycles = cycles.value
  ; stall_cycles = stall_cycles.value
  ; instrs = instrs.value
  ; mem_reqs = mem_reqs.value
  ; divergent = divergent.value
  ; queue_full_cycles = queue_full_cycles.value
  }
