(* Top level: core + instruction memory + memory controller + DRAM.

   The host loads the instruction and data memories through the host ports
   while the GPU is idle, sets thread_count, pulses start, waits for done and
   reads results back through host_dmem_raddr / host_dmem_rdata. *)
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; start : 'a
    ; thread_count : 'a [@bits 8]
    ; host_imem_we : 'a
    ; host_imem_addr : 'a [@bits 8]
    ; host_imem_data : 'a [@bits 16]
    ; host_dmem_we : 'a
    ; host_dmem_addr : 'a [@bits 8]
    ; host_dmem_wdata : 'a [@bits 8]
    ; host_dmem_raddr : 'a [@bits 8]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { done_ : 'a
    ; busy : 'a
    ; host_dmem_rdata : 'a [@bits 8]
    ; pc : 'a [@bits 8]
    ; state : 'a [@bits 3]
    ; cycles : 'a [@bits 32]
    ; stall_cycles : 'a [@bits 32]
    ; instrs : 'a [@bits 32]
    ; mem_reqs : 'a [@bits 32]
    ; divergent : 'a [@bits 16]
    ; queue_full_cycles : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

let create (config : Config.t) (i : _ I.t) : _ O.t =
  Config.validate config;
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let threads = Isa.threads_per_block in
  (* Wires break the combinational knot between core and memory system. *)
  let imem_rdata = wire 16 in
  let grant = Array.init threads (fun _ -> wire 1) in
  let resp = { Mem_ctrl.valid = wire 1; tid = wire 2; rdata = wire 8 } in
  let queue_full_stall = wire 1 in
  let core =
    Core.create ~spec ~start:i.start ~thread_count:i.thread_count ~imem_rdata ~grant ~resp
      ~queue_full_stall
  in
  let imem =
    multiport_memory 256
      ~write_ports:
        [| { Write_port.write_clock = i.clock
           ; write_address = i.host_imem_addr
           ; write_enable = i.host_imem_we
           ; write_data = i.host_imem_data
           }
        |]
      ~read_addresses:[| core.imem_addr |]
  in
  imem_rdata <== imem.(0);
  let mem =
    Mem_ctrl.create ~config ~spec ~clock:i.clock ~reqs:core.lsu ~host_we:i.host_dmem_we
      ~host_addr:i.host_dmem_addr ~host_wdata:i.host_dmem_wdata ~host_raddr:i.host_dmem_raddr
  in
  Array.iteri (fun t g -> g <== mem.grant.(t)) grant;
  resp.valid <== mem.resp.valid;
  resp.tid <== mem.resp.tid;
  resp.rdata <== mem.resp.rdata;
  queue_full_stall <== mem.queue_full_stall;
  { O.done_ = core.done_
  ; busy = core.busy
  ; host_dmem_rdata = mem.host_rdata
  ; pc = core.pc
  ; state = core.state
  ; cycles = core.cycles
  ; stall_cycles = core.stall_cycles
  ; instrs = core.instrs
  ; mem_reqs = core.mem_reqs
  ; divergent = core.divergent
  ; queue_full_cycles = core.queue_full_cycles
  }

let circuit ?(name = "gpu") config =
  let module C = Circuit.With_interface (I) (O) in
  C.create_exn ~name (create config)

let verilog config =
  let buf = Buffer.create 65536 in
  Rtl.output ~output_mode:(Rtl.Output_mode.To_buffer buf) Rtl.Language.Verilog (circuit config);
  Buffer.contents buf
