(* Cycle accurate simulation driver around Hardcaml's Cyclesim. *)
open Hardcaml

module Sim = Cyclesim.With_interface (Gpu.I) (Gpu.O)

type stats =
  { cycles : int
  ; stall_cycles : int
  ; instrs : int
  ; mem_reqs : int
  ; divergent : int
  ; queue_full_cycles : int
  }

type run_result =
  { mem : int array
  ; stats : stats
  }

exception Timeout of int

type t =
  { sim : Sim.t
  ; i : Bits.t ref Gpu.I.t
  ; o : Bits.t ref Gpu.O.t
  ; close : unit -> unit
  }

let create ?vcd config =
  let sim = Sim.create ~config:Cyclesim.Config.trace_all (Gpu.create config) in
  let sim, close =
    match vcd with
    | None -> (sim, fun () -> ())
    | Some path ->
      let oc = Out_channel.open_text path in
      (Vcd.wrap oc sim, fun () -> Out_channel.close oc)
  in
  { sim; i = Cyclesim.inputs sim; o = Cyclesim.outputs sim; close }

let set r w v = r := Bits.of_int ~width:w v
let get r = Bits.to_int !r

let cycle t = Cyclesim.cycle t.sim

(* Load program and data, launch, wait for done, read everything back.
   [code] is written to all 256 instruction slots (unused slots are NOP). *)
let run ?(max_cycles = 200_000) t ~threads ~(code : int array) (mem0 : int array) =
  let i = t.i in
  i.clear := Bits.vdd;
  cycle t;
  i.clear := Bits.gnd;
  i.host_imem_we := Bits.vdd;
  for a = 0 to 255 do
    set i.host_imem_addr 8 a;
    set i.host_imem_data 16 (if a < Array.length code then code.(a) else 0);
    cycle t
  done;
  i.host_imem_we := Bits.gnd;
  i.host_dmem_we := Bits.vdd;
  for a = 0 to 255 do
    set i.host_dmem_addr 8 a;
    set i.host_dmem_wdata 8 mem0.(a);
    cycle t
  done;
  i.host_dmem_we := Bits.gnd;
  set i.thread_count 8 threads;
  i.start := Bits.vdd;
  cycle t;
  i.start := Bits.gnd;
  let n = ref 0 in
  while get t.o.done_ = 0 do
    if !n >= max_cycles then raise (Timeout !n);
    cycle t;
    incr n
  done;
  let mem =
    Array.init 256 (fun a ->
      set i.host_dmem_raddr 8 a;
      cycle t;
      get t.o.host_dmem_rdata)
  in
  let o = t.o in
  { mem
  ; stats =
      { cycles = get o.cycles
      ; stall_cycles = get o.stall_cycles
      ; instrs = get o.instrs
      ; mem_reqs = get o.mem_reqs
      ; divergent = get o.divergent
      ; queue_full_cycles = get o.queue_full_cycles
      }
  }

let stats_to_string s =
  Printf.sprintf
    "cycles=%d stall_cycles=%d (%.1f%%) instrs=%d mem_reqs=%d divergent=%d queue_full_cycles=%d"
    s.cycles s.stall_cycles
    (100. *. float s.stall_cycles /. float (max 1 s.cycles))
    s.instrs s.mem_reqs s.divergent s.queue_full_cycles

(* Run once without tracing; if [check] fails, rerun with a VCD dump so the
   failure can be inspected in a waveform viewer. Returns the result and the
   VCD path when one was written. *)
let run_checked ?max_cycles ?(vcd_dir = "results/failures") ~name ~check config ~threads ~code
    mem0 =
  let t = create config in
  let r = run ?max_cycles t ~threads ~code mem0 in
  if check r then (r, None)
  else begin
    (try Sys.mkdir "results" 0o755 with Sys_error _ -> ());
    (try Sys.mkdir vcd_dir 0o755 with Sys_error _ -> ());
    let path = Filename.concat vcd_dir (name ^ ".vcd") in
    let t = create ~vcd:path config in
    let r = run ?max_cycles t ~threads ~code mem0 in
    t.close ();
    (r, Some path)
  end
