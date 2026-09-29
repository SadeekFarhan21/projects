(* Hardware tests on the hand written kernels, across several memory system
   configurations. Each run is checked against the CPU reference and against
   the ISA interpreter's retired instruction and memory request counts. *)
open Gpu_lib

let checks = ref 0
let failures = ref 0

let check name cond =
  incr checks;
  if not cond then begin
    incr failures;
    Printf.printf "FAIL %s\n%!" name
  end

let configs =
  [ Config.default
  ; { Config.dram_latency = 1; dram_interval = 1; queue_depth = 2 }
  ; { Config.dram_latency = 3; dram_interval = 4; queue_depth = 2 }
  ; { Config.dram_latency = 16; dram_interval = 1; queue_depth = 8 }
  ]

let test_kernel config (k : Kernels.kernel) =
  let p = Asm.assemble k.source in
  let spec = Interp.run ~threads:k.threads ~code:p.code k.mem in
  let name = Printf.sprintf "%s_lat%d_int%d_q%d" k.name config.Config.dram_latency
      config.dram_interval config.queue_depth in
  let r, vcd =
    Harness.run_checked ~name
      ~check:(fun r -> k.expected r.mem && r.mem = spec.mem)
      config ~threads:k.threads ~code:p.code k.mem
  in
  (match vcd with Some path -> Printf.printf "waveform written to %s\n" path | None -> ());
  check (name ^ " matches CPU reference") (k.expected r.mem);
  check (name ^ " full memory matches interpreter") (r.mem = spec.mem);
  check (name ^ " instruction count") (r.stats.instrs = spec.instrs);
  check (name ^ " memory request count") (r.stats.mem_reqs = spec.mem_ops);
  check (name ^ " stall cycles below total") (r.stats.stall_cycles < r.stats.cycles)

let test_zero_threads () =
  let t = Harness.create Config.default in
  let mem = Array.init 256 (fun a -> a) in
  let code = (Asm.assemble (Kernels.vecadd_source 8)).code in
  let r = Harness.run t ~threads:0 ~code mem in
  check "zero threads leaves memory untouched" (r.mem = mem);
  check "zero threads takes zero kernel cycles" (r.stats.cycles = 0)

(* Launching twice on the same simulator (restart from the Done state) must
   give identical results and counters. *)
let test_relaunch () =
  let k = Kernels.matmul 4 in
  let code = (Asm.assemble k.source).code in
  let t = Harness.create Config.default in
  let r1 = Harness.run t ~threads:k.threads ~code k.mem in
  let r2 = Harness.run t ~threads:k.threads ~code k.mem in
  check "relaunch gives same memory" (r1.mem = r2.mem);
  check "relaunch gives same cycle count" (r1.stats = r2.stats)

(* Read only index registers: writes to R13..R15 are dropped. *)
let test_special_registers () =
  let p =
    Asm.assemble
      {|CONST R13, #99
        CONST R14, #99
        CONST R15, #99
        MUL R0, %blockIdx, %blockDim
        ADD R0, R0, %threadIdx
        CONST R1, #100
        ADD R1, R1, R0
        STR R1, %blockDim
        CONST R2, #200
        ADD R2, R2, R0
        STR R2, %threadIdx
        RET|}
  in
  let r = Harness.run (Harness.create Config.default) ~threads:10 ~code:p.code (Array.make 256 0) in
  check "blockDim reads 4" (Array.for_all (fun v -> v = 4) (Array.sub r.mem 100 10));
  check "threadIdx reads t" (Array.sub r.mem 200 10 = Array.init 10 (fun g -> g mod 4));
  check "inactive threads do not store" (r.mem.(110) = 0 && r.mem.(210) = 0)

let () =
  List.iter
    (fun c ->
      List.iter (test_kernel c)
        (Kernels.bench_set () @ [ Kernels.vecadd 7; Kernels.vecadd 13; Kernels.matmul 3 ]))
    configs;
  test_zero_threads ();
  test_relaunch ();
  test_special_registers ();
  Printf.printf "test_kernels: %d checks, %d failures\n" !checks !failures;
  if !failures > 0 then exit 1
