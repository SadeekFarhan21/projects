(* Differential test: random kernels on the Hardcaml core against the ISA
   interpreter. Compares the full data memory and the retired instruction,
   memory request and divergent branch counters. The number of kernels per
   configuration can be raised with FUZZ_N (default 150). *)
open Gpu_lib

let configs =
  [ Config.default
  ; { Config.dram_latency = 1; dram_interval = 1; queue_depth = 2 }
  ; { Config.dram_latency = 5; dram_interval = 3; queue_depth = 2 }
  ]

let () =
  let n = try int_of_string (Sys.getenv "FUZZ_N") with Not_found -> 150 in
  let total = ref 0 and mismatches = ref 0 in
  List.iteri
    (fun ci config ->
      let st = Random.State.make [| 1000 + ci |] in
      for k = 1 to n do
        let code, threads, mem = Randprog.kernel st in
        let spec = Interp.run ~threads ~code mem in
        let ok (r : Harness.run_result) =
          r.mem = spec.mem && r.stats.instrs = spec.instrs
          && r.stats.mem_reqs = spec.mem_ops && r.stats.divergent = spec.divergent
        in
        let name = Printf.sprintf "random_cfg%d_%d" ci k in
        let r, vcd = Harness.run_checked ~name ~check:ok config ~threads ~code mem in
        incr total;
        if not (ok r) then begin
          incr mismatches;
          Printf.printf "MISMATCH %s (%s), threads=%d, vcd=%s\n%s\n%!" name
            (Config.to_string config) threads
            (Option.value vcd ~default:"none")
            (Asm.disassemble code)
        end
      done)
    configs;
  Printf.printf "test_random: %d kernels, %d mismatches\n" !total !mismatches;
  if !mismatches > 0 then exit 1
