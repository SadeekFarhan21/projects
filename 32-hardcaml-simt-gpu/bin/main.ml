(* Command line front end.

     gpu asm FILE                       assemble, print hex
     gpu disasm FILE                    assemble, print a listing
     gpu run FILE [config flags] [--vcd PATH]
     gpu emit-verilog OUT.v [config flags]
     gpu fuzz --n N --seed S --out results/fuzz.json
     gpu bench --out results/bench.csv
     gpu gen-vectors DIR --n N --seed S [config flags]
     gpu compare-vectors DIR --out results/verilator_agreement.json

   config flags: --latency L --interval I --queue Q *)
open Gpu_lib

let read_file path = In_channel.with_open_text path In_channel.input_all

let write_file path s =
  (match Filename.dirname path with
   | "." -> ()
   | d -> ignore (Sys.command (Printf.sprintf "mkdir -p %s" (Filename.quote d))));
  Out_channel.with_open_text path (fun oc -> output_string oc s)

(* tiny flag parser: --key value pairs plus positional arguments *)
let parse args =
  let rec go pos flags = function
    | k :: v :: rest when String.length k > 2 && String.sub k 0 2 = "--" ->
      go pos ((String.sub k 2 (String.length k - 2), v) :: flags) rest
    | x :: rest -> go (x :: pos) flags rest
    | [] -> (List.rev pos, flags)
  in
  go [] [] args

let flag flags k default =
  match List.assoc_opt k flags with Some v -> int_of_string v | None -> default

let config_of flags =
  let d = Config.default in
  { Config.dram_latency = flag flags "latency" d.dram_latency
  ; dram_interval = flag flags "interval" d.dram_interval
  ; queue_depth = flag flags "queue" d.queue_depth
  }

let load_program path =
  let p = Asm.assemble (read_file path) in
  let mem = Array.make 256 0 in
  List.iter (fun (a, v) -> mem.(a) <- v) p.data;
  (p, mem)

let cmd_run pos flags =
  let path = List.hd pos in
  let p, mem = load_program path in
  let config = config_of flags in
  let threads = flag flags "threads" (Option.value p.threads ~default:4) in
  let vcd = List.assoc_opt "vcd" flags in
  let t = Harness.create ?vcd config in
  let r = Harness.run t ~threads ~code:p.code mem in
  t.close ();
  let spec = Interp.run ~threads ~code:p.code mem in
  Printf.printf "%s\nconfig: %s, threads: %d\n%s\ninterpreter agrees: %b\n" path
    (Config.to_string config) threads (Harness.stats_to_string r.stats) (spec.mem = r.mem);
  print_endline "data memory (non zero rows):";
  for row = 0 to 15 do
    let vals = Array.sub r.mem (row * 16) 16 in
    if Array.exists (fun v -> v <> 0) vals then
      Printf.printf "  %3d: %s\n" (row * 16)
        (String.concat " " (Array.to_list (Array.map (Printf.sprintf "%3d") vals)))
  done

(* Randomized differential run over several memory configurations. *)
let cmd_fuzz flags =
  let n = flag flags "n" 1000 and seed = flag flags "seed" 42 in
  let out = Option.value (List.assoc_opt "out" flags) ~default:"results/fuzz.json" in
  let configs =
    [ Config.default
    ; { Config.dram_latency = 1; dram_interval = 1; queue_depth = 2 }
    ; { Config.dram_latency = 5; dram_interval = 3; queue_depth = 2 }
    ; { Config.dram_latency = 32; dram_interval = 1; queue_depth = 16 }
    ]
  in
  let t0 = Unix.gettimeofday () in
  let rows =
    List.mapi
      (fun ci config ->
        let st = Random.State.make [| seed; ci |] in
        let mism = ref 0 and cyc = ref 0 and stall = ref 0 and instrs = ref 0 in
        let div_kernels = ref 0 and div_branches = ref 0 and qfull = ref 0 in
        let sim = Harness.create config in
        for k = 1 to n do
          let code, threads, mem = Randprog.kernel st in
          let spec = Interp.run ~threads ~code mem in
          let r = Harness.run sim ~threads ~code mem in
          let ok =
            r.mem = spec.mem && r.stats.instrs = spec.instrs
            && r.stats.mem_reqs = spec.mem_ops && r.stats.divergent = spec.divergent
          in
          if not ok then begin
            incr mism;
            let name = Printf.sprintf "fuzz_cfg%d_%d" ci k in
            let _, vcd =
              Harness.run_checked ~name ~check:(fun _ -> false) config ~threads ~code mem
            in
            Printf.printf "MISMATCH %s vcd=%s\n%!" name (Option.value vcd ~default:"none")
          end;
          cyc := !cyc + r.stats.cycles;
          stall := !stall + r.stats.stall_cycles;
          instrs := !instrs + r.stats.instrs;
          qfull := !qfull + r.stats.queue_full_cycles;
          if spec.divergent > 0 then incr div_kernels;
          div_branches := !div_branches + spec.divergent
        done;
        Printf.printf "%s: %d kernels, %d mismatches\n%!" (Config.to_string config) n !mism;
        Printf.sprintf
          {|    {"latency": %d, "interval": %d, "queue": %d, "kernels": %d, "mismatches": %d, "total_cycles": %d, "stall_cycles": %d, "instrs": %d, "queue_full_cycles": %d, "kernels_with_divergent_branch": %d, "divergent_branches": %d}|}
          config.dram_latency config.dram_interval config.queue_depth n !mism !cyc !stall
          !instrs !qfull !div_kernels !div_branches)
      configs
  in
  let secs = Unix.gettimeofday () -. t0 in
  write_file out
    (Printf.sprintf "{\n  \"seed\": %d,\n  \"wall_seconds\": %.1f,\n  \"configs\": [\n%s\n  ]\n}\n"
       seed secs (String.concat ",\n" rows));
  Printf.printf "wrote %s (%.1f s)\n" out secs

(* Cycle counts per kernel over a sweep of memory system parameters. *)
let cmd_bench flags =
  let out = Option.value (List.assoc_opt "out" flags) ~default:"results/bench.csv" in
  let buf = Buffer.create 4096 in
  Buffer.add_string buf
    "sweep,kernel,threads,latency,interval,queue,cycles,stall_cycles,stall_frac,instrs,mem_reqs,queue_full_cycles,cycles_per_instr,correct\n";
  let run sweep config (k : Kernels.kernel) =
    let p = Asm.assemble k.source in
    let r = Harness.run (Harness.create config) ~threads:k.threads ~code:p.code k.mem in
    let s = r.stats in
    Buffer.add_string buf
      (Printf.sprintf "%s,%s,%d,%d,%d,%d,%d,%d,%.4f,%d,%d,%d,%.3f,%b\n" sweep k.name k.threads
         config.Config.dram_latency config.dram_interval config.queue_depth s.cycles
         s.stall_cycles
         (float s.stall_cycles /. float s.cycles)
         s.instrs s.mem_reqs s.queue_full_cycles
         (float s.cycles /. float s.instrs)
         (k.expected r.mem));
    Printf.printf "%-8s %-12s %s  %s\n%!" sweep k.name (Config.to_string config)
      (Harness.stats_to_string s)
  in
  let kernels = Kernels.bench_set () in
  List.iter
    (fun lat ->
      List.iter (run "latency" { Config.dram_latency = lat; dram_interval = 1; queue_depth = 4 }) kernels)
    [ 1; 2; 4; 8; 16; 32; 64 ];
  List.iter
    (fun iv ->
      List.iter (run "interval" { Config.dram_latency = 8; dram_interval = iv; queue_depth = 4 }) kernels)
    [ 1; 2; 4; 8 ];
  List.iter
    (fun q ->
      List.iter (run "queue" { Config.dram_latency = 8; dram_interval = 4; queue_depth = q }) kernels)
    [ 2; 4; 8 ];
  write_file out (Buffer.contents buf);
  Printf.printf "wrote %s\n" out

(* Test vectors for the Verilator run. File format, whitespace separated:
     threads
     256 instruction words (hex)
     256 data bytes (hex)
   Expected output (.hardcaml.out), one line each:
     cycles stall_cycles instrs mem_reqs divergent queue_full_cycles
     256 data bytes (hex) *)
let format_out (r : Harness.run_result) =
  let s = r.stats in
  Printf.sprintf "%d %d %d %d %d %d\n%s\n" s.cycles s.stall_cycles s.instrs s.mem_reqs
    s.divergent s.queue_full_cycles
    (String.concat " " (Array.to_list (Array.map (Printf.sprintf "%02x") r.mem)))

let cmd_gen_vectors pos flags =
  let dir = List.hd pos in
  let n = flag flags "n" 200 and seed = flag flags "seed" 7 in
  let config = config_of flags in
  ignore (Sys.command (Printf.sprintf "mkdir -p %s" (Filename.quote dir)));
  let st = Random.State.make [| seed |] in
  let named =
    List.map
      (fun (k : Kernels.kernel) -> (k.name, (Asm.assemble k.source).code, k.threads, k.mem))
      (Kernels.bench_set ())
  in
  let random =
    List.init n (fun i ->
      let code, threads, mem = Randprog.kernel st in
      (Printf.sprintf "random_%04d" i, code, threads, mem))
  in
  let sim = Harness.create config in
  List.iter
    (fun (name, code, threads, mem) ->
      let words = Array.init 256 (fun a -> if a < Array.length code then code.(a) else 0) in
      write_file
        (Filename.concat dir (name ^ ".in"))
        (Printf.sprintf "%d\n%s\n%s\n" threads
           (String.concat " " (Array.to_list (Array.map (Printf.sprintf "%04x") words)))
           (String.concat " " (Array.to_list (Array.map (Printf.sprintf "%02x") mem))));
      let r = Harness.run sim ~threads ~code mem in
      write_file (Filename.concat dir (name ^ ".hardcaml.out")) (format_out r))
    (named @ random);
  Printf.printf "wrote %d vectors to %s (%s)\n" (List.length named + n) dir
    (Config.to_string config)

let cmd_compare_vectors pos flags =
  let dir = List.hd pos in
  let out =
    Option.value (List.assoc_opt "out" flags) ~default:"results/verilator_agreement.json"
  in
  let names =
    Sys.readdir dir |> Array.to_list
    |> List.filter (fun f -> Filename.check_suffix f ".in")
    |> List.map Filename.chop_extension |> List.sort compare
  in
  let agree = ref 0 and differ = ref [] and missing = ref [] and cycles = ref 0 in
  List.iter
    (fun name ->
      let hc = Filename.concat dir (name ^ ".hardcaml.out")
      and vl = Filename.concat dir (name ^ ".verilator.out") in
      if not (Sys.file_exists vl) then missing := name :: !missing
      else if read_file hc = read_file vl then begin
        incr agree;
        cycles := !cycles + int_of_string (List.hd (String.split_on_char ' ' (read_file hc)))
      end
      else differ := name :: !differ)
    names;
  let q l = String.concat ", " (List.map (Printf.sprintf "%S") (List.rev l)) in
  write_file out
    (Printf.sprintf
       "{\n  \"vectors\": %d,\n  \"agree\": %d,\n  \"differ\": [%s],\n  \"missing\": [%s],\n  \"total_cycles_compared\": %d\n}\n"
       (List.length names) !agree (q !differ) (q !missing) !cycles);
  Printf.printf "%d vectors, %d agree, %d differ, %d missing -> %s\n" (List.length names) !agree
    (List.length !differ) (List.length !missing) out;
  if !differ <> [] || !missing <> [] then exit 1

let () =
  match Array.to_list Sys.argv |> List.tl with
  | "asm" :: rest ->
    let p, _ = load_program (List.hd (fst (parse rest))) in
    print_endline (Asm.to_hex p.code)
  | "disasm" :: rest ->
    let p, _ = load_program (List.hd (fst (parse rest))) in
    print_endline (Asm.disassemble p.code)
  | "run" :: rest ->
    let pos, flags = parse rest in
    cmd_run pos flags
  | "emit-verilog" :: rest ->
    let pos, flags = parse rest in
    let out = List.hd pos in
    write_file out (Gpu.verilog (config_of flags));
    Printf.printf "wrote %s (%s)\n" out (Config.to_string (config_of flags))
  | "fuzz" :: rest -> cmd_fuzz (snd (parse rest))
  | "bench" :: rest -> cmd_bench (snd (parse rest))
  | "gen-vectors" :: rest ->
    let pos, flags = parse rest in
    cmd_gen_vectors pos flags
  | "compare-vectors" :: rest ->
    let pos, flags = parse rest in
    cmd_compare_vectors pos flags
  | _ ->
    prerr_endline
      "usage: gpu (asm|disasm|run|emit-verilog|fuzz|bench|gen-vectors|compare-vectors) ...";
    exit 2
