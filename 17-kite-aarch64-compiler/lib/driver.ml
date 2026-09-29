(* Pipeline glue shared by the CLI, the test runner and the benchmarks. *)

type opt_level = O0 | O1 | O2
(* O0: no IR optimisation, every temp in a stack slot.
   O1: IR optimisations, every temp in a stack slot.
   O2: IR optimisations + linear-scan register allocation. *)

let opt_of_string = function
  | "0" -> O0 | "1" -> O1 | "2" -> O2
  | s -> failwith ("unknown optimisation level -O" ^ s)

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic;
  s

let write_file path s =
  let oc = open_out_bin path in
  output_string oc s;
  close_out oc

(* Parse and type-check. Raises Diag.Compile_error. *)
let frontend (src : string) : Ast.program =
  let p = Parser.parse_program src in
  Typecheck.check_program p;
  p

let to_ir ~opt (p : Ast.program) : Ir.program =
  let ir = Lower.lower_program p in
  if opt = O0 then ir else Opt.optimize ir

let to_asm ~opt (ir : Ir.program) : string =
  let alloc_for f = if opt = O2 then Regalloc.allocate f else Codegen.all_slots f in
  Codegen.emit_program ir alloc_for

let run_cmd (argv : string list) : int * string =
  let cmd = String.concat " " (List.map Filename.quote argv) ^ " 2>&1" in
  let ic = Unix.open_process_in cmd in
  let b = Buffer.create 256 in
  (try
     while true do
       Buffer.add_channel b ic 1
     done
   with End_of_file -> ());
  let code = match Unix.close_process_in ic with Unix.WEXITED c -> c | _ -> 255 in
  (code, Buffer.contents b)

(* Compile the C runtime once per runtime version into a cache dir. *)
let runtime_object () : string =
  let digest = Digest.to_hex (Digest.string Runtime_src.source) in
  let dir = Filename.concat (Filename.get_temp_dir_name ()) ("kite-rt-" ^ String.sub digest 0 12) in
  let obj = Filename.concat dir "kite_rt.o" in
  if not (Sys.file_exists obj) then begin
    (try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ());
    let c = Filename.concat dir (Printf.sprintf "kite_rt_%d.c" (Unix.getpid ())) in
    write_file c Runtime_src.source;
    let tmp_obj = Filename.concat dir (Printf.sprintf "kite_rt_%d.o" (Unix.getpid ())) in
    let code, output = run_cmd [ "cc"; "-O2"; "-c"; c; "-o"; tmp_obj ] in
    if code <> 0 then failwith ("failed to compile runtime:\n" ^ output);
    Unix.rename tmp_obj obj
  end;
  obj

(* Assemble [asm] and link it with the runtime into [exe]. *)
let assemble_and_link ~asm ~exe =
  let s_file = exe ^ ".s" in
  write_file s_file asm;
  let code, output = run_cmd [ "cc"; "-o"; exe; s_file; runtime_object () ] in
  if code <> 0 then failwith ("cc failed:\n" ^ output)

let compile_file ~opt ~src_path ~exe ~keep_asm =
  let p = frontend (read_file src_path) in
  let asm = to_asm ~opt (to_ir ~opt p) in
  assemble_and_link ~asm ~exe;
  if not keep_asm then Sys.remove (exe ^ ".s")
