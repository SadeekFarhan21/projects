(* kitec: the Kite compiler driver.

   kitec build FILE [-o EXE] [-O0|-O1|-O2] [--keep-asm]   compile to a native executable
   kitec run FILE                                          run with the reference interpreter
   kitec check FILE                                        parse and type-check only
   kitec fmt FILE                                          pretty-print canonical source
   kitec ir FILE [-O0|-O1]                                 dump the IR
   kitec asm FILE [-O0|-O1|-O2]                            print the assembly
   kitec tokens FILE                                       dump the token stream *)

open Kite

let usage () =
  prerr_string
    "usage: kitec <build|run|check|fmt|ir|asm|tokens> FILE [-o EXE] [-O0|-O1|-O2] [--keep-asm]\n";
  exit 2

let () =
  let args = Array.to_list Sys.argv |> List.tl in
  let cmd, rest = match args with c :: r -> (c, r) | [] -> usage () in
  let opt = ref Driver.O2 and out = ref None and file = ref None and keep = ref false in
  let rec parse = function
    | [] -> ()
    | "-o" :: x :: r -> out := Some x; parse r
    | "--keep-asm" :: r -> keep := true; parse r
    | s :: r when String.length s = 3 && String.sub s 0 2 = "-O" ->
      opt := Driver.opt_of_string (String.sub s 2 1); parse r
    | s :: r when !file = None && s.[0] <> '-' -> file := Some s; parse r
    | s :: _ -> prerr_endline ("unknown argument " ^ s); usage ()
  in
  parse rest;
  let path = match !file with Some f -> f | None -> usage () in
  let src = try Driver.read_file path with Sys_error m -> prerr_endline m; exit 2 in
  try
    match cmd with
    | "tokens" ->
      List.iter
        (fun (t : Lexer.t) -> Printf.printf "%d:%d\t%s\n" t.loc.line t.loc.col (Lexer.show t.tok))
        (Lexer.tokenize src)
    | "fmt" -> print_string (Pretty.program (Parser.parse_program src))
    | "check" -> ignore (Driver.frontend src)
    | "run" ->
      let p = Driver.frontend src in
      let stdout_s, code, stderr_s = Interp.run p in
      print_string stdout_s;
      flush stdout;
      prerr_string stderr_s;
      exit code
    | "ir" ->
      let opt = if !opt = Driver.O2 then Driver.O1 else !opt in
      let ir = Driver.to_ir ~opt (Driver.frontend src) in
      print_string (Ir.show_program ir);
      Printf.printf "\n// %d IR instructions (including terminators)\n" (Ir.program_instr_count ir)
    | "asm" ->
      let p = Driver.frontend src in
      print_string (Driver.to_asm ~opt:!opt (Driver.to_ir ~opt:!opt p))
    | "build" ->
      let exe =
        match !out with Some o -> o | None -> Filename.remove_extension (Filename.basename path)
      in
      Driver.compile_file ~opt:!opt ~src_path:path ~exe ~keep_asm:!keep
    | _ -> usage ()
  with
  | Diag.Compile_error (phase, loc, msg) ->
    prerr_string (Diag.render ~file:path ~src phase loc msg);
    exit 1
  | Failure m ->
    prerr_endline ("kitec: " ^ m);
    exit 1
