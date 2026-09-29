(* Differential test runner.

   For every tests/programs/NAME.kite:
     1. run the reference interpreter,
     2. compile natively at -O0, -O1 and -O2 and run each executable
        (at most 2 at a time, each killed after 10 seconds),
   and require all four transcripts to equal tests/programs/NAME.out.
   A transcript is stdout, followed (only if the exit status is non-zero
   or stderr is non-empty) by the exit status and stderr.

   For every tests/fail/NAME.kite, the first line is
     // expect: LINE:COL: PHASE error: MESSAGE
   and compilation must fail with exactly that diagnostic.

   Usage: run_programs PROGRAMS_DIR FAIL_DIR [--update] [--only NAME]
   --update rewrites .out files from the interpreter (review the diff!). *)

open Kite

let transcript out code err =
  if code = 0 && err = "" then out
  else Printf.sprintf "%s--- exit %d\n--- stderr\n%s" out code err

let read = Driver.read_file

(* Each compiled test program is killed after this many seconds, so a
   miscompiled infinite loop shows up as a failure instead of a hang. *)
let timeout_s =
  match Sys.getenv_opt "KITE_TEST_TIMEOUT" with Some s -> int_of_string s | None -> 10

(* The machine is shared: keep at most this many build+run jobs at once. *)
let jobs_parallel = 2

let show_diff expected actual =
  let el = String.split_on_char '\n' expected and al = String.split_on_char '\n' actual in
  let rec go i el al =
    match el, al with
    | [], [] -> ()
    | e :: er, a :: ar when e = a -> go (i + 1) er ar
    | e, a ->
      Printf.printf "      first difference at line %d:\n        expected: %s\n        actual:   %s\n" i
        (match e with x :: _ -> String.escaped x | [] -> "<eof>")
        (match a with x :: _ -> String.escaped x | [] -> "<eof>")
  in
  go 1 el al

let () =
  let args = Array.to_list Sys.argv |> List.tl in
  let prog_dir, fail_dir = match args with p :: f :: _ -> (p, f) | _ -> failwith "usage" in
  let update = List.mem "--update" args in
  let only =
    let rec f = function "--only" :: n :: _ -> Some n | _ :: r -> f r | [] -> None in
    f args
  in
  let tmp = Filename.concat (Filename.get_temp_dir_name ()) (Printf.sprintf "kite-tests-%d" (Unix.getpid ())) in
  Unix.mkdir tmp 0o755;
  let pass = ref 0 and fail = ref 0 and failures = ref [] and timeouts = ref 0 in
  let check name what expected actual =
    if expected = actual then incr pass
    else begin
      incr fail;
      failures := (name ^ " [" ^ what ^ "]") :: !failures;
      Printf.printf "FAIL %s [%s]\n" name what;
      show_diff expected actual
    end
  in
  let files d =
    Sys.readdir d |> Array.to_list
    |> List.filter (fun f -> Filename.check_suffix f ".kite")
    |> List.filter (fun f -> match only with Some n -> Filename.remove_extension f = n | None -> true)
    |> List.sort compare
  in
  let progs = files prog_dir in
  (* Phase 1 (in process): front end, interpreter, and assembly for every
     program at every level. Phase 2 (parallel, via xargs -P): assemble,
     link and run each executable. Phase 3: compare transcripts. *)
  let rt_obj = Driver.runtime_object () in
  let jobs = ref [] in
  let expected_of = Hashtbl.create 64 in
  List.iter
    (fun f ->
      let name = Filename.remove_extension f in
      let path = Filename.concat prog_dir f in
      let expected_path = Filename.concat prog_dir (name ^ ".out") in
      match Driver.frontend (read path) with
      | exception Diag.Compile_error (ph, loc, msg) ->
        incr fail;
        failures := name :: !failures;
        Printf.printf "FAIL %s: does not compile: %s\n" name (Diag.header ph loc msg)
      | ast ->
        let o, c, e = Interp.run ast in
        let interp = transcript o c e in
        if update then Driver.write_file expected_path interp;
        let expected = if Sys.file_exists expected_path then read expected_path else "<missing .out file>" in
        Hashtbl.replace expected_of name expected;
        check name "interp" expected interp;
        List.iter
          (fun (lvl, tag) ->
            let exe = Filename.concat tmp (name ^ "_" ^ tag) in
            match Driver.to_asm ~opt:lvl (Driver.to_ir ~opt:lvl ast) with
            | asm ->
              Driver.write_file (exe ^ ".s") asm;
              jobs := (name, tag, exe) :: !jobs
            | exception e ->
              incr fail;
              failures := (name ^ " [" ^ tag ^ "]") :: !failures;
              Printf.printf "FAIL %s [%s]: code generation raised %s\n" name tag (Printexc.to_string e))
          [ (Driver.O0, "O0"); (Driver.O1, "O1"); (Driver.O2, "O2") ])
    progs;
  let jobs = List.rev !jobs in
  let script = Filename.concat tmp "job.sh" in
  Driver.write_file script
    (Printf.sprintf
       "#!/bin/sh\nexe=\"$1\"\nif cc -o \"$exe\" \"$exe.s\" %s > \"$exe.ccerr\" 2>&1; then\n  perl -e 'alarm shift @ARGV; exec @ARGV or die' %d \"$exe\" > \"$exe.stdout\" 2> \"$exe.stderr\"\n  echo $? > \"$exe.code\"\nelse\n  echo cc > \"$exe.code\"\nfi\n"
       (Filename.quote rt_obj) timeout_s);
  let list = Filename.concat tmp "jobs.txt" in
  Driver.write_file list (String.concat "\n" (List.map (fun (_, _, e) -> e) jobs) ^ "\n");
  ignore (Sys.command (Printf.sprintf "xargs -P %d -n 1 sh %s < %s" jobs_parallel (Filename.quote script) (Filename.quote list)));
  List.iter
    (fun (name, tag, exe) ->
      let code = try String.trim (read (exe ^ ".code")) with Sys_error _ -> "missing" in
      if code = "142" then begin
        incr fail;
        incr timeouts;
        failures := (name ^ " [" ^ tag ^ "]") :: !failures;
        Printf.printf "FAIL %s [%s]: timed out after %d s (killed)\n" name tag timeout_s
      end else if code = "missing" then begin
        incr fail;
        failures := (name ^ " [" ^ tag ^ "]") :: !failures;
        Printf.printf "FAIL %s [%s]: no result (job did not run)\n" name tag
      end else if code = "cc" then begin
        incr fail;
        failures := (name ^ " [" ^ tag ^ "]") :: !failures;
        Printf.printf "FAIL %s [%s]: cc failed:\n%s\n" name tag (read (exe ^ ".ccerr"))
      end else
        check name tag (Hashtbl.find expected_of name)
          (transcript (read (exe ^ ".stdout")) (int_of_string code) (read (exe ^ ".stderr"))))
    jobs;
  let fails = files fail_dir in
  List.iter
    (fun f ->
      let name = Filename.remove_extension f in
      let src = read (Filename.concat fail_dir f) in
      let first = List.hd (String.split_on_char '\n' src) in
      let prefix = "// expect: " in
      let n = String.length prefix in
      if String.length first < n || String.sub first 0 n <> prefix then begin
        incr fail;
        Printf.printf "FAIL %s: missing '// expect:' line\n" name
      end else begin
        let expected = String.sub first n (String.length first - n) in
        let actual =
          match Driver.frontend src with
          | _ -> "<compiled successfully>"
          | exception Diag.Compile_error (ph, loc, msg) ->
            (* rendering must not crash either *)
            ignore (Diag.render ~file:f ~src ph loc msg);
            Diag.header ph loc msg
        in
        check ("fail/" ^ name) "diagnostic" expected actual
      end)
    fails;
  ignore (Sys.command ("rm -rf " ^ Filename.quote tmp));
  Printf.printf "\n%d programs x 4 runs (interp, O0, O1, O2) + %d compile-fail cases\n"
    (List.length progs) (List.length fails);
  Printf.printf "passed %d, failed %d (%d of them timeouts)\n" !pass !fail !timeouts;
  if !fail > 0 then begin
    List.iter (fun f -> Printf.printf "  failed: %s\n" f) (List.rev !failures);
    exit 1
  end
