(* Random differential testing.

   Generates random, well-typed Kite programs that stress integer
   arithmetic (wrapping, shifts, division), control flow and calls,
   then checks that the interpreter and the native code at -O0, -O1
   and -O2 print exactly the same thing.

   Each executable is killed after 10 seconds (a hang counts as a
   mismatch) and at most 2 build+run jobs run at once.

   Usage: fuzz COUNT SEED [OUT_DIR]
   Failing programs are copied to OUT_DIR (default: fuzz_failures/). *)

open Kite

let st = ref (Random.State.make [| 0 |])
let rint n = Random.State.int !st n
let pick l = List.nth l (rint (List.length l))

let nvars = 6

let literal () =
  match rint 10 with
  | 0 -> pick [ "9223372036854775807"; "(-9223372036854775807 - 1)"; "0x7fffffff"; "0xffffffff" ]
  | 1 -> pick [ "0"; "1"; "(-1)"; "63"; "64" ]
  | 2 -> string_of_int (rint 100000)
  | _ -> string_of_int (rint 20)

let rec expr loopvars depth =
  if depth = 0 || rint 4 = 0 then
    match rint 3 with
    | 0 -> literal ()
    | _ ->
      let vars = List.init nvars (fun i -> "v" ^ string_of_int i) @ loopvars in
      pick vars
  else
    let a () = expr loopvars (depth - 1) in
    match rint 14 with
    | 0 | 1 -> Printf.sprintf "(%s + %s)" (a ()) (a ())
    | 2 -> Printf.sprintf "(%s - %s)" (a ()) (a ())
    | 3 -> Printf.sprintf "(%s * %s)" (a ()) (a ())
    | 4 -> Printf.sprintf "(%s / (%s | 1))" (a ()) (a ())
    | 5 -> Printf.sprintf "(%s %% (%s | 1))" (a ()) (a ())
    | 6 -> Printf.sprintf "(%s & %s)" (a ()) (a ())
    | 7 -> Printf.sprintf "(%s ^ %s)" (a ()) (a ())
    | 8 -> Printf.sprintf "(%s << %s)" (a ()) (a ())
    | 9 -> Printf.sprintf "(%s >> %s)" (a ()) (a ())
    | 10 -> Printf.sprintf "(-%s)" (a ())
    | 11 -> Printf.sprintf "mix(%s, %s)" (a ()) (a ())
    | 12 -> Printf.sprintf "twice(%s)" (a ())
    | _ -> Printf.sprintf "pick(%s, %s, %s)" (cond loopvars (depth - 1)) (a ()) (a ())

and cond loopvars depth =
  let a () = expr loopvars depth in
  match rint 8 with
  | 0 -> Printf.sprintf "%s < %s" (a ()) (a ())
  | 1 -> Printf.sprintf "%s <= %s" (a ()) (a ())
  | 2 -> Printf.sprintf "%s == %s" (a ()) (a ())
  | 3 -> Printf.sprintf "%s != %s" (a ()) (a ())
  | 4 -> Printf.sprintf "%s > %s" (a ()) (a ())
  | 5 -> Printf.sprintf "(%s >= %s) && (%s < %s)" (a ()) (a ()) (a ()) (a ())
  | 6 -> Printf.sprintf "(%s == %s) || !(%s > %s)" (a ()) (a ()) (a ()) (a ())
  | _ -> Printf.sprintf "%s %% 2 == 0" (a ())

let rec stmts b ind loopvars depth n =
  for _ = 1 to n do
    let v = "v" ^ string_of_int (rint nvars) in
    match (if depth = 0 then rint 3 else rint 7) with
    | 0 | 1 -> Buffer.add_string b (Printf.sprintf "%s%s = %s;\n" ind v (expr loopvars 3))
    | 2 -> Buffer.add_string b (Printf.sprintf "%sprintln(%s);\n" ind (expr loopvars 2))
    | 3 | 4 ->
      Buffer.add_string b (Printf.sprintf "%sif %s {\n" ind (cond loopvars 2));
      stmts b (ind ^ "    ") loopvars (depth - 1) (1 + rint 3);
      Buffer.add_string b (Printf.sprintf "%s} else {\n" ind);
      stmts b (ind ^ "    ") loopvars (depth - 1) (1 + rint 3);
      Buffer.add_string b (Printf.sprintf "%s}\n" ind)
    | 5 ->
      let k = "k" ^ string_of_int depth in
      Buffer.add_string b (Printf.sprintf "%sfor %s in 0..%d {\n" ind k (1 + rint 6));
      stmts b (ind ^ "    ") (k :: loopvars) (depth - 1) (1 + rint 3);
      Buffer.add_string b (Printf.sprintf "%s}\n" ind)
    | _ ->
      (* a closure capturing an immutable snapshot *)
      Buffer.add_string b
        (Printf.sprintf "%s{\n%s    let snap = %s;\n%s    let g = fn(x: int) -> int { return x * 7 + snap; };\n%s    %s = g(%s);\n%s}\n"
           ind ind (expr loopvars 2) ind ind v (expr loopvars 2) ind)
  done

let program () =
  let b = Buffer.create 2048 in
  Buffer.add_string b
    "fn mix(a: int, b: int) -> int {\n    return (a ^ (b << 3)) - (b >> 2);\n}\n\n\
     fn twice(a: int) -> int {\n    return a + a;\n}\n\n\
     fn pick(c: bool, a: int, b: int) -> int {\n    if c {\n        return a;\n    }\n    return b;\n}\n\n\
     fn main() {\n";
  for i = 0 to nvars - 1 do
    Buffer.add_string b (Printf.sprintf "    var v%d = %s;\n" i (literal ()))
  done;
  stmts b "    " [] 3 (4 + rint 6);
  for i = 0 to nvars - 1 do
    Buffer.add_string b (Printf.sprintf "    println(v%d);\n" i)
  done;
  Buffer.add_string b "}\n";
  Buffer.contents b

let () =
  let count = int_of_string Sys.argv.(1) and seed = int_of_string Sys.argv.(2) in
  let outdir = if Array.length Sys.argv > 3 then Sys.argv.(3) else "fuzz_failures" in
  st := Random.State.make [| seed |];
  let tmp = Filename.concat (Filename.get_temp_dir_name ()) (Printf.sprintf "kite-fuzz-%d" (Unix.getpid ())) in
  Unix.mkdir tmp 0o755;
  let rt_obj = Driver.runtime_object () in
  let srcs = Array.init count (fun _ -> program ()) in
  let expected = Array.make count "" in
  let jobs = ref [] in
  Array.iteri
    (fun i src ->
      let ast = Driver.frontend src in
      let o, c, e = Interp.run ast in
      expected.(i) <- Printf.sprintf "%s[exit %d]%s" o c e;
      List.iter
        (fun (lvl, tag) ->
          let exe = Filename.concat tmp (Printf.sprintf "p%d_%s" i tag) in
          Driver.write_file (exe ^ ".s") (Driver.to_asm ~opt:lvl (Driver.to_ir ~opt:lvl ast));
          jobs := (i, tag, exe) :: !jobs)
        [ (Driver.O0, "O0"); (Driver.O1, "O1"); (Driver.O2, "O2") ])
    srcs;
  if Sys.getenv_opt "FUZZ_SHOW" <> None then
    Printf.printf "--- program 0 ---\n%s--- interpreter output ---\n%s\n" srcs.(0) expected.(0);
  let script = Filename.concat tmp "job.sh" in
  Driver.write_file script
    (Printf.sprintf
       "#!/bin/sh\nexe=\"$1\"\ncc -o \"$exe\" \"$exe.s\" %s && perl -e 'alarm shift @ARGV; exec @ARGV or die' %d \"$exe\" > \"$exe.out\" 2> \"$exe.err\"; echo $? > \"$exe.code\"\n"
       (Filename.quote rt_obj)
       (match Sys.getenv_opt "KITE_TEST_TIMEOUT" with Some s -> int_of_string s | None -> 10));
  let list = Filename.concat tmp "jobs.txt" in
  Driver.write_file list (String.concat "\n" (List.map (fun (_, _, e) -> e) !jobs) ^ "\n");
  ignore (Sys.command (Printf.sprintf "xargs -P 2 -n 1 sh %s < %s" (Filename.quote script) (Filename.quote list)));
  let bad = ref 0 and timeouts = ref 0 in
  List.iter
    (fun (i, tag, exe) ->
      let rd f = try Driver.read_file f with Sys_error _ -> "" in
      let actual =
        Printf.sprintf "%s[exit %s]%s" (rd (exe ^ ".out")) (String.trim (rd (exe ^ ".code"))) (rd (exe ^ ".err"))
      in
      if actual <> expected.(i) then begin
        incr bad;
        let timed_out = String.trim (rd (exe ^ ".code")) = "142" in
        if timed_out then incr timeouts;
        (try Unix.mkdir outdir 0o755 with Unix.Unix_error _ -> ());
        Driver.write_file (Filename.concat outdir (Printf.sprintf "fuzz_%d_%d.kite" seed i)) srcs.(i);
        Printf.printf "MISMATCH program %d at %s%s\n" i tag (if timed_out then " (timed out)" else "")
      end)
    (List.rev !jobs);
  ignore (Sys.command ("rm -rf " ^ Filename.quote tmp));
  Printf.printf "fuzz: seed %d, %d random programs x 3 optimisation levels, %d mismatches (%d of them timeouts)\n"
    seed count !bad !timeouts;
  if !bad > 0 then exit 1
