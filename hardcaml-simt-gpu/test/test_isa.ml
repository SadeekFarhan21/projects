(* Unit tests: ISA encoding, assembler, interpreter on known kernels, and an
   exhaustive check of the ALU gates against the ISA arithmetic. *)
open Gpu_lib

let checks = ref 0
let failures = ref 0

let check name cond =
  incr checks;
  if not cond then begin
    incr failures;
    Printf.printf "FAIL %s\n%!" name
  end

let random_instr st =
  let r () = Random.State.int st 16 and b () = Random.State.bool st in
  match Random.State.int st 11 with
  | 0 -> Isa.Nop
  | 1 -> Isa.Br { n = b (); z = b (); p = b (); target = Random.State.int st 256 }
  | 2 -> Isa.Cmp (r (), r ())
  | 3 -> Isa.Add (r (), r (), r ())
  | 4 -> Isa.Sub (r (), r (), r ())
  | 5 -> Isa.Mul (r (), r (), r ())
  | 6 -> Isa.Div (r (), r (), r ())
  | 7 -> Isa.Ldr (r (), r ())
  | 8 -> Isa.Str (r (), r ())
  | 9 -> Isa.Const (r (), Random.State.int st 256)
  | _ -> Isa.Ret

let test_encoding () =
  let st = Random.State.make [| 11 |] in
  for _ = 1 to 20_000 do
    let i = random_instr st in
    check ("decode (encode i) = i for " ^ Isa.to_string i) (Isa.decode (Isa.encode i) = i)
  done

let test_assembler_roundtrip () =
  let st = Random.State.make [| 12 |] in
  for _ = 1 to 500 do
    let code = Array.init (1 + Random.State.int st 60) (fun _ -> Isa.encode (random_instr st)) in
    let text =
      Array.to_list code |> List.map (fun w -> Isa.to_string (Isa.decode w)) |> String.concat "\n"
    in
    let p = Asm.assemble text in
    check "disassemble then assemble is the identity" (p.code = code)
  done

let test_assembler_details () =
  let p =
    Asm.assemble
      {|.threads 6
        .data 10 1 2 0x10
        start: CONST R1, #5   ; comment
        loop:
        ADD R1, R1, %threadIdx // other comment
        BRnp loop
        BRnzp start
        RET|}
  in
  check "threads directive" (p.threads = Some 6);
  check "data directive" (p.data = [ (10, 1); (11, 2); (12, 16) ]);
  check "label resolution"
    (Array.to_list p.code
    = List.map Isa.encode
        [ Isa.Const (1, 5)
        ; Isa.Add (1, 1, 15)
        ; Isa.Br { n = true; z = false; p = true; target = 1 }
        ; Isa.Br { n = true; z = true; p = true; target = 0 }
        ; Isa.Ret
        ]);
  let raises src = try ignore (Asm.assemble src); false with Asm.Error _ -> true in
  check "bad register rejected" (raises "ADD R1, R16, R2");
  check "bad immediate rejected" (raises "CONST R1, #300");
  check "unknown mnemonic rejected" (raises "FOO R1");
  check "duplicate label rejected" (raises "a:\na:\nRET");
  check "bad branch flags rejected" (raises "BRx 0")

let test_interp_kernels () =
  List.iter
    (fun (k : Kernels.kernel) ->
      let p = Asm.assemble k.source in
      check ("threads directive in " ^ k.name) (p.threads = Some k.threads);
      let r = Interp.run ~threads:k.threads ~code:p.code k.mem in
      check ("interpreter matches CPU reference on " ^ k.name) (k.expected r.mem);
      check ("no divergence in " ^ k.name) (r.divergent = 0))
    (Kernels.bench_set () @ [ Kernels.vecadd 7; Kernels.vecadd 13; Kernels.matmul 3 ])

(* Build a tiny combinational circuit around the ALU and sweep all 2^16
   operand pairs through Cyclesim. *)
let test_alu_exhaustive () =
  let open Hardcaml in
  let a = Signal.input "a" 8 and b = Signal.input "b" 8 and op = Signal.input "op" 4 in
  let res = Alu.result ~op ~rs:a ~rt:b ~imm:(Signal.of_int ~width:8 0x5a) in
  let nzp = Alu.compare_nzp a b in
  let circ =
    Circuit.create_exn ~name:"alu" [ Signal.output "res" res; Signal.output "nzp" nzp ]
  in
  let sim = Cyclesim.create circ in
  let ia = Cyclesim.in_port sim "a" and ib = Cyclesim.in_port sim "b" in
  let iop = Cyclesim.in_port sim "op" in
  let ores = Cyclesim.out_port sim "res" and onzp = Cyclesim.out_port sim "nzp" in
  let bad = ref 0 in
  let ops =
    [ (Isa.op_add, fun x y -> (x + y) land 255)
    ; (Isa.op_sub, fun x y -> (x - y) land 255)
    ; (Isa.op_mul, fun x y -> x * y land 255)
    ; (Isa.op_div, Isa.div8)
    ; (Isa.op_const, fun _ _ -> 0x5a)
    ]
  in
  List.iter
    (fun (code, f) ->
      iop := Bits.of_int ~width:4 code;
      for x = 0 to 255 do
        for y = 0 to 255 do
          ia := Bits.of_int ~width:8 x;
          ib := Bits.of_int ~width:8 y;
          Cyclesim.cycle sim;
          if Bits.to_int !ores <> f x y then incr bad;
          if code = Isa.op_add && Bits.to_int !onzp <> Isa.nzp_of_compare x y then incr bad
        done
      done)
    ops;
  check (Printf.sprintf "ALU exhaustive (%d mismatches)" !bad) (!bad = 0)

let () =
  test_encoding ();
  test_assembler_roundtrip ();
  test_assembler_details ();
  test_interp_kernels ();
  test_alu_exhaustive ();
  Printf.printf "test_isa: %d checks, %d failures\n" !checks !failures;
  if !failures > 0 then exit 1
