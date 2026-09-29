(* Random kernel generator for differential testing.

   Shape of a generated kernel:

     prologue   CONSTs into R0..R9, some mixed with %threadIdx / %blockIdx so
                threads hold different values
     segment    random ALU ops, CMPs, loads, stores and forward branches
     loop       R10 counts to R11 (1..3 iterations) around another segment
     segment
     epilogue   store R0..R9 of every thread to 96 + 10 * global_idx + r

   Forward branches never leave their segment and the only backward branch is
   the counted loop, so every kernel terminates. Branch conditions depend on
   per thread data, so divergent branches do happen; both models follow
   thread 0 and count them. Writes to R13..R15 are generated on purpose to
   check they are dropped. *)

open Isa

type item =
  | I of Isa.t
  | Fwd of bool * bool * bool * int (* nzp mask, skip this many items *)
  | Loop_back of int (* BRn to the item with this index *)

let gen_segment st ~len =
  let r () = Random.State.int st 10 (* R0..R9 *) in
  let src () =
    let k = Random.State.int st 16 in
    if k < 10 || k >= 13 then k else r ()
  in
  let dst () = if Random.State.int st 20 = 0 then 13 + Random.State.int st 3 else r () in
  let items = Array.make len (I Nop) in
  for k = 0 to len - 1 do
    items.(k) <-
      (match Random.State.int st 100 with
      | x when x < 14 -> I (Add (dst (), src (), src ()))
      | x when x < 26 -> I (Sub (dst (), src (), src ()))
      | x when x < 38 -> I (Mul (dst (), src (), src ()))
      | x when x < 46 -> I (Div (dst (), src (), src ()))
      | x when x < 56 -> I (Const (dst (), Random.State.int st 256))
      | x when x < 66 -> I (Cmp (src (), src ()))
      | x when x < 76 -> I (Ldr (dst (), src ()))
      | x when x < 86 -> I (Str (src (), src ()))
      | x when x < 88 -> I Nop
      | _ ->
        let remaining = len - 1 - k in
        let skip = if remaining = 0 then 0 else Random.State.int st (remaining + 1) in
        let b () = Random.State.bool st in
        Fwd (b (), b (), b (), skip))
  done;
  Array.to_list items

let generate st =
  let prologue =
    List.concat
      (List.init 10 (fun r ->
         let c = I (Const (r, Random.State.int st 256)) in
         match Random.State.int st 3 with
         | 0 -> [ c; I (Add (r, r, reg_thread_idx)) ]
         | 1 -> [ c; I (Mul (r, r, reg_thread_idx)); I (Add (r, r, reg_block_idx)) ]
         | _ -> [ c ]))
  in
  let seg () = gen_segment st ~len:(4 + Random.State.int st 12) in
  let s1 = seg () in
  let iters = 1 + Random.State.int st 3 in
  let loop_init = [ I (Const (10, 0)); I (Const (11, iters)); I (Const (12, 1)) ] in
  let body = seg () in
  let loop_tail = [ I (Add (10, 10, 12)); I (Cmp (10, 11)) ] in
  let s2 = seg () in
  let epilogue =
    [ I (Mul (10, reg_block_idx, reg_block_dim))
    ; I (Add (10, 10, reg_thread_idx))
    ; I (Const (11, 10))
    ; I (Mul (10, 10, 11))
    ; I (Const (11, 96))
    ; I (Add (10, 10, 11))
    ; I (Const (12, 1))
    ]
    @ List.concat (List.init 10 (fun r -> [ I (Str (10, r)); I (Add (10, 10, 12)) ]))
    @ [ I Ret ]
  in
  let before_body = prologue @ s1 @ loop_init in
  let body_start = List.length before_body in
  let items =
    before_body @ body @ loop_tail @ [ Loop_back body_start ] @ s2 @ epilogue
  in
  List.mapi
    (fun idx it ->
      match it with
      | I ins -> ins
      | Fwd (n, z, p, skip) -> Br { n; z; p; target = idx + 1 + skip }
      | Loop_back target -> Br { n = true; z = false; p = false; target })
    items

(* A random kernel: program, thread count (1..16, so 1..4 blocks and partially
   filled last blocks) and random initial memory. *)
let kernel st =
  let prog = generate st in
  let threads = 1 + Random.State.int st 16 in
  let mem = Array.init 256 (fun _ -> Random.State.int st 256) in
  (Array.of_list (List.map Isa.encode prog), threads, mem)
