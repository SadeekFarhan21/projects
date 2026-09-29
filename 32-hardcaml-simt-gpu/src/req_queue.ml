(* A small register based FIFO: the request queue between the arbiter and DRAM.
   The head entry is visible combinationally on [q]. Callers must not push
   when [full] or pop when [empty]; the testbench checks this never happens. *)
open Hardcaml
open Signal

type t =
  { q : Signal.t
  ; empty : Signal.t
  ; full : Signal.t
  ; count : Signal.t
  }

let create ~spec ~depth ~push ~pop d =
  let abits =
    let rec lg n = if n <= 1 then 0 else 1 + lg ((n + 1) / 2) in
    lg depth
  in
  let open Always in
  let wr_ptr = Variable.reg spec ~width:abits in
  let rd_ptr = Variable.reg spec ~width:abits in
  let count = Variable.reg spec ~width:(abits + 1) in
  let entries =
    Array.init depth (fun k ->
      reg spec ~enable:(push &: (wr_ptr.value ==:. k)) d -- Printf.sprintf "queue_entry%d" k)
  in
  compile
    [ when_ push [ wr_ptr <-- wr_ptr.value +:. 1 ]
    ; when_ pop [ rd_ptr <-- rd_ptr.value +:. 1 ]
    ; if_ (push &: ~:pop)
        [ count <-- count.value +:. 1 ]
        [ when_ (pop &: ~:push) [ count <-- count.value -:. 1 ] ]
    ];
  { q = mux rd_ptr.value (Array.to_list entries)
  ; empty = count.value ==:. 0
  ; full = count.value ==:. depth
  ; count = count.value
  }
