(* Memory controller and simulated DRAM.

     thread LSUs --pending--> [arbiter] --push--> [request queue] --pop--> [DRAM pipeline] --> response
                  <--grant---                                                 latency L        (tid, data)

   - The arbiter grants at most one pending LSU per cycle, lowest thread index
     first, and only when the queue has room.
   - DRAM accepts the queue head when it is not busy, then stays busy for
     [dram_interval] cycles. Every accepted request comes out of a
     [dram_latency] stage pipeline in order.
   - The data memory is read and written when a request leaves the pipeline,
     so requests take effect in the order they were queued. That ordering is
     what makes "highest thread wins" hold for colliding stores. *)
open Hardcaml
open Signal

let tid_bits = 2
let addr_bits = 8
let data_bits = 8

type lsu_request =
  { pending : Signal.t
  ; addr : Signal.t
  ; we : Signal.t
  ; wdata : Signal.t
  }

type response =
  { valid : Signal.t
  ; tid : Signal.t
  ; rdata : Signal.t
  }

type t =
  { grant : Signal.t array
  ; resp : response
  ; host_rdata : Signal.t
  ; queue_full_stall : Signal.t (* some LSU is pending but the queue is full *)
  ; queue_count : Signal.t
  ; dram_accept : Signal.t
  }

(* entry layout, msb first: tid, we, addr, wdata *)
let pack ~tid ~we ~addr ~wdata = concat_msb [ tid; we; addr; wdata ]

let unpack e =
  let wdata = sel_bottom e data_bits in
  let addr = select e (data_bits + addr_bits - 1) data_bits in
  let we = bit e (data_bits + addr_bits) in
  let tid = select e (data_bits + addr_bits + tid_bits) (data_bits + addr_bits + 1) in
  (tid, we, addr, wdata)

let create ~(config : Config.t) ~spec ~clock ~(reqs : lsu_request array) ~host_we ~host_addr
    ~host_wdata ~host_raddr =
  let n = Array.length reqs in
  (* Arbiter: fixed priority, lowest index wins. *)
  let queue_full = wire 1 in
  let lower_pending = Array.make n gnd in
  for t = 1 to n - 1 do
    lower_pending.(t) <- lower_pending.(t - 1) |: reqs.(t - 1).pending
  done;
  let grant =
    Array.init n (fun t ->
      reqs.(t).pending &: ~:(lower_pending.(t)) &: ~:queue_full -- Printf.sprintf "grant%d" t)
  in
  let any_pending = List.fold_left ( |: ) gnd (Array.to_list (Array.map (fun r -> r.pending) reqs)) in
  let push = any_pending &: ~:queue_full in
  let granted_tid = onehot_to_binary (concat_lsb (Array.to_list grant)) in
  let sel f = mux granted_tid (Array.to_list (Array.map f reqs)) in
  let entry =
    pack
      ~tid:(uresize granted_tid tid_bits)
      ~we:(sel (fun r -> r.we))
      ~addr:(sel (fun r -> r.addr))
      ~wdata:(sel (fun r -> r.wdata))
  in
  (* Request queue. *)
  let pop = wire 1 in
  let q = Req_queue.create ~spec ~depth:config.queue_depth ~push ~pop entry in
  queue_full <== q.full;
  (* DRAM front end: accept when the queue has a head and DRAM is idle. *)
  let ready =
    if config.dram_interval = 1 then vdd
    else
      let open Always in
      let w = num_bits_to_represent config.dram_interval in
      let busy = Variable.reg spec ~width:w in
      compile
        [ if_ pop
            [ busy <-- of_int ~width:w (config.dram_interval - 1) ]
            [ when_ (busy.value <>:. 0) [ busy <-- busy.value -:. 1 ] ]
        ];
      busy.value ==:. 0
  in
  let accept = ~:(q.empty) &: ready -- "dram_accept" in
  pop <== accept;
  (* Fixed latency pipeline. *)
  let rec pipe k valid e =
    if k = 0 then (valid, e)
    else
      pipe (k - 1)
        (reg spec valid -- Printf.sprintf "dram_stage%d_valid" (config.dram_latency - k + 1))
        (reg spec e)
  in
  let out_valid, out_e = pipe config.dram_latency accept q.q in
  let out_tid, out_we, out_addr, out_wdata = unpack out_e in
  (* Data memory, shared with the host load/readback port. The host only
     writes while the GPU is idle. *)
  let dram_write = out_valid &: out_we in
  let read_data =
    multiport_memory 256
      ~write_ports:
        [| { Write_port.write_clock = clock
           ; write_address = mux2 dram_write out_addr host_addr
           ; write_enable = dram_write |: host_we
           ; write_data = mux2 dram_write out_wdata host_wdata
           }
        |]
      ~read_addresses:[| out_addr; host_raddr |]
  in
  { grant
  ; resp = { valid = out_valid; tid = out_tid; rdata = read_data.(0) }
  ; host_rdata = read_data.(1)
  ; queue_full_stall = any_pending &: queue_full
  ; queue_count = q.count
  ; dram_accept = accept
  }
