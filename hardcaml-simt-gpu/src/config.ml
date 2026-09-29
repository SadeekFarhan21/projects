(* Elaboration time parameters of the memory system. Each distinct value
   produces a different circuit. *)
type t =
  { dram_latency : int (* cycles from DRAM accepting a request to its response *)
  ; dram_interval : int (* minimum cycles between two accepted requests *)
  ; queue_depth : int (* entries in the request queue, a power of two >= 2 *)
  }

let default = { dram_latency = 8; dram_interval = 2; queue_depth = 4 }

let validate c =
  if c.dram_latency < 1 then invalid_arg "dram_latency must be >= 1";
  if c.dram_interval < 1 then invalid_arg "dram_interval must be >= 1";
  let pow2 n = n >= 2 && n land (n - 1) = 0 in
  if not (pow2 c.queue_depth) then invalid_arg "queue_depth must be a power of two >= 2"

let to_string c =
  Printf.sprintf "latency=%d interval=%d queue=%d" c.dram_latency c.dram_interval c.queue_depth
