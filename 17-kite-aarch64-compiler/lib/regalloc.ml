(* Linear-scan register allocation (Poletto and Sarkar, 1999).

   Blocks are laid out in list order and every instruction gets a
   position. Each temp gets one conservative live interval
   [first position, last position], extended to cover every block where
   it is live-in or live-out (from the dataflow liveness in Opt). The
   intervals are scanned in order of start; a temp gets a free register
   from x19-x28 if one is available, otherwise the interval that ends
   furthest away is spilled to a stack slot.

   Only callee-saved registers are handed out, so values survive calls
   without any caller-save code; the price is saving/restoring the ones
   used in the prologue/epilogue. *)

open Ir

let pool = [ "x19"; "x20"; "x21"; "x22"; "x23"; "x24"; "x25"; "x26"; "x27"; "x28" ]

let allocate (f : func) : Codegen.alloc =
  let live_in, live_out = Opt.liveness f in
  let start = Hashtbl.create 64 and stop = Hashtbl.create 64 in
  let touch t p =
    (match Hashtbl.find_opt start t with
     | Some s when s <= p -> ()
     | _ -> Hashtbl.replace start t p);
    match Hashtbl.find_opt stop t with
    | Some e when e >= p -> ()
    | _ -> Hashtbl.replace stop t p
  in
  List.iter (fun t -> touch t 0) f.params;
  Option.iter (fun t -> touch t 0) f.env;
  let pos = ref 1 in
  List.iter
    (fun b ->
      let bstart = !pos in
      List.iter
        (fun i ->
          List.iter (fun t -> touch t !pos) (uses_of i);
          Option.iter (fun t -> touch t !pos) (def_of i);
          incr pos)
        b.instrs;
      let bend = !pos in
      List.iter (fun t -> touch t bend) (term_uses b.term);
      incr pos;
      Opt.IS.iter (fun t -> touch t bstart) (Hashtbl.find live_in b.label);
      Opt.IS.iter (fun t -> touch t bend) (Hashtbl.find live_out b.label))
    f.blocks;
  let intervals =
    Hashtbl.fold (fun t s acc -> (t, s, Hashtbl.find stop t) :: acc) start []
    |> List.sort (fun (t1, s1, _) (t2, s2, _) -> if s1 <> s2 then compare s1 s2 else compare t1 t2)
  in
  let assign : (temp, Codegen.location) Hashtbl.t = Hashtbl.create 64 in
  let nslots = ref 0 in
  let spill t = Hashtbl.replace assign t (Codegen.Slot !nslots); incr nslots in
  let free = ref pool in
  let used = Hashtbl.create 16 in
  (* active: (end, temp, reg), kept sorted by end *)
  let active = ref [] in
  List.iter
    (fun (t, s, e) ->
      (* expire intervals that ended before this one starts *)
      let expired, still = List.partition (fun (e', _, _) -> e' < s) !active in
      List.iter (fun (_, _, r) -> free := r :: !free) expired;
      active := still;
      match !free with
      | r :: rest ->
        free := rest;
        Hashtbl.replace used r ();
        Hashtbl.replace assign t (Codegen.Reg r);
        active := List.sort compare ((e, t, r) :: !active)
      | [] ->
        (* spill whichever of the current and the active intervals ends last *)
        let last_e, last_t, last_r = List.nth !active (List.length !active - 1) in
        if last_e > e then begin
          spill last_t;
          Hashtbl.replace assign t (Codegen.Reg last_r);
          active := List.sort compare ((e, t, last_r) :: List.filter (fun (_, t', _) -> t' <> last_t) !active)
        end else spill t)
    intervals;
  let saved = List.filter (Hashtbl.mem used) pool in
  let lo = Hashtbl.create 32 in
  Hashtbl.iter (fun l s -> Hashtbl.replace lo l (Opt.IS.elements s)) live_out;
  {
    Codegen.loc_of =
      (fun t -> match Hashtbl.find_opt assign t with Some l -> l | None -> Codegen.Reg "x10" (* never live *));
    nslots = !nslots;
    saved;
    live_out = Some lo;
  }
