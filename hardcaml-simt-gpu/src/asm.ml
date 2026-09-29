(* A two pass assembler for the GPU ISA.

   Syntax, one statement per line:

     ; comment (also // comment)
     .threads 16            number of threads to launch
     .data 32 1 2 3         initial data memory, starting at address 32
     loop:                  label (may share a line with an instruction)
     ADD R1, R2, %threadIdx
     CONST R3, #10
     BRn loop
     RET
*)

type program =
  { code : int array
  ; threads : int option
  ; data : (int * int) list (* (address, value) pairs *)
  }

exception Error of int * string

let fail line fmt = Printf.ksprintf (fun s -> raise (Error (line, s))) fmt

let strip_comment s =
  let cut s pat =
    let n = String.length pat in
    let rec go i =
      if i + n > String.length s then s
      else if String.sub s i n = pat then String.sub s 0 i
      else go (i + 1)
    in
    go 0
  in
  cut (cut s ";") "//"

let tokens s =
  String.map (fun c -> if c = ',' || c = '\t' then ' ' else c) s
  |> String.split_on_char ' '
  |> List.filter (fun t -> t <> "")

let parse_int line s =
  let s = if String.length s > 0 && s.[0] = '#' then String.sub s 1 (String.length s - 1) else s in
  match int_of_string_opt s with
  | Some v -> v
  | None -> fail line "bad number %S" s

let parse_reg line s =
  match String.lowercase_ascii s with
  | "%blockidx" -> Isa.reg_block_idx
  | "%blockdim" -> Isa.reg_block_dim
  | "%threadidx" -> Isa.reg_thread_idx
  | l when String.length l >= 2 && l.[0] = 'r' -> (
    match int_of_string_opt (String.sub l 1 (String.length l - 1)) with
    | Some r when r >= 0 && r <= 15 -> r
    | _ -> fail line "bad register %S" s)
  | _ -> fail line "bad register %S" s

let is_label_def t = String.length t > 1 && t.[String.length t - 1] = ':'

(* Split a line into (labels defined, remaining tokens). *)
let split_labels toks =
  let rec go acc = function
    | t :: rest when is_label_def t -> go (String.sub t 0 (String.length t - 1) :: acc) rest
    | rest -> (List.rev acc, rest)
  in
  go [] toks

let assemble (src : string) : program =
  let lines = String.split_on_char '\n' src in
  let labels = Hashtbl.create 16 in
  (* Pass 1: assign addresses to labels. *)
  let pc = ref 0 in
  List.iteri
    (fun i l ->
      let line = i + 1 in
      let defs, rest = split_labels (tokens (strip_comment l)) in
      List.iter
        (fun d ->
          if Hashtbl.mem labels d then fail line "duplicate label %s" d;
          Hashtbl.replace labels d !pc)
        defs;
      match rest with
      | [] -> ()
      | t :: _ when t.[0] = '.' -> ()
      | _ -> incr pc)
    lines;
  (* Pass 2: encode. *)
  let code = ref [] and threads = ref None and data = ref [] in
  List.iteri
    (fun i l ->
      let line = i + 1 in
      let _, rest = split_labels (tokens (strip_comment l)) in
      let reg = parse_reg line and num = parse_int line in
      let target s =
        match Hashtbl.find_opt labels s with
        | Some a -> a
        | None -> num s
      in
      let emit ins = code := Isa.encode ins :: !code in
      match rest with
      | [] -> ()
      | [ ".threads"; n ] -> threads := Some (num n)
      | ".data" :: addr :: vs ->
        let a = num addr in
        List.iteri (fun k v -> data := (a + k, num v land 0xff) :: !data) vs
      | d :: _ when d.[0] = '.' -> fail line "unknown directive %s" d
      | m :: args -> (
        let m' = String.uppercase_ascii m in
        try
          match (m', args) with
          | "NOP", [] -> emit Isa.Nop
          | "RET", [] -> emit Isa.Ret
          | "CMP", [ s; t ] -> emit (Isa.Cmp (reg s, reg t))
          | "ADD", [ d; s; t ] -> emit (Isa.Add (reg d, reg s, reg t))
          | "SUB", [ d; s; t ] -> emit (Isa.Sub (reg d, reg s, reg t))
          | "MUL", [ d; s; t ] -> emit (Isa.Mul (reg d, reg s, reg t))
          | "DIV", [ d; s; t ] -> emit (Isa.Div (reg d, reg s, reg t))
          | "LDR", [ d; s ] -> emit (Isa.Ldr (reg d, reg s))
          | "STR", [ s; t ] -> emit (Isa.Str (reg s, reg t))
          | "CONST", [ d; v ] -> emit (Isa.Const (reg d, num v))
          | _, [ t ] when String.length m' >= 2 && String.sub m' 0 2 = "BR" ->
            let flags = String.lowercase_ascii (String.sub m 2 (String.length m - 2)) in
            let has c = String.contains flags c in
            String.iter
              (fun c -> if not (c = 'n' || c = 'z' || c = 'p') then fail line "bad branch %s" m)
              flags;
            (* the flags are taken literally: plain BR is never taken, BRnzp always *)
            emit (Isa.Br { n = has 'n'; z = has 'z'; p = has 'p'; target = target t })
          | _ -> fail line "cannot parse %S" (String.trim l)
        with Invalid_argument msg -> fail line "%s" msg))
    lines;
  let code = Array.of_list (List.rev !code) in
  if Array.length code > 256 then fail 0 "program has %d instructions, max 256" (Array.length code);
  { code; threads = !threads; data = List.rev !data }

let of_instrs ?threads ?(data = []) instrs =
  { code = Array.of_list (List.map Isa.encode instrs); threads; data }

let disassemble code =
  Array.to_list code
  |> List.mapi (fun i w -> Printf.sprintf "%3d  %04x  %s" i w (Isa.to_string (Isa.decode w)))
  |> String.concat "\n"

let to_hex code = Array.to_list code |> List.map (Printf.sprintf "%04x") |> String.concat "\n"
