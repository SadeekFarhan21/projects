(* Source positions and compile-time diagnostics.

   Every stage (lexer, parser, type checker) reports errors through
   [Compile_error], which carries a phase, a position and a message.
   [render] turns that into the familiar

     file.kite:3:9: type error: expected int, found bool
        |
      3 |   let x: int = true;
        |                ^

   format with a caret under the offending column. *)

type loc = { line : int; col : int }

let dummy = { line = 0; col = 0 }

type phase = Lex | Parse | Type

exception Compile_error of phase * loc * string

let phase_name = function
  | Lex -> "lex error"
  | Parse -> "parse error"
  | Type -> "type error"

let error phase loc fmt = Printf.ksprintf (fun m -> raise (Compile_error (phase, loc, m))) fmt

(* Header line only, used by tests: "3:9: type error: ..." *)
let header phase loc msg = Printf.sprintf "%d:%d: %s: %s" loc.line loc.col (phase_name phase) msg

let source_line src n =
  let lines = String.split_on_char '\n' src in
  match List.nth_opt lines (n - 1) with Some l -> l | None -> ""

let render ~file ~src phase loc msg =
  let b = Buffer.create 256 in
  Buffer.add_string b (Printf.sprintf "%s:%s\n" file (header phase loc msg));
  if loc.line > 0 then begin
    let text = source_line src loc.line in
    let num = string_of_int loc.line in
    let pad = String.make (String.length num) ' ' in
    Buffer.add_string b (Printf.sprintf " %s |\n" pad);
    Buffer.add_string b (Printf.sprintf " %s | %s\n" num text);
    (* Preserve tabs so the caret lines up with the source text. *)
    let caret_pad =
      String.init (max 0 (loc.col - 1)) (fun i ->
          if i < String.length text && text.[i] = '\t' then '\t' else ' ')
    in
    Buffer.add_string b (Printf.sprintf " %s | %s^\n" pad caret_pad)
  end;
  Buffer.contents b
