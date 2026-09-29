(* Hand-written lexer. Produces a list of tokens, each tagged with the
   line and column (both 1-based) of its first character. *)

open Diag

type token =
  | INT of int64
  | STR of string
  | IDENT of string
  (* keywords *)
  | FN | LET | VAR | IF | ELSE | WHILE | FOR | IN | RETURN | BREAK | CONTINUE
  | STRUCT | TRUE | FALSE
  (* punctuation *)
  | LPAREN | RPAREN | LBRACE | RBRACE | LBRACKET | RBRACKET
  | COMMA | SEMI | COLON | DOT | DOTDOT | ARROW
  (* operators *)
  | ASSIGN | EQ | NE | LT | LE | GT | GE
  | PLUS | MINUS | STAR | SLASH | PERCENT
  | AMP | PIPE | CARET | SHL | SHR | ANDAND | OROR | BANG
  | EOF

type t = { tok : token; loc : loc }

let keywords =
  [ ("fn", FN); ("let", LET); ("var", VAR); ("if", IF); ("else", ELSE);
    ("while", WHILE); ("for", FOR); ("in", IN); ("return", RETURN);
    ("break", BREAK); ("continue", CONTINUE); ("struct", STRUCT);
    ("true", TRUE); ("false", FALSE) ]

let show = function
  | INT n -> Int64.to_string n
  | STR s -> Printf.sprintf "%S" s
  | IDENT s -> s
  | FN -> "fn" | LET -> "let" | VAR -> "var" | IF -> "if" | ELSE -> "else"
  | WHILE -> "while" | FOR -> "for" | IN -> "in" | RETURN -> "return"
  | BREAK -> "break" | CONTINUE -> "continue" | STRUCT -> "struct"
  | TRUE -> "true" | FALSE -> "false"
  | LPAREN -> "(" | RPAREN -> ")" | LBRACE -> "{" | RBRACE -> "}"
  | LBRACKET -> "[" | RBRACKET -> "]" | COMMA -> "," | SEMI -> ";"
  | COLON -> ":" | DOT -> "." | DOTDOT -> ".." | ARROW -> "->"
  | ASSIGN -> "=" | EQ -> "==" | NE -> "!=" | LT -> "<" | LE -> "<="
  | GT -> ">" | GE -> ">=" | PLUS -> "+" | MINUS -> "-" | STAR -> "*"
  | SLASH -> "/" | PERCENT -> "%" | AMP -> "&" | PIPE -> "|" | CARET -> "^"
  | SHL -> "<<" | SHR -> ">>" | ANDAND -> "&&" | OROR -> "||" | BANG -> "!"
  | EOF -> "end of file"

let is_digit c = c >= '0' && c <= '9'
let is_hex c = is_digit c || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
let is_ident_start c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c = '_'
let is_ident c = is_ident_start c || is_digit c

let tokenize (src : string) : t list =
  let n = String.length src in
  let pos = ref 0 and line = ref 1 and col = ref 1 in
  let toks = ref [] in
  let peek k = if !pos + k < n then src.[!pos + k] else '\000' in
  let advance () =
    if src.[!pos] = '\n' then (incr line; col := 1) else incr col;
    incr pos
  in
  let here () = { line = !line; col = !col } in
  let emit tok loc = toks := { tok; loc } :: !toks in
  while !pos < n do
    let c = peek 0 in
    let start = here () in
    if c = ' ' || c = '\t' || c = '\r' || c = '\n' then advance ()
    else if c = '/' && peek 1 = '/' then
      while !pos < n && peek 0 <> '\n' do advance () done
    else if is_digit c then begin
      let buf = Buffer.create 16 in
      if c = '0' && (peek 1 = 'x' || peek 1 = 'X') then begin
        advance (); advance ();
        while is_hex (peek 0) || peek 0 = '_' do
          if peek 0 <> '_' then Buffer.add_char buf (peek 0);
          advance ()
        done;
        if Buffer.length buf = 0 then error Lex start "hex literal needs at least one digit";
        if is_ident_start (peek 0) then
          error Lex (here ()) "unexpected character '%c' after number" (peek 0);
        if Buffer.length buf > 16 then error Lex start "hex literal 0x%s does not fit in 64 bits" (Buffer.contents buf);
        emit (INT (Int64.of_string ("0x" ^ Buffer.contents buf))) start
      end else begin
        while is_digit (peek 0) || peek 0 = '_' do
          if peek 0 <> '_' then Buffer.add_char buf (peek 0);
          advance ()
        done;
        if is_ident_start (peek 0) then
          error Lex (here ()) "unexpected character '%c' after number" (peek 0);
        let s = Buffer.contents buf in
        match Int64.of_string_opt s with
        | Some v when String.length s <= 19 -> emit (INT v) start
        | _ -> error Lex start "integer literal %s is too large (max 9223372036854775807)" s
      end
    end
    else if is_ident_start c then begin
      let b = !pos in
      while is_ident (peek 0) do advance () done;
      let s = String.sub src b (!pos - b) in
      emit (match List.assoc_opt s keywords with Some k -> k | None -> IDENT s) start
    end
    else if c = '"' then begin
      advance ();
      let buf = Buffer.create 16 in
      let closed = ref false in
      while not !closed do
        if !pos >= n || peek 0 = '\n' then error Lex start "unterminated string literal";
        let ch = peek 0 in
        if ch = '"' then (advance (); closed := true)
        else if ch = '\\' then begin
          let esc_loc = here () in
          advance ();
          if !pos >= n then error Lex start "unterminated string literal";
          let e = peek 0 in
          (match e with
           | 'n' -> Buffer.add_char buf '\n'
           | 't' -> Buffer.add_char buf '\t'
           | 'r' -> Buffer.add_char buf '\r'
           | '0' -> Buffer.add_char buf '\000'
           | '\\' -> Buffer.add_char buf '\\'
           | '"' -> Buffer.add_char buf '"'
           | _ -> error Lex esc_loc "unknown escape sequence '\\%c'" e);
          advance ()
        end else (Buffer.add_char buf ch; advance ())
      done;
      emit (STR (Buffer.contents buf)) start
    end
    else begin
      let two = if !pos + 1 < n then String.sub src !pos 2 else "" in
      let tok2 =
        match two with
        | "==" -> Some EQ | "!=" -> Some NE | "<=" -> Some LE | ">=" -> Some GE
        | "&&" -> Some ANDAND | "||" -> Some OROR | "<<" -> Some SHL | ">>" -> Some SHR
        | "->" -> Some ARROW | ".." -> Some DOTDOT
        | _ -> None
      in
      match tok2 with
      | Some t -> advance (); advance (); emit t start
      | None ->
        let t =
          match c with
          | '(' -> LPAREN | ')' -> RPAREN | '{' -> LBRACE | '}' -> RBRACE
          | '[' -> LBRACKET | ']' -> RBRACKET | ',' -> COMMA | ';' -> SEMI
          | ':' -> COLON | '.' -> DOT | '=' -> ASSIGN | '<' -> LT | '>' -> GT
          | '+' -> PLUS | '-' -> MINUS | '*' -> STAR | '/' -> SLASH
          | '%' -> PERCENT | '&' -> AMP | '|' -> PIPE | '^' -> CARET | '!' -> BANG
          | _ ->
            if Char.code c < 32 || Char.code c > 126 then
              error Lex start "unexpected byte 0x%02x" (Char.code c)
            else error Lex start "unexpected character '%c'" c
        in
        advance (); emit t start
    end
  done;
  emit EOF (here ());
  List.rev !toks
