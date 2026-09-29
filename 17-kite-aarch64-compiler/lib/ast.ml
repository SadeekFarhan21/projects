(* Abstract syntax tree.

   The parser builds this tree; the type checker then fills in the
   mutable annotation fields (expression types, variable resolutions,
   lambda captures, field indices). The interpreter and the IR lowering
   both consume the annotated ("typed") tree. *)

type loc = Diag.loc

(* Types as written in source. *)
type ty_expr =
  | TEName of string * loc           (* int, bool, string, or a struct name *)
  | TEArray of ty_expr * loc         (* [T] *)
  | TEFn of ty_expr list * ty_expr option * loc  (* fn(T, U) -> R *)

(* Semantic types. *)
type ty =
  | TInt
  | TBool
  | TStr
  | TUnit
  | TArray of ty
  | TStruct of string
  | TFun of ty list * ty
  | TUnknown  (* placeholder before type checking *)

let rec show_ty = function
  | TInt -> "int"
  | TBool -> "bool"
  | TStr -> "string"
  | TUnit -> "unit"
  | TArray t -> "[" ^ show_ty t ^ "]"
  | TStruct s -> s
  | TFun (ps, r) ->
    "fn(" ^ String.concat ", " (List.map show_ty ps) ^ ")"
    ^ (if r = TUnit then "" else " -> " ^ show_ty r)
  | TUnknown -> "?"

type binop =
  | Add | Sub | Mul | Div | Mod
  | Lt | Le | Gt | Ge | Eq | Ne
  | And | Or
  | BAnd | BOr | BXor | Shl | Shr

type unop = Neg | Not

type builtin =
  | BPrint | BPrintln | BLen | BToStr | BSubstr | BChr | BAssert

let builtins =
  [ ("print", BPrint); ("println", BPrintln); ("len", BLen); ("to_str", BToStr);
    ("substr", BSubstr); ("chr", BChr); ("assert", BAssert) ]

(* What a name refers to, filled in by the checker. Locals get a
   program-wide unique id. *)
type var_res =
  | Unresolved
  | RLocal of int
  | RFunc of string
  | RBuiltin of builtin

type expr = { desc : expr_desc; loc : loc; mutable ty : ty }

and expr_desc =
  | IntLit of int64
  | BoolLit of bool
  | StrLit of string
  | Var of string * var_res ref
  | Binary of binop * expr * expr
  | Unary of unop * expr
  | Call of expr * expr list
  | Index of expr * expr
  | Field of expr * string * int ref           (* field index, set by checker *)
  | ArrayLit of expr list
  | ArrayRepeat of expr * expr                  (* [init; count] *)
  | StructLit of string * (string * expr) list
  | Lambda of lambda

and lambda = {
  lparams : param list;
  lret : ty_expr option;
  lbody : block;
  mutable lcaptures : (int * ty) list;  (* captured local ids, in order *)
  mutable lparam_ids : int list;
  mutable lret_ty : ty;
  mutable lid : int;                    (* unique lambda number *)
}

and param = { pname : string; pty : ty_expr; ploc : loc }

and stmt = { sdesc : stmt_desc; sloc : loc }

and stmt_desc =
  | Let of bool * string * ty_expr option * expr * int ref  (* mutable?, name, annot, init, id *)
  | Assign of expr * expr
  | ExprStmt of expr
  | If of expr * block * block option
  | While of expr * block
  | For of string * expr * expr * block * int ref
  | Return of expr option
  | Break
  | Continue
  | Block of block

and block = stmt list

type fn_decl = {
  fname : string;
  fparams : param list;
  fret : ty_expr option;
  fbody : block;
  floc : loc;
  mutable fparam_ids : int list;
  mutable fparam_tys : ty list;
  mutable fret_ty : ty;
}

type struct_decl = { sname : string; sfields : (string * ty_expr * loc) list; sdloc : loc }

type decl = DFn of fn_decl | DStruct of struct_decl

type program = decl list

let mk desc loc = { desc; loc; ty = TUnknown }
