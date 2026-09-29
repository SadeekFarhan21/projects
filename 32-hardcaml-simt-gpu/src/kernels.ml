(* Hand written kernels, their memory layouts and plain OCaml CPU references. *)

type kernel =
  { name : string
  ; source : string
  ; threads : int
  ; mem : int array (* initial data memory *)
  ; expected : int array -> bool (* checks the final memory *)
  ; outputs : int * int (* first output address, count *)
  }

(* vector add, C[i] = A[i] + B[i] for i < n. A at 0, B at n, C at 2n. *)
let vecadd_source n =
  Printf.sprintf
    {|; vector add, one thread per element
.threads %d
MUL R0, %%blockIdx, %%blockDim
ADD R0, R0, %%threadIdx        ; i = global thread index
CONST R1, #0                  ; base of A
CONST R2, #%d                 ; base of B
CONST R3, #%d                 ; base of C
ADD R4, R1, R0
LDR R4, R4                    ; A[i]
ADD R5, R2, R0
LDR R5, R5                    ; B[i]
ADD R6, R4, R5
ADD R7, R3, R0
STR R7, R6                    ; C[i] = A[i] + B[i]
RET
|}
    n n (2 * n)

let vecadd_ref a b = Array.map2 (fun x y -> (x + y) land 0xff) a b

let vecadd ?(seed = 1) n =
  if 3 * n > 256 then invalid_arg "vecadd: 3n must fit in 256 bytes";
  let st = Random.State.make [| seed |] in
  let a = Array.init n (fun _ -> Random.State.int st 256) in
  let b = Array.init n (fun _ -> Random.State.int st 256) in
  let mem = Array.make 256 0 in
  Array.blit a 0 mem 0 n;
  Array.blit b 0 mem n n;
  let c = vecadd_ref a b in
  { name = Printf.sprintf "vecadd_%d" n
  ; source = vecadd_source n
  ; threads = n
  ; mem
  ; expected = (fun m -> Array.sub m (2 * n) n = c)
  ; outputs = (2 * n, n)
  }

(* n x n matrix multiply, one thread per output element, loop over k.
   A at 0, B at n*n, C at 2*n*n, row major. *)
let matmul_source n =
  Printf.sprintf
    {|; matrix multiply C = A * B, one thread per element of C
.threads %d
MUL R0, %%blockIdx, %%blockDim
ADD R0, R0, %%threadIdx        ; i = global thread index
CONST R1, #1                  ; increment
CONST R2, #%d                 ; N
CONST R3, #0                  ; base of A
CONST R4, #%d                 ; base of B
CONST R5, #%d                 ; base of C
DIV R6, R0, R2                ; row = i / N
MUL R7, R6, R2
SUB R7, R0, R7                ; col = i - row * N
CONST R8, #0                  ; acc
CONST R9, #0                  ; k
loop:
MUL R10, R6, R2
ADD R10, R10, R9
ADD R10, R10, R3              ; &A[row][k]
LDR R10, R10
MUL R11, R9, R2
ADD R11, R11, R7
ADD R11, R11, R4              ; &B[k][col]
LDR R11, R11
MUL R12, R10, R11
ADD R8, R8, R12               ; acc += A[row][k] * B[k][col]
ADD R9, R9, R1
CMP R9, R2
BRn loop                      ; while k < N
ADD R9, R5, R0
STR R9, R8                    ; C[row][col] = acc
RET
|}
    (n * n) n (n * n) (2 * n * n)

let matmul_ref n a b =
  Array.init (n * n) (fun idx ->
    let r = idx / n and c = idx mod n in
    let acc = ref 0 in
    for k = 0 to n - 1 do
      acc := !acc + (a.((r * n) + k) * b.((k * n) + c))
    done;
    !acc land 0xff)

let matmul ?(seed = 2) n =
  let nn = n * n in
  if 3 * nn > 256 then invalid_arg "matmul: 3n^2 must fit in 256 bytes";
  let st = Random.State.make [| seed |] in
  (* small values keep the 4x4 case exact; wraparound is checked anyway *)
  let a = Array.init nn (fun _ -> Random.State.int st 16) in
  let b = Array.init nn (fun _ -> Random.State.int st 16) in
  let mem = Array.make 256 0 in
  Array.blit a 0 mem 0 nn;
  Array.blit b 0 mem nn nn;
  let c = matmul_ref n a b in
  { name = Printf.sprintf "matmul_%dx%d" n n
  ; source = matmul_source n
  ; threads = nn
  ; mem
  ; expected = (fun m -> Array.sub m (2 * nn) nn = c)
  ; outputs = (2 * nn, nn)
  }

let standard () = [ vecadd 8; matmul 4 ]
let bench_set () = [ vecadd 8; vecadd 32; vecadd 64; matmul 2; matmul 4; matmul 8 ]
