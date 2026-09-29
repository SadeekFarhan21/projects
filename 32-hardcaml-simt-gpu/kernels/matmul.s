; 4x4 matrix multiply C = A * B, one thread per element of C
; A at 0, B at 16, C at 32, all row major
.threads 16
.data 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
.data 16 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1

MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx        ; i = global thread index
CONST R1, #1                  ; increment
CONST R2, #4                  ; N
CONST R3, #0                  ; base of A
CONST R4, #16                 ; base of B
CONST R5, #32                 ; base of C
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
