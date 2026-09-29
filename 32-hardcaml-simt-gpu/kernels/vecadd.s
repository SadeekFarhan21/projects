; vector add, C[i] = A[i] + B[i], 8 elements, one thread each
; A at 0, B at 8, C at 16
.threads 8
.data 0 0 1 2 3 4 5 6 7
.data 8 0 1 2 3 4 5 6 7

MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx        ; i = global thread index
CONST R1, #0                  ; base of A
CONST R2, #8                  ; base of B
CONST R3, #16                 ; base of C
ADD R4, R1, R0
LDR R4, R4                    ; A[i]
ADD R5, R2, R0
LDR R5, R5                    ; B[i]
ADD R6, R4, R5
ADD R7, R3, R0
STR R7, R6                    ; C[i] = A[i] + B[i]
RET
