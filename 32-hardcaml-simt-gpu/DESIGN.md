# Design

This is v0 of a small SIMT GPU written in Hardcaml. It follows the overall
shape of tiny-gpu (one core, four threads in lockstep, a tiny ISA, a memory
controller in front of slow memory) but the RTL, the encoding choices around
reset and branching, and the verification setup are my own.

## Block diagram

```
                    host ports (load imem/dmem, thread_count, start, readback)
                                   |
      +----------------------------+---------------------------------------------+
      | gpu (top)                  v                                             |
      |                                                                          |
      |   +------------------+    pc     +------------------+                    |
      |   | instruction mem  |<----------|       core       |                    |
      |   | 256 x 16, async  |---------->|                  |                    |
      |   +------------------+   instr   |  FSM + dispatcher|                    |
      |                                  |  pc, instr reg   |                    |
      |                                  |                  |                    |
      |      thread 0   thread 1   thread 2   thread 3      |                    |
      |     +-------+  +-------+  +-------+  +-------+      |                    |
      |     | R0-12 |  | R0-12 |  | R0-12 |  | R0-12 |  regfiles, NZP,         |
      |     | NZP   |  | NZP   |  | NZP   |  | NZP   |  ALU (add sub mul div)  |
      |     | ALU   |  | ALU   |  | ALU   |  | ALU   |                         |
      |     | LSU   |  | LSU   |  | LSU   |  | LSU   |                         |
      |     +---+---+  +---+---+  +---+---+  +---+---+      |                    |
      |         | pending/grant/response (per thread)       |                    |
      |   +-----v------------------------------------------v-----+              |
      |   | memory controller                                    |              |
      |   |   fixed priority arbiter (1 grant per cycle)         |              |
      |   |        |                                             |              |
      |   |        v                                             |              |
      |   |   request queue (FIFO, depth Q) {tid, we, addr, data}|              |
      |   |        |  pop when DRAM idle (every I cycles)        |              |
      |   |        v                                             |              |
      |   |   DRAM pipeline, L stages ----> data mem 256 x 8 ----+--> response  |
      |   |                                  (read/write at exit) (tid, rdata)  |
      |   +------------------------------------------------------+              |
      +--------------------------------------------------------------------------+
```

## Core state machine

```
            start (thread_count > 0)
   Idle ------------------------------> Launch --> Fetch --> Decode --> Execute
    ^  \  start (thread_count = 0)        ^                               |  |  |
    |   +------------------> Done <-------+--- RET, last block ----------+  |  |
    |                         |           |                                  |  |
    +---------- (start) ------+           +--- RET, more blocks (block++) ---+  |
                                                                                |
                         Fetch <-- Writeback <-- Wait (all LSUs done) <-- LDR/STR
                                       ^
                                       +------------- ALU op, CMP, BR, NOP
```

- Launch: pc = 0, clear R0..R12, NZP = Z, compute the active mask for the
  block (`4 * block + t < thread_count`).
- Fetch: latch `imem[pc]` into the instruction register.
- Decode: read rs and rt for each thread into operand registers. R13..R15
  read the block index, block size (4) and thread index.
- Execute: each thread's ALU result and compare flags are latched. LDR/STR mark
  every active LSU pending and go to Wait. RET either launches the next block
  or finishes.
- Wait: stall until every active LSU has its response.
- Writeback: write rd (ALU or load data), NZP for CMP, update pc (branch target
  if thread 0's condition holds, else pc + 1), retire the instruction.

An ALU instruction takes 4 cycles (Fetch, Decode, Execute, Writeback). A memory
instruction takes 4 cycles plus its Wait time.

## ISA

16 bit instructions, opcode in bits 15..12. 13 general registers per thread,
three read only index registers, one NZP register per thread.

| Opcode | Mnemonic | Semantics |
|---|---|---|
| 0000 | NOP | nothing |
| 0001 | BRnzp nzp, target | pc = target if (NZP of thread 0) & nzp is nonzero |
| 0010 | CMP rs, rt | NZP = unsigned compare of rs and rt |
| 0011 | ADD rd, rs, rt | rd = rs + rt mod 256 |
| 0100 | SUB rd, rs, rt | rd = rs - rt mod 256 |
| 0101 | MUL rd, rs, rt | rd = low 8 bits of rs * rt |
| 0110 | DIV rd, rs, rt | rd = rs / rt unsigned, x / 0 = 255 |
| 0111 | LDR rd, rs | rd = mem[rs] |
| 1000 | STR rs, rt | mem[rs] = rt |
| 1001 | CONST rd, imm8 | rd = imm8 |
| 1111 | RET | end of block |

## Key data structures

### Per thread state (in `src/core.ml`)

| Signal | Width | Purpose |
|---|---|---|
| `regs.(t).(k)` | 8 | R0..R12, zeroed at Launch |
| `nzp.(t)` | 3 | N, Z, P flags, set to Z at Launch |
| `active.(t)` | 1 | thread exists in this block |
| `rs_val.(t)`, `rt_val.(t)` | 8 | operands latched in Decode, also the LSU address and store data |
| `result.(t)`, `cmp_flags.(t)` | 8, 3 | latched in Execute |
| `lsu_state.(t)` | 2 | Idle, Pending, Inflight, Done |
| `lsu_rdata.(t)` | 8 | load data |

### Request queue entry (in `src/mem_ctrl.ml`)

`{tid : 2, we : 1, addr : 8, wdata : 8}` packed into 19 bits. The queue is a
register FIFO with read and write pointers and an occupancy counter, so `full`
and `empty` are single comparisons.

### DRAM pipeline

`L` stages of `{valid, entry}` registers. A request is performed on the data
memory when it leaves the last stage, and the response carries `tid` back to
the owning LSU. A busy counter blocks acceptance for `I - 1` cycles after each
accepted request.

### Host interface (in `src/gpu.ml`)

`host_imem_*` and `host_dmem_*` write ports load memories while idle,
`host_dmem_raddr / host_dmem_rdata` is an asynchronous readback port, and six
performance counters (cycles, stall cycles, instructions, memory requests,
divergent branches, queue full cycles) are exposed as outputs.

## Invariants

1. All active threads of a block execute the same instruction in the same
   cycle. There is exactly one pc.
2. Thread 0 of a running block is always active, so it can always decide a
   branch.
3. The arbiter never pushes into a full queue and DRAM never pops an empty
   one (grant is gated by `full`, accept by `empty`).
4. Requests reach the data memory in queue order, and the arbiter enqueues in
   thread order, so the stores of one STR land in thread order and the highest
   active thread wins a collision.
5. The core leaves Wait only when every active LSU is Done, so a load's data is
   in `lsu_rdata` before Writeback reads it, and no request is ever in flight
   while the core fetches the next instruction.
6. Writes to R13..R15 are dropped, and inactive threads never write registers
   or issue memory requests.
7. The hardware counters for retired instructions, memory requests and
   divergent branches equal the interpreter's counts for the same kernel. The
   tests check this on every run, which catches control bugs that happen not to
   change memory.

## Trade-offs

### Chosen

- Multi cycle FSM, no pipeline. It matches the tiny-gpu reference and keeps
  every instruction's timing easy to reason about, at the cost of 4 cycles per
  instruction at best.
- Memory ops execute at the DRAM exit, not at acceptance. This gives a simple
  in order memory model that the interpreter can mirror exactly.
- Instruction memory is on chip with an asynchronous read, so Fetch is a single
  cycle and only data accesses see the DRAM model. tiny-gpu also routes
  instruction fetch through a memory controller; I cut that to keep the stall
  metric about data memory only.
- Branches follow thread 0 and count disagreement instead of stalling or
  trapping. This keeps v0 well defined on any program (the random kernels do
  diverge) and gives a number to motivate the reconvergence stack milestone.
- NZP resets to Z rather than 0, so `BRnzp` is an unconditional branch even
  before the first CMP. The assembler takes branch flags literally, so plain
  `BR` is never taken.
- DRAM latency, issue interval and queue depth are elaboration time parameters.
  Each value is a different circuit, which is how real configurable IP works
  and keeps the RTL free of runtime configuration registers.
- The dispatcher lives inside the core FSM. With a single core there is nothing
  to dispatch between, so a separate module would only add wiring.

### Rejected

- A separate issue slot per memory channel (tiny-gpu has several channels). One
  channel with a request queue and an issue interval shows queueing and back
  pressure with less logic; more channels come with the multi core milestone.
- Using Hardcaml's built in `Fifo`. It uses a RAM with a registered read and
  would add a cycle to the head of the queue; the hand written register FIFO is
  4 to 16 entries and exposes the head combinationally.
- A shared register file with a thread index in the address. Separate register
  arrays per thread are what a real SIMT lane looks like and let every thread
  write back in the same cycle.
- Trapping on divergence. It would make random program generation much harder
  (every branch condition would need to be uniform) and hide the problem
  instead of measuring it.
