# Design of 16-os v0

This document describes how the kernel is put together and why. Addresses come from a 128 MiB QEMU `virt` boot (see `results/boot-log.txt`).

### Privilege model

QEMU starts every hart in machine mode inside OpenSBI (`-bios default`). OpenSBI sets up PMP, delegates most traps and interrupts to supervisor mode, and jumps to the kernel entry at `0x80200000` in S-mode with `a0` = hart id and `a1` = the device tree address. The kernel then talks to firmware only through `ecall` using the SBI extensions TIME (set timer), HSM (hart state), SRST (power off) and DBCN (early console). There is no U-mode in v0.

### Physical memory map

```
0x0000_0000_0010_0000  sifive,test0      write (code<<16)|0x3333 -> QEMU exits with code
0x0000_0000_0c00_0000  PLIC              priority, enable, threshold, claim/complete
0x0000_0000_1000_0000  NS16550A UART     irq 10
0x0000_0000_8000_0000  +-----------------------------+
                       | OpenSBI (reserved in DT)    |  0x80000000..0x80060000
                       | unused gap                  |  0x80060000..0x80200000
0x0000_0000_8020_0000  +-----------------------------+ _kernel_start
                       | .text              R-X      |
0x0000_0000_8020_a000  +-----------------------------+ _text_end
                       | .rodata + ksyms    R--      |
0x0000_0000_8020_f000  +-----------------------------+ _rodata_end
                       | .data, .bss        RW-      |  cpus[], emergency stacks, tables
0x0000_0000_8025_5000  +-----------------------------+ _stacks_start
                       | boot stacks, 8 slots        |  slot = [guard 4K][stack 16K]
0x0000_0000_8027_d000  +-----------------------------+ _kernel_end
                       | page bitmap (4 KiB)         |
0x0000_0000_8027_e000  +-----------------------------+
                       | free page frames   RW-      |  page tables, stacks, heap
                       |            ...              |
0x0000_0000_87e0_0000  | device tree blob (reserved) |
0x0000_0000_8800_0000  +-----------------------------+ end of RAM (from the DT)
```

### Virtual memory map (Sv39, one kernel page table)

```
0x0000_0000_0010_0000 ..                 identity: test device, PLIC, UART
0x0000_0000_8020_0000 .. 0x8800_0000     identity: kernel image and all RAM,
                                         permissions per section as above
0x0000_0020_0000_0000                    TEST_VA, scratch page used by self-tests
0x0000_003f_0000_0000 + slot*0x5000      kernel thread stacks:
                                           +0x0000 guard page (unmapped)
                                           +0x1000 .. +0x5000 stack (16 KiB, RW-)
0x0000_0040_0000_0000                    MAXVA used by v0 (bit 38 stays clear)
```

Every leaf PTE is created with A set, and D set if writable, so the kernel never depends on hardware A/D updates. Kernel mappings carry G. Nothing is mapped U. Page-table pages come from the page allocator. The boot kernel table uses 71 pages.

### Boot sequence

1. `_start` (boot/entry.S) masks `sie`, rejects hart ids at or above `NCPU`, and runs an `amoswap` lottery on a word in `.data`. It lives in `.data` so the BSS clear cannot reset it. Losers park in `wfi`. With SBI HSM only the boot hart arrives anyway, and the other harts show up as "stopped".
2. The winner points `sp` at the top of its boot stack slot, zeroes BSS, clears `s0` and `ra` so backtraces terminate, and calls `kmain(hartid, dtb)`.
3. `kmain` puts `&cpus[hartid]` in `tp`, records the boot stack bounds and emergency stack, clears `sstatus.FS/SUM/SIE`, and installs `stvec`.
4. Early printing goes through the SBI debug console. The device tree is parsed into `struct boot_info`.
5. The UART is initialized and the console switches to it.
6. `pmm_init` puts its bitmap right after the kernel image and frees everything above it, minus reserved ranges and the DTB.
7. `kvm_init` builds the kernel page table and `kvm_enable` writes `satp` and flushes. The PC is identity mapped, so execution continues without a jump.
8. Heap, thread table (the boot flow becomes this hart's idle thread), PLIC plus UART RX interrupt, and the SBI timer come up.
9. The mode from `bootargs` picks the entry of the `main` kernel thread. Interrupts are enabled and the boot hart enters `idle_loop`.

### Trap flow

```
 trap (exception or interrupt, S-mode)
   |
   v
 kernelvec (arch/trapvec.S)
   - sscratch <- t0; is sp < cpu->kstack_lo + frame size?  -> yes: kstack_overflow
   - push 304-byte trapframe on the CURRENT stack:          (emergency stack,
       x1..x31, sepc, sstatus, scause, stval,               report, exit 3)
       frame record {old s0, sepc} so backtraces cross the trap
   - call kernel_trap(tf)
   |
   v
 kernel_trap (core/trap.c)
   - scause bit 63 set: interrupt
       5 timer    -> timer_interrupt (rearm, ticks++, wakeup) then preempt_tick
       9 external -> plic_claim, uart_intr, plic_complete
       1 software -> clear SSIP
   - else exception
       cpu->onfault set -> record cause, sepc = onfault, return (probe helpers)
       otherwise        -> readable report, panic, exit code 3
   |
   v
 back in kernelvec: interrupts off, restore sepc and sstatus, reload registers
 (tp is not restored, it identifies the hart), sret
```

The frame lives on the interrupted thread's own stack. That is what allows preemption: the timer path calls `yield()` inside `kernel_trap`, another thread runs, and when this thread is chosen again it returns out of `kernel_trap` and `sret`s to where it was interrupted.

The fault probes (`probe_read64`, `probe_write64`, `probe_exec`, `probe_illegal`, `probe_ebreak`) store a recovery address in `cpu->onfault` before one risky instruction. This is the same idea as Linux exception tables. The self-tests use them to prove that a page is unmapped or read-only without crashing the kernel.

### Scheduler

States are UNUSED, EMBRYO, RUNNABLE, RUNNING, SLEEPING and ZOMBIE. There is a static table of 32 threads, a FIFO run queue and one idle thread per hart that never enters the queue.

```
 thread_create -> RUNNABLE --sched picks--> RUNNING --yield/preempt--> RUNNABLE
                                              |  \--sleep_on--> SLEEPING --wakeup--> RUNNABLE
                                              \--thread_exit--> ZOMBIE --thread_join--> UNUSED
```

- `sched()` pops the run queue head (or the idle thread) and calls `swtch(&cur->ctx, &next->ctx)` directly. `swtch` saves and restores only `ra`, `sp` and `s0..s11`, because the C caller already saved everything else.
- A new thread starts at `thread_trampoline`, which zeroes `ra` and `s0` (ending the backtrace chain) and jumps to `thread_start`. That function releases `sched_lock`, enables interrupts and calls the thread function.
- Preemption uses one 10 ms tick of quantum (`QUANTUM_TICKS = 1`, `HZ = 100`).
- `idle_loop` checks the run queue with interrupts off and runs `wfi` while they are still off, then enables them. `wfi` wakes on a pending interrupt even when `SIE` is clear, so a wakeup between the check and the `wfi` is not lost.
- `sleep_on(chan, lk)` takes `sched_lock` before releasing `lk`. `wakeup` needs `sched_lock`, so it cannot slip between the sleeper's condition check and its state change. That rules out the lost-wakeup race.

### Locking rules and invariants

- Spinlocks disable interrupts on the local hart for as long as they are held (`push_off`/`pop_off` nest). The outermost `push_off` remembers whether interrupts were on.
- A timer interrupt therefore never preempts code holding a spinlock, and an interrupt handler never spins on a lock its own hart holds.
- `sched()` asserts that `sched_lock` is held and is the only spinlock held (`noff == 1`), that interrupts are off, and that the current thread is no longer RUNNING.
- `intena` belongs to the thread, not the hart. `sched()` saves it in a local variable across `swtch`.
- `sched_lock` is acquired by the thread that switches out and released by the thread that switches in, either in the caller of `sched()` or in `thread_start`.
- Sleep locks may be held across sleeps and only from thread context. Spinlocks must never be held across a sleep (enforced by the `noff == 1` check).
- Lock order: `cons_in.lk` then `sched_lock` then `console`. Separately, `kmalloc` then `pmm`. Nothing that takes `sched_lock` holds `pmm` or `kmalloc`. Stacks are mapped and unmapped outside `sched_lock`.
- Once `panicking` is set, the console lock is bypassed, because the crashing code may hold it.
- The idle thread never sleeps and never exits (both panic).
- `cpu->kstack_lo/hi` always describe the stack the hart is running on. They are updated in `sched()` before `swtch`, while interrupts are off.

### Memory management

- The page allocator is a bitmap with 1 bit per frame (4 KiB of bitmap for 128 MiB). Allocation scans 64-bit words from a rotating hint and uses `ctz`. Freeing checks alignment, range and the bit, so a double free returns `-E_DOUBLEFREE` and is counted. `pmm_alloc` zeroes the page, while `pmm_alloc_nozero` does not.
- The heap has seven size classes of 32 to 2048 bytes. Each class carves whole pages into equal blocks. Every block has a 16-byte header (magic, class, size), so `kfree` needs no size argument and payloads are 16-byte aligned. Double frees and bad pointers are rejected by magic. Requests above 2032 bytes take contiguous pages.
- Kernel stacks are 4 separately allocated frames mapped contiguously at a high VA above an unmapped guard page. The boot stacks use the same layout inside the image.

### Trade-offs chosen and rejected

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Kernel address layout | Identity map of RAM and devices | Higher-half kernel. Nothing needs it before user mode, and identity addresses are easier to debug. It will be revisited with user processes. |
| Page size | 4 KiB everywhere | 2 MiB megapages for free RAM. They would save most of the 71 table pages and TLB entries, but complicate unmap and permission splits. |
| Frame allocator | Bitmap | Free list. It is O(1), but cannot detect double frees without extra metadata. |
| Heap | Size-class slabs plus page runs | First-fit with coalescing. It is more memory efficient but slower and harder to check. |
| Context switch | Direct thread to thread | Per-CPU scheduler thread (two switches per yield, simpler lock handoff). |
| Run queue | One global FIFO under `sched_lock` | Per-CPU queues. They are only needed once more harts run. |
| Trap stack | Current kernel stack | Per-hart trap stack via `sscratch`. That breaks preemption inside traps. |
| Registers saved on trap | All 31 GPRs | Caller-saved only. Saving everything gives a complete register dump in fault reports. |
| Floating point | Kernel built without F/D, `sstatus.FS = Off` | Saving FP state on every trap. Any accidental FP use traps as an illegal instruction instead. |
| Timer | SBI TIME extension | Writing `stimecmp` directly (Sstc is present). SBI is portable to firmware without Sstc. |
| Console | SBI early, then polled UART TX, interrupt RX | Interrupt-driven TX. It is not worth a TX ring buffer in v0. |
| Symbols in panics | Two-pass link embedding a sorted symbol table | Host-side `addr2line` only. On-screen names make CI logs readable. |
| Exit codes | SBI SRST for success, `sifive,test0` for failure | SRST alone, which cannot carry an exit status. |
| Tests | In-kernel suite with QEMU exit codes | Host-side unit tests of kernel files. They would not exercise traps, paging or preemption. |

### Where this differs from xv6-riscv, and why

I read MIT 6.1810's xv6-riscv and the privileged spec for reference. The code here is my own.

- xv6 boots in M-mode with no firmware and configures the timer itself. This kernel runs under OpenSBI and uses SBI calls, which is how Linux and most real boards boot.
- xv6 hard-codes the end of RAM (`PHYSTOP`). This kernel reads memory size, reserved ranges and device addresses from the device tree.
- xv6's frame allocator is a free list filled with junk on free, with no double-free detection. This one is a bitmap and rejects double frees.
- xv6 has no general-purpose heap. This kernel has `kmalloc`/`kfree` with checked frees.
- xv6 switches through a per-CPU scheduler context (two `swtch` calls per switch). This kernel switches directly, and hands off `sched_lock` from the old thread to the new one. The benchmark cost is 190 guest instructions per scheduler switch against 34 per raw `swtch` (`results/bench-summary.txt`).
- xv6's `kernelvec` saves caller-saved registers only. This one saves all of them, adds a frame record for backtraces, and checks for stack overflow before pushing.
- xv6's `printf` has no field widths and no symbolized backtrace. This kernel has both.
- Like xv6, this kernel places kernel stacks at high VAs with guard pages, keeps the hart pointer in `tp` and does not restore `tp` from the trap frame.
