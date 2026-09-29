---
layout: post
title: a RISC-V kernel that tests itself on every boot
tags:
  - operating-systems
  - risc-v
  - c
description: >-
  A small RISC-V kernel with Sv39 paging, traps and preemptive threads, plus a
  self-test suite that runs every time it boots.
date: 2026-09-29 02:17:07
---


I wrote 16-os, a small kernel for 64-bit RISC-V, in freestanding C and assembly. It boots on QEMU's `virt` board under OpenSBI, runs in supervisor mode the way Linux does, reads the machine's layout from the device tree, turns on Sv39 paging with per-section permissions and guard pages under every stack, handles traps and interrupts, and schedules preemptive kernel threads. The whole thing is **3,521 lines** of C, assembly, headers and linker script, and **0x96ce bytes** (38,606) of machine code in `.text`.

The number I trust most is not a benchmark. Every boot can run a built-in suite of **22 self-tests** that powers QEMU off with a pass or fail exit code, and four more boots crash on purpose and must print a readable report and exit with code 3. That makes `make test` a headless command, and on 2026-09-26 an independent clean rebuild (`make clean`, `make -j2`, `make test`) passed every self-test and all four crash reports.

Under QEMU's deterministic `-icount` mode a raw context switch costs **34** guest instructions, a switch through the scheduler **190**, a trap round trip **140** and a call into firmware **291**. These are QEMU instruction counts, not cycles on real hardware. The wall-clock timings I also took ran inside QEMU on a Mac with a load average of 129 to 151 from other jobs, so they serve only as a cross-check.

Code is in `projects/16-riscv-sv39-kernel`. v0 is kernel only, with no user mode yet.

*Reading note.* The argument is that the hard part of a small kernel is not paging or the trap vector, which worked the first time they were switched on, but making every claim checkable and then finding the races that only show up when the same boot runs thirty or forty times. Skip to [problems](#problems) for the bugs.

## table of contents

- [what I wanted to build](#what-i-wanted-to-build)
- [theory](#theory)
- [architecture](#architecture)
- [implementation](#implementation)
- [problems](#problems)
- [experiments](#experiments)
- [results](#results)
- [what I would change](#what-i-would-change)
- [reproducibility](#reproducibility)

## what I wanted to build

I wanted a kernel small enough to read in an evening that still does the things a real one has to do. The goals for v0 were these.

- Boot through the standard firmware interface (SBI) rather than owning machine mode, the way Linux boots on real boards.
- Hard-code nothing about the machine. RAM size, device addresses and the timer frequency come from the device tree.
- Turn on paging with real permissions and an unmapped guard page below every stack.
- Turn every unexpected trap into a readable report with a symbolized backtrace.
- Run kernel threads that preempt each other, with spinlocks, sleep locks and wakeups that are never lost.
- Check every one of those claims with a test that runs headless and fails loudly.

The last goal shaped the rest. A kernel that writes to its own text segment, expects a store page fault, recovers and reports the result through QEMU's exit status proves the permission is enforced, which a boot banner does not. I read MIT's xv6-riscv and the RISC-V privileged specification for reference. The code is my own, and I note where it differs from xv6.

## theory

### three privilege levels and a firmware interface

RISC-V has machine mode (M), supervisor mode (S) and user mode (U). On QEMU's `virt` board with `-bios default`, every hart starts in M-mode inside OpenSBI. OpenSBI configures physical memory protection, delegates most traps and interrupts down to S-mode and then jumps to the kernel at `0x80200000` in S-mode, with the hart id in `a0` and the physical address of the device tree in `a1`.

From then on the kernel asks firmware for services with `ecall`, the same instruction a user program would use to ask a kernel. The interface is the Supervisor Binary Interface, and I use four of its extensions. TIME arms the next timer interrupt, HSM reports the state of other harts, SRST powers the machine off, and DBCN gives a console before any UART driver exists.

### what a trap does in hardware

When an exception or interrupt is taken in S-mode, the hardware does very little. It saves the PC in `sepc`, writes the cause to `scause` (the top bit says interrupt or exception, and the low bits give the code), puts the faulting address or the faulting instruction's bits in `stval`, copies the interrupt-enable bit `sstatus.SIE` into `SPIE` and clears `SIE`, and jumps to the address in `stvec`. Every general-purpose register still holds what the interrupted code left in it, so the handler's first instruction runs with no free scratch register. Saving registers is software's job (the `sscratch` CSR exists to make room for the first one), and `sret` undoes the CSR part.

### Sv39 in one diagram

Sv39 translates a 39-bit virtual address through three levels of page tables. Each table is one 4 KiB page of 512 eight-byte entries, and each level is indexed by 9 bits of the address.

```
 38        30 29        21 20        12 11            0
+------------+------------+------------+---------------+
|  VPN[2]    |  VPN[1]    |  VPN[0]    |  page offset  |
+------------+------------+------------+---------------+
      |            |            |
      v            v            v
   L2 table --> L1 table --> L0 table --> leaf PTE: PPN | D A G U X W R V
```

An entry with any of R, W or X set is a leaf. One with only V set points to the next level. The A (accessed) and D (dirty) bits have a subtle rule. Without the Svadu extension enabled, hardware that finds A clear on an access, or D clear on a write, raises a page fault instead of setting the bit. QEMU's hart advertises Svadu, but I did not want correctness to depend on it, so every leaf is created with A set, and D set if it is writable.

### what a context switch has to save

A switch between kernel threads is an ordinary function call. The calling convention already makes the caller spill every caller-saved register it still needs, so the switch only has to save the callee-saved ones, which are `ra`, `sp` and `s0` to `s11`. That is 14 registers stored into the old thread's context and 14 loaded from the new one. The trick is that `ra` is loaded from the new context, so `ret` returns into a different thread.

### locking with interrupts

On one hart the danger with a spinlock is an interrupt handler that wants a lock its own hart holds, which spins forever. The standard rule, which xv6 also uses, is that holding any spinlock disables interrupts locally. The first acquire remembers whether they were on, and the last release restores that.

The classic scheduling bug is the lost wakeup. A thread checks a condition, finds it false, and is about to go to sleep. Before its state changes, the condition becomes true and the wakeup fires, finds nobody sleeping and does nothing. Then the thread goes to sleep, forever. The fix is to make the check and the state change atomic with respect to `wakeup`, which in practice means both must hold the same lock.

## architecture

The kernel is one image linked at `0x80200000`. Text, rodata and data each start on a page boundary so they can get different permissions. After them come the boot stack slots, each a guard page followed by a 16 KiB stack. This is the physical layout from a 128 MiB boot, with addresses taken from the boot log.

```
0x0000_0000_0010_0000  sifive,test0      write (code<<16)|0x3333 -> QEMU exits with code
0x0000_0000_0c00_0000  PLIC              interrupt controller
0x0000_0000_1000_0000  NS16550A UART     irq 10
0x0000_0000_8000_0000  +-----------------------------+
                       | OpenSBI (reserved in DT)    |  0x80000000..0x80060000
                       | unused gap                  |  0x80060000..0x80200000
0x0000_0000_8020_0000  +-----------------------------+ _kernel_start
                       | .text              R-X      |
0x0000_0000_8020_a000  +-----------------------------+
                       | .rodata + ksyms    R--      |
0x0000_0000_8020_f000  +-----------------------------+
                       | .data, .bss        RW-      |
0x0000_0000_8025_5000  +-----------------------------+
                       | boot stacks, 8 slots        |  slot = [guard 4K][stack 16K]
0x0000_0000_8027_d000  +-----------------------------+ _kernel_end
                       | page bitmap (4 KiB)         |
                       | free page frames   RW-      |  page tables, stacks, heap
0x0000_0000_87e0_0000  | device tree blob (reserved) |
0x0000_0000_8800_0000  +-----------------------------+ end of RAM (from the DT)
```

The kernel page table identity-maps all of RAM and the three devices it uses, each region with its own permissions. Thread stacks are the exception. Each lives at its own high virtual address starting at `0x0000_003f_0000_0000`, as four separately allocated frames mapped contiguously above an unmapped guard page. The boot log shows the table needed **71 pages**, which is mostly the cost of using 4 KiB leaves for 128 MiB of identity-mapped RAM.

Traps follow one path, and the important design choice is where the frame goes.

```
 trap (exception or interrupt, S-mode)
   |
   v
 kernelvec (arch/trapvec.S)
   - is sp within one frame of the stack bottom?  ---- yes --> emergency stack,
   - push a 304-byte frame on the CURRENT stack:                report, exit 3
       x1..x31, sepc, sstatus, scause, stval,
       and a fake frame record so backtraces cross the trap
   - call kernel_trap(tf)
   |
   v
 kernel_trap (core/trap.c)
   - interrupt 5, timer     -> rearm, ticks++, wakeup, then maybe yield()
   - interrupt 9, external  -> PLIC claim, UART receive, complete
   - exception with a probe armed -> resume at the recovery address
   - any other exception    -> readable report, panic, exit code 3
   |
   v
 back in kernelvec: interrupts off, restore sepc and sstatus and registers, sret
```

The frame lives on the interrupted thread's own stack, not a per-hart trap stack, and that is what makes preemption simple. The timer path calls `yield()` inside `kernel_trap`, other threads run on their own stacks, and when this thread is picked again it returns out of `kernel_trap` and `sret`s to where it was interrupted. A per-hart trap stack would be overwritten by the next trap.

The scheduler is a static table of 32 threads, a FIFO run queue under one spinlock, and one idle thread per hart made from the boot flow. Its states are these.

```
 thread_create -> RUNNABLE --sched picks--> RUNNING --yield/preempt--> RUNNABLE
                                              |  \--sleep_on--> SLEEPING --wakeup--> RUNNABLE
                                              \--thread_exit--> ZOMBIE --thread_join--> UNUSED
```

xv6 switches from a thread to a per-CPU scheduler context and from there to the next thread, which is two `swtch` calls per switch. I switch directly from the old thread to the new one. That halves the switches, and the price is a lock handoff that I describe below.

The kernel command line, passed through the device tree's `bootargs`, picks the first thread's job. It can be the demo with an interactive UART monitor, the self-tests, the benchmarks, or one of four deliberate crashes.

## implementation

### the first instructions

OpenSBI jumps to `_start` with interrupts off and the MMU off. The entry code rejects hart ids beyond its tables, runs a lottery so exactly one hart continues, points `sp` at that hart's boot stack, zeroes BSS and calls `kmain`.

```asm
    la t0, boot_lottery
    li t1, 1
    amoswap.w.aq t1, t1, (t0)
    bnez t1, park              /* someone else is the boot hart */
```

The lottery word lives in `.data`, because the winner's BSS clear would reset a BSS word and let a slow second hart win again. With SBI HSM the other hart never arrives (the boot log reports it as "stopped"), so the lottery only matters for older firmware. The entry code also clears `s0` and `ra`, so every backtrace ends at a zero frame pointer.

### finding the machine

The device tree parser walks the flattened blob once. Properties always precede child nodes, so a node is finished when its first child or end token appears. One pass collects memory, the `reserved-memory` children (OpenSBI reserves itself there), the UART, the PLIC, the `sifive,test0` exit device, the timebase and `bootargs`. On the 128 MiB machine it found **32,768 pages** and left **32,127** free after the kernel, the bitmap, OpenSBI's ranges and the blob itself.

### a page allocator that notices double frees

The physical allocator is a bitmap with one bit per 4 KiB frame, which is 4 KiB of bitmap for 128 MiB. Allocation scans 64-bit words from a rotating hint and uses count-trailing-zeros to find a free bit, so the common case touches one word.

I picked a bitmap over a free list because freeing becomes a bit test, so a double free, an unaligned free or a free outside RAM returns an error instead of corrupting a list. xv6's free list cannot detect a double free at all. One self-test allocates every free page (32,050 at that point in the suite), writes a tag into each, chains them together, checks every tag and frees them all.

### paging, and proving the permissions

Because RAM is identity mapped, the PC after the `satp` write is still valid, so paging turns on without a jump. The mapping function sets A and D up front.

```c
/* Set A (and D for writable pages) up front: without Svadu the
 * hardware raises a page fault instead of setting them. */
u64 ad = PTE_A | ((perm & PTE_W) ? PTE_D : 0);
*pte = PA2PTE(pa + off) | perm | ad | PTE_V;
```

A permission table is only a claim until something tries to violate it. The self-tests write to text and to rodata and expect a store page fault, execute a `ret` placed in `.data` and expect an instruction page fault, and read one byte below every stack and expect a load page fault. Doing that without killing the kernel needs fault probes, which are small assembly helpers that store a recovery address in the per-hart structure before one risky instruction.

```asm
probe_read64:              /* int probe_read64(u64 addr, u64 *out) */
    la t0, 1f
    sd t0, CPU_ONFAULT(tp)
    ld t1, 0(a0)           /* the one instruction allowed to fault */
    sd zero, CPU_ONFAULT(tp)
    sd t1, 0(a1)
    li a0, 0
    ret
1:  li a0, -1
    ret
```

If the load faults, `kernel_trap` sees `onfault` set, records the cause, points `sepc` at the recovery label and returns, and the probe returns -1. It is the idea behind Linux's exception tables, reduced to one instruction per helper.

### a heap with checked frees

`kmalloc` has seven power-of-two size classes from 32 to 2,048 bytes. Each class cuts whole pages into equal blocks, and every block carries a 16-byte header with a magic value, the class and the size. So `kfree` needs no size, payloads stay 16-byte aligned, and a double free or foreign pointer fails the magic check. Larger requests take contiguous pages. A stress test runs 5,000 random operations of 1 to 3,000 bytes and checks every block's fill pattern before freeing it.

### the trap vector checks for overflow before it pushes

The obvious trap vector subtracts the frame size from `sp` and starts storing. If the trap was a stack overflow into the guard page, that first store faults again, which traps again, and the kernel loops forever without printing anything. So the vector checks first.

```asm
kernelvec:
    csrw sscratch, t0
    ld t0, CPU_KSTACK_LO(tp)
    addi t0, t0, TF_SIZE
    bltu sp, t0, kstack_overflow   /* pushing would fault again */
    csrr t0, sscratch
    addi sp, sp, -TF_SIZE
    ...
```

`tp` points at the hart's `struct cpu`, whose `kstack_lo` the scheduler updates on every switch. On overflow the vector moves to a per-hart emergency stack and reports. In the deliberate crash the report said `sp` was 16 bytes below the stack bottom, named the faulting instruction as a store in `recurse+0xc`, and folded the recursion into one line, "same frame repeated 509 more times", before ending at `crash_thread` and `thread_start`.

The vector saves all 31 registers, where xv6 saves only the caller-saved ones, which gives every fault report a complete register dump. It also writes a fake frame record holding the interrupted `s0` and `sepc`, so a backtrace walks through the trap into the faulting code.

### symbolized backtraces from a two-pass link

Raw addresses are useless in a CI log, so the kernel carries its own symbol table. The catch is that the table depends on the final addresses and adding it could move them. So the Makefile links twice. The first link has an empty table, its text symbols become a sorted C array, and the second link puts that array in `.rodata`. `.rodata` follows `.text`, so no code address can move, and every build proves it.

```make
@$(NM) -n $(BUILD)/kernel.pass1.elf | grep -i ' t ' > $(BUILD)/text1.txt
@$(NM) -n $@ | grep -i ' t ' > $(BUILD)/text2.txt
@cmp -s $(BUILD)/text1.txt $(BUILD)/text2.txt || (echo "error: text symbols moved between link passes"; exit 1)
```

Backtraces walk the frame-pointer chain (`-fno-omit-frame-pointer`), where each function saves `ra` at `fp - 8` and the caller's `fp` at `fp - 16`. The lookup uses `ra - 1`, because a call to a `noreturn` function can be the last instruction of its function, which would put `ra` in the next one. That detail comes back in problem 4.

### the scheduler and the lock handoff

`sched()` pops the head of the run queue, or the idle thread if it is empty, and switches straight to it. Before switching it asserts the invariants that make this safe, which are that `sched_lock` is held and is the only spinlock held, that interrupts are off, and that the current thread is no longer `RUNNING`.

```c
    int intena = c->intena; /* belongs to this thread, not to the hart */
    swtch(&cur->ctx, &next->ctx);
    mycpu()->intena = intena;
```

The "were interrupts on before the first lock" flag lives in the per-hart structure but describes the thread that took the lock, so `sched` carries it across the switch in a local variable. The lock handoff is the price of switching directly. The old thread acquires `sched_lock`, and the new thread releases it, either in the caller of its own earlier `sched` or, if it has never run, in `thread_start`.

```c
void thread_start(void) {
    spin_unlock(&sched_lock); /* acquired by the thread that switched to us */
    intr_on();
    struct thread *t = thread_current();
    t->fn(t->arg);
    thread_exit(0);
}
```

`sleep_on` prevents the lost wakeup by taking `sched_lock` before it releases the caller's lock. `wakeup` needs `sched_lock` too, so it cannot run between the sleeper's check of its condition and its state becoming `SLEEPING`.

```c
void sleep_on(void *chan, struct spinlock *lk) {
    ...
    if (lk != &sched_lock) {
        spin_lock(&sched_lock);
        spin_unlock(lk);
    }
    t->chan = chan;
    t->state = T_SLEEPING;
    sched();
    ...
```

The idle loop has its own race. If an interrupt made a thread runnable between its run-queue check and its `wfi`, the hart would sleep until the next tick. So it checks with interrupts off and runs `wfi` while they are still off, which works because `wfi` wakes on a pending interrupt even when `SIE` is clear. Preemption is one 10 ms tick at 100 Hz, with deadlines on a fixed grid so ticks do not drift.

### exit codes that QEMU can see

SBI's system reset can power off but cannot carry an exit code. QEMU's `sifive,test0` device can, since writing `(code << 16) | 0x3333` makes QEMU exit with that code. So a pass powers off through SBI and a failure writes to the test device. The harness boots QEMU headless under a timeout, types `ping` into the UART to exercise receive interrupts, and checks the exit code and the report text. The kernel is also built without F and D and runs with `sstatus.FS` off, so a stray floating-point instruction traps instead of corrupting state that traps never save.

## problems

I expected the usual trouble when paging came on, and it did not happen. The first boot with `satp` written ran the whole demo. I think three choices prevented the usual failures. Every section boundary is page aligned in the linker script, A and D are set on every leaf, and the trap frame offsets are shared between C and assembly through one header that C checks with `_Static_assert`. The bugs I did hit were subtler, and three of the four were races.

### 1. a flaky timer test, and a fix that fixed nothing

The timer self-test busy-waits 100 ms and checks two things, that about 10 ticks happened, and that the global `ticks` counter and this hart's timer-interrupt count advanced by the same amount. It failed once with `check failed: mycpu()->ntimer - n0 == dt`.

My first theory was the compiler caching a counter that the interrupt handler updates, so I made the counters `volatile`, and the test passed. Later I rebuilt without `volatile` to reproduce the failure, and it still passed. The disassembly showed the counter was reloaded after an opaque `kprintf` call anyway, so `volatile` had changed nothing.

The real bug was in the test. It read `ticks` and the per-hart count as two separate loads with interrupts enabled, and at the end of the test a slow `kprintf` sat between the two loads. A tick landing in that gap made the two snapshots disagree by one. The fix takes both values in one snapshot with interrupts off.

```c
static void tick_snapshot(u64 *t, u64 *n) {
    push_off();
    *t = ticks;
    *n = mycpu()->ntimer;
    pop_off();
}
```

To be sure this time, I booted the `only=ticks` test 30 times with each version under plain TCG. The version before the fix failed **4 of 30** boots, and the fixed version **0 of 30**. The failing boots saw 3 to 9 ticks instead of 10, so the host (load average 299) was starving QEMU's vCPU thread, which widens the window. A fix for a failure you cannot reproduce is a guess, and my first guess was wrong.

### 2. a store the compiler was right to delete

`volatile` was still needed, just somewhere else. The timer latency benchmark sets a `recording` flag, spins until 200 ticks pass, and clears the flag. The interrupt handler records a latency sample whenever the flag is set.

```c
    timer_lat.recording = true;
    u64 end = ticks + 200;
    while (ticks < end)
        ;
    timer_lat.recording = false;
```

The loop reads only the `volatile` `ticks`. To the compiler, `recording = true` is followed by `recording = false` with nothing in between that could observe it, so it deleted the first store. The busy variant recorded no samples and printed this.

```
BENCH timer_latency_busy     n=0 min=0 p50=0 p99=0 max=0 mean=18446744073709551615 ns (resolution 100 ns)
```

That mean is 2^64 - 1, and it is a lesson of its own. RISC-V integer division by zero does not trap. It returns all ones, so `sum / n` with `n = 0` quietly produced the largest `u64`. The idle variant worked because an opaque call to `thread_sleep_ticks` sat between its stores. Marking the latency structure `volatile` and guarding the division fixed it, and a rebuild with both fixes removed reproduces exactly this output.

### 3. a failing boot that exited with 0

After the kernel printed a perfect page-fault report, `make test` once failed because QEMU had exited with 0 instead of 3. The report was right and the verdict was wrong.

My exit path wrote the failure value to the test device and then, "just in case", fell through to an SBI shutdown call. QEMU acts on the test device write asynchronously, so the vCPU kept running for a moment. It reached the SBI call, and OpenSBI's poweroff on this board writes the *pass* value to the same device. Two exit requests were in flight, and whichever QEMU processed decided the exit code.

The fix is to never fall through. After the failure write the hart parks.

```c
    if (bootinfo.test_base) {
        mmio_write32(bootinfo.test_base, code == 0 ? 0x5555 : (((u32)code << 16) | 0x3333));
        /* QEMU acts on this write asynchronously. Do not fall through to the
         * SRST call below ... */
        for (;;) wfi();
    }
```

I measured it the same way as the first bug, 20 boots each of `crash=pagefault` and `crash=panic`, expecting exit code 3. Before the fix **10 of 40** boots exited with the wrong code, and after it **0 of 40**. In a suite that boots each crash once, a one-in-four failure looks like an occasional red build that passes on rerun, which is exactly what people learn to ignore.

<figure data-figure="chart:projects/os-from-scratch/os-from-scratch-bug-repro"></figure>

### 4. a function the thread never called

Every backtrace from a kernel thread ended with one bogus frame. With the fix reverted, the deliberate panic's backtrace reads like this.

```
  #3  0x0000000080206204  crash_thread+0x48
  #4  0x0000000080200d94  thread_start+0x2c
  #5  0x0000000080200d68  thread_init+0x6c
```

`thread_init` runs once at boot and never calls a thread's function. A fresh thread's context had `ra` pointing at `thread_start`, so the first `swtch` "returned" there, and `thread_start`'s prologue saved that `ra` as its own return address. The backtrace looked up `ra - 1`, the last byte of whatever precedes `thread_start` in memory, which was `thread_init`.

The fix is a three-instruction trampoline. A new context's `ra` points at it, and it clears `ra` and `s0` before jumping to `thread_start`, so the frame record `thread_start` saves ends the chain.

```asm
thread_trampoline:
    li ra, 0
    li s0, 0
    j thread_start
```

### 5. smaller things that cost time

- `-std=c17` turns off the `asm` keyword, so the kernel builds as `gnu17`.
- Clang's `-Winfinite-recursion` (an error under `-Werror`) rejected the deliberate stack-overflow function until the recursion depended on a `volatile` flag.
- zsh does not split an unquoted `$VAR` into words, so my first batch of reproduction runs silently started nothing.
- `uart_init` resets the receive FIFO, which throws away any keystrokes QEMU delivered before the driver ran. The test harness waits for the self-test banner before it types.

### 6. the host was the noisiest component

Fifteen other builds shared this machine and the load average sat in the hundreds. Under plain TCG, QEMU's virtual clock follows the host clock, so a starved vCPU thread misses ticks, which are dropped rather than replayed. I loosened the tick test's lower bound and added a second pass of the suite under `-icount`, which does not depend on the host, and that is why this post's headline numbers are instruction counts.

## experiments

All measurements run inside QEMU and none is a hardware number. Under plain TCG a timing includes translation, emulator bookkeeping and host scheduling, on a host heavily loaded by other jobs. Under `-icount shift=0,sleep=off`, virtual time advances one nanosecond per guest instruction, so an icount "nanosecond" is an instruction count. It is deterministic and ignores host load, but it treats every instruction as equal, which a real core does not. Icount numbers are the primary result here, and TCG timings a cross-check. Every benchmark loops and divides, because `rdtime` ticks at 10 MHz, a resolution of 100 ns.

### what a switch costs

A raw `swtch` between two contexts costs **34** guest instructions in each direction. Most of that is the 28 loads and stores, and the rest is the return and the benchmark loop around it. A full `yield` from one thread to another through the scheduler costs **190**. The other 156 instructions are `sched_lock` with its interrupt masking and nesting counter, the run queue, the assertions in `sched` and bookkeeping. So the register switch is a small part of the cost and the locking around it is most of it. I did not measure xv6 on the same setup, so I cannot say how much the direct switch saves in practice.

### traps against firmware calls

An `ebreak` trap into my vector and back costs **140** instructions. That covers the overflow check, 31 register stores, 4 CSR reads, dispatch, probe recovery, the restore and `sret`. An SBI `ecall` round trip costs **291**, about twice as much, since OpenSBI saves and dispatches in its own M-mode handler. Under TCG the gap nearly closes, 1,158 to 1,225 ns against 1,520 to 1,583 ns over three runs. My reading is that on this emulator most of a trap's cost is QEMU leaving translated code, which both paths pay.

### the allocators

Allocating a page without zeroing costs **105** instructions and freeing one costs **103**. Allocating with zeroing costs **2,184**, so clearing 4 KiB dominates everything else in the allocator by a factor of about 20. A `kmalloc(64)` and `kfree` pair costs **222**. Under TCG the zeroing allocation took 1,622 to 1,645 ns, *less* than its instruction count, because QEMU runs a tight store loop faster than one instruction per nanosecond.

<figure data-figure="chart:projects/os-from-scratch/os-from-scratch-icount-costs"></figure>

### timer interrupt latency

The timer handler records the gap between the programmed deadline and the moment the C handler reads `rdtime`. I measured it with the hart idle in `wfi` and with a thread spinning, 200 ticks each.

Under icount the median was **100 ns** idle and busy, with a mean of 85 and 84 ns. The timebase resolution is 100 ns, so the kernel's path from deadline to handler is within one tick. Under plain TCG the median was between **2.333 ms and 2.614 ms** in all six runs, idle and busy, at load averages between 129 and 151. That is a steady offset, not scattered noise, and it appears even when idle. My guess, which I have not verified, is how QEMU arms its host timer on macOS. The icount numbers do show the offset is not in the kernel.

<figure data-figure="chart:projects/os-from-scratch/os-from-scratch-timer-latency"></figure>

### is icount actually reproducible

Two icount benchmark runs gave identical numbers in every column, and a raw diff of their benchmark lines was empty. The icount timer test saw exactly 10 ticks in 100 ms. A benchmark that gives the same answer twice can detect a one-instruction change on the switch path, which no wall-clock measurement on a loaded host could.

### why the lock matters

Four threads each increment a shared counter, 80,000 increments in total, with a short delay between each read and each write so that preemption has a window to land in. Under the spinlock the count was exactly **80,000** in both passes. The unlocked control is more interesting. Under icount it lost **3,605** updates and ended at 76,395. Under plain TCG it lost **0**.

So the control is informational, not an assertion. One plausible reason for the TCG result is that a tick arriving while the lock is held is deferred until `spin_unlock` re-enables interrupts, just before the unlocked read, which biases preemption toward the edge of the racy window. I have not measured it.

## results

- `make test` passes. That is 22 self-tests under plain TCG, including the UART receive test with typed input, 21 under icount (the receive test needs typed input and runs only in the TCG pass), and four crash scenarios that each print the expected report and exit with code 3.
- On 2026-09-26 an independent clean rebuild (`make clean`, `make -j2`, `make test`) passed every self-test and all four crash reports. That rerun checked correctness, not the benchmarks.
- Under icount the kernel reaches its first thread **16,061 µs** of virtual time after reset. That includes OpenSBI's own startup.
- The kernel page table uses **71 pages**, and 32,127 of 32,768 pages are free after boot.
- In the demo, four workers are preempted between 4 and 7 times each, their shared tally ends at the expected 16, and the boot takes 48 context switches before it powers off.
- Under icount a raw `swtch` costs **34** guest instructions, a scheduler switch **190**, a trap **140**, an SBI call **291**, and timer latency is within one 100 ns tick.
- The interactive monitor works over UART receive interrupts, with 10 interrupts for the scripted session.

## what I would change

### go higher-half before user mode

The identity map made v0 easy to debug, since every address in a fault report is also physical. User programs want the low half of the address space, so I would move the kernel to the top before writing any user-mode code.

### make the scheduler SMP-safe in the one place it is not

`thread_join` frees a zombie's stack as soon as it sees `ZOMBIE`. On one hart the zombie has already left its stack, but with two it could still be finishing `sched` on it. It needs an "on CPU" flag, and each hart should get its own run queue.

### write stimecmp directly

The hart advertises Sstc, which lets S-mode write its timer compare register without an SBI call. That would replace an SBI round trip on every tick (the probe call I measured costs 291 instructions) with one CSR write, and I would like to know whether it also removes the TCG timer offset.

### tidy the memory manager and the console

The 1.6 MiB between OpenSBI and the kernel image is never used, slab pages are never returned, and the contiguous allocator is a first-fit scan from zero. `kfree` on a wild pointer can fault on the header read, which a fault probe would turn into an error. UART output is polled, so a long `kprintf` keeps interrupts off while it writes.

### build repeated-boot testing in from the start

The exit-code race and the tick race were both found by booting the same kernel 30 or 40 times. A single passing `make test` did not catch either, and both would have shown up as a flaky red build. Running each boot many times, preferably on a loaded host, should be a standard test mode, not a script I wrote after the fact.

## reproducibility

The toolchain is Homebrew LLVM and QEMU on macOS. Apple's clang has no RISC-V backend. This project was built with clang 23.1.2, LLD 23.1.2 and QEMU 11.1.1, whose bundled OpenSBI (v1.8.1 in the logs) is what `-bios default` loads. From `projects/16-riscv-sv39-kernel`, the following commands build, test and reproduce every result.

```sh
brew install qemu llvm lld
make toolchain        # checks the toolchain and prints versions

make                  # build/kernel.elf (two-pass link, checks that text did not move)
make test             # self-tests under TCG and icount, plus four crash reports
make bench            # 3 TCG runs and 2 icount runs, writes results/bench-summary.txt
make results          # regenerates everything in results/

make run              # interactive demo and monitor; quit with Ctrl-A then X
make debug            # QEMU halted at reset with a gdb stub on :1234
make lldb             # in a second terminal, attach Apple lldb to 127.0.0.1:1234
```

A single test can be selected on the kernel command line, which is how the tick race was reproduced.

```sh
qemu-system-riscv64 -machine virt -cpu rv64 -smp 2 -m 128M -nographic \
  -bios default -kernel build/kernel.elf -append "mode=test only=ticks"
```

Add `-icount shift=0,sleep=off` for deterministic virtual time. Every number in this post comes from a file in `results/`, which holds the raw logs of each run, with the host load average recorded in the benchmark files.

Code is in `projects/16-riscv-sv39-kernel`, with the kernel in `kernel/`, the test and benchmark scripts in `tools/`, the memory map, trap flow and locking rules in `DESIGN.md`, and the full build log in `DEVLOG.md`.
