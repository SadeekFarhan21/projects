# Writing a RISC-V kernel from scratch

### What I wanted to build

I wanted a kernel small enough to read in an evening that still does the things a real one has to do. It should boot on a standard firmware interface, find its own memory, turn on paging with proper permissions, take interrupts, and run several threads that preempt each other. Every one of those claims should be checked by a test that runs headless and fails loudly. The target is 64-bit RISC-V on QEMU's `virt` board, with OpenSBI as machine-mode firmware so my code runs in supervisor mode the way Linux does. v0 stops before user mode.

The result is about 3521 lines of C, assembly, headers and linker script, with 0x96ce bytes of machine code in `.text` (`results/code-size.txt`). A test boot passes 22 self-tests (`results/selftest-tcg.txt`), and four more boots check that crashes produce readable reports and the right exit code.

### Theory

RISC-V has three privilege levels. OpenSBI owns machine mode. It configures physical memory protection, delegates traps to supervisor mode and then gets out of the way. The kernel asks it for services with `ecall`, the same way a user program asks a kernel. I used four SBI extensions, TIME to arm the timer, HSM to ask about other harts, SRST to power off and DBCN for the earliest console output.

A trap in S-mode does four things in hardware. It saves the PC in `sepc`, writes the cause to `scause` (the top bit says interrupt or exception), puts the faulting address or instruction bits in `stval`, and moves the old interrupt-enable bit into `sstatus.SPIE` while clearing `SIE`. Then it jumps to `stvec`. Everything else, including every general-purpose register, is software's job. `sret` reverses it.

Sv39 paging translates a virtual address through three levels of page tables, each level indexed by a slice of the address, with the low bits as the offset inside the page. A leaf entry carries R, W, X, U, G, A and D bits. Without the Svadu extension enabled, hardware that finds A or D clear raises a page fault instead of setting them, so I set them when I create each mapping.

A context switch between kernel threads only has to save callee-saved registers (`ra`, `sp`, `s0` to `s11`). The switch is an ordinary function call, so the compiler has already spilled the caller-saved ones. The hard part of scheduling is locking. The classic bug is the lost wakeup, where a thread checks a condition, gets interrupted, the condition changes and the wakeup fires, and only then does the thread go to sleep, forever.

### Architecture

The kernel is one image linked at `0x80200000`. Text, rodata and data/bss each start on a page boundary so they can get different permissions. After them come the boot stack slots, each a guard page followed by the stack itself. The kernel page table identity-maps RAM and the three devices it uses (UART, PLIC and the `sifive,test0` exit device). Thread stacks live at high virtual addresses (the first one starts at 0x0000003f00001000), with an unmapped guard page below each one. The boot log shows the resulting layout and that the table needed 71 pages (`results/boot-log.txt`).

Traps push a frame with every general-purpose register on the current kernel stack and call `kernel_trap` in C. Keeping the frame on the thread's own stack is what makes preemption simple. The timer handler can call `yield()`, run other threads for a while, and later return through the same frame.

The scheduler switches directly from one thread to the next. xv6 goes through a per-CPU scheduler thread. The run queue is a FIFO under one spinlock, and each hart has an idle thread made from its boot flow. A test boot and a demo boot pick their behavior from the kernel command line, which QEMU passes through the device tree.

DESIGN.md has the memory map, the trap flow diagram and the locking rules.

### Implementation

The entry code runs an `amoswap` lottery so only one hart continues, sets up that hart's stack, zeroes BSS and calls `kmain`. The lottery word sits in `.data`, because a BSS clear would reset it. With SBI HSM the second hart never arrives at all. The boot log reports it as "stopped" (`results/boot-log.txt`).

The device tree parser walks the flattened blob once. Properties always come before child nodes, so I finish a node when its first child or its end token appears. It collects the memory node, the `reserved-memory` children (OpenSBI reserves its own range there), the memory reservation block, the UART, the PLIC, the test device, the timebase and `bootargs`. For the 128 MiB machine it found 32768 pages, and 32127 were free after the kernel, the bitmap, OpenSBI and the DTB were taken out (`results/boot-log.txt`).

The page allocator is a bitmap. I picked it over a free list because freeing then becomes a bit test, so a double free is detected and returns an error instead of corrupting a list. One self-test allocates every free page, 32050 of them at that point, chains them through their first word with a tag, checks the tags, and frees them all (`results/selftest-tcg.txt`).

Paging maps each section with its own permissions and sets A and D up front. The self-tests prove the permissions rather than trust them. They write to text and rodata and expect a store page fault, and they execute a `ret` instruction placed in `.data` and expect an instruction page fault. They also read one byte below every stack and expect a load page fault. To do that without killing the kernel I wrote fault probes. A probe stores a recovery address in the per-hart struct before one risky instruction, and the trap handler resumes there if the instruction faults. It is the same idea as Linux exception tables.

The heap has seven power-of-two size classes. Each class cuts whole pages into blocks with a 16-byte header, so `kfree` needs no size and can reject double frees by checking a magic value. Anything bigger than a class gets contiguous pages.

The trap vector checks for stack overflow before it pushes anything. If `sp` is within one frame of the bottom of the current stack, pushing would fault again, and the kernel would loop forever. In that case the vector switches to a per-hart emergency stack and reports. In the crash test the report said `sp` was 16 bytes below the stack bottom and folded the recursion into one line, "same frame repeated 509 more times" (`results/crash-stackoverflow.txt`). Page fault reports walk the page table and say which level was invalid. They also flag addresses in page zero as probable NULL dereferences (`results/crash-pagefault.txt`).

For readable panics I link the kernel twice. The first link has an empty symbol table. Its text symbols become a sorted C array, and the second link includes that array in `.rodata`. `.rodata` comes after `.text`, so no code address can move, and the Makefile compares the text symbols of both links to prove it. Backtraces walk the frame-pointer chain (`-fno-omit-frame-pointer`), and the trap frame contains a fake frame record, so a backtrace from a fault goes through the trap into the code that faulted (`results/crash-panic.txt`).

The kernel is compiled for `rv64imac` even though the machine is `rv64gc`, and it clears `sstatus.FS`. Traps never save floating-point registers, so any FP instruction the compiler slipped in would trap instead of silently corrupting a thread's state.

### Problems

I expected triple-fault-style trouble when paging came on, and it did not happen. The first boot with `satp` written ran the whole demo. I think three choices prevented the usual failures. Every section boundary is page aligned in the linker script, A and D are set on every leaf, and the trap frame offsets are shared between C and assembly through one header checked by `_Static_assert`. The bugs I did hit were subtler.

The first was a flaky self-test. The timer test failed once with "check failed: mycpu()->ntimer - n0 == dt". I first blamed the compiler caching a counter that the interrupt handler updates, and made the counters `volatile`. The test then passed, but when I later rebuilt the kernel without `volatile` to reproduce the bug for this log, it still passed. The disassembly showed the counter reloaded after an opaque `kprintf` call anyway. The real bug was in the test. It read `ticks` and the per-hart interrupt count as two separate loads with interrupts enabled, and at the end a slow `kprintf` sat between the two loads, so a tick landing in between made them disagree by one. I fixed it by taking both values in one snapshot with interrupts off. The pre-fix test failed 4 of 30 boots and the fixed test 0 of 30 (`results/bug-repro-tick-race.txt`). The lesson is that a fix you cannot reproduce the failure for is a guess.

The `volatile` was still needed elsewhere. My timer latency benchmark sets `recording = true`, spins until 200 ticks pass, then sets it back to false. The spin loop reads only the volatile `ticks`, so the compiler treated the first store as dead and removed it. The busy benchmark then recorded n=0 samples, and the mean printed as 18446744073709551615 (`results/bug-repro-volatile.txt`). That number is itself a lesson. RISC-V integer division by zero does not trap, it returns all ones. The idle variant of the same benchmark worked, because a call to `thread_sleep_ticks` sat between the stores. Marking the shared statistics `volatile` and guarding the division fixed it.

The third was a wrong exit code. After the kernel printed a perfect page-fault report, QEMU once exited with 0 instead of 3, and `make test` failed. My exit path wrote the failure value to the test device and then fell through to an SBI shutdown call "just in case". QEMU acted on the first write asynchronously, the vCPU kept running, and OpenSBI's poweroff wrote the success value to the same device. Whichever request QEMU processed decided the exit code. Before the fix, 10 of 40 crash boots returned the wrong code, and after it 0 of 40 (`results/exit-race.txt`). The fix is to park the hart after the test-device write.

The fourth was a bogus frame at the end of every thread backtrace. A new thread's context starts with `ra` pointing at `thread_start`. The prologue saved that as the return address, and my lookup of `ra - 1` landed in whatever function sits before `thread_start` in memory. With the fix reverted, the panic backtrace ends in `thread_init+0x6c`, a function the thread never called (`results/bug-repro-trampoline.txt`). A three-instruction trampoline that zeroes `ra` and `s0` ends the chain cleanly.

Smaller things also cost time. `-std=c17` turns off the `asm` keyword, so the kernel builds as `gnu17`. Clang's `-Winfinite-recursion` rejected my deliberate stack-overflow function until the recursion depended on a volatile flag. The zsh shell on this Mac does not split an unquoted `$VAR` into words, so my first batch of reproduction runs silently started nothing. The first `lldb` attach failed until I pointed it at `127.0.0.1` rather than `localhost`. While writing the test harness I also noticed that `uart_init` resets the receive FIFO, which throws away any keystrokes QEMU delivered before the driver ran. The harness waits for the self-test banner before it types.

The last problem was the host. Fifteen other builds ran on this machine at the same time, and the load average sat in the hundreds (the benchmark files record it). Under plain TCG, QEMU's virtual clock follows the host clock, so a starved vCPU thread misses timer ticks. I loosened the tick test's lower bound for that reason and added a second pass of the whole suite under `-icount`, where time is deterministic.

### Experiments

All measurements run inside QEMU and are not real hardware numbers. Under plain TCG, QEMU translates guest code to host code and the timings include translation and host scheduling. Under `-icount shift=0,sleep=off`, virtual time advances exactly one nanosecond per guest instruction, so a "nanosecond" there is really an instruction count. I report both. Every benchmark loops many times and divides, because `rdtime` ticks at the device tree timebase of 10000000 Hz, a resolution of 100 ns.

The first question was what a context switch costs. A raw `swtch` between two contexts costs 34 guest instructions in each direction, and a full `yield` through the scheduler costs 190 (`results/bench-summary.txt`). The difference is taking and releasing `sched_lock` (with the CSR writes to mask interrupts), queue operations and bookkeeping. Under TCG the same two operations took 50 to 52 ns and 570 to 601 ns across three runs.

The second was trap cost against firmware calls. An `ebreak` trap into my vector and back costs 140 instructions, and an SBI `ecall` round trip into OpenSBI costs 291 (`results/bench-summary.txt`). Under TCG the gap nearly closes, 1158 to 1225 ns against 1520 to 1583 ns. On this emulator most of a trap's cost is QEMU leaving translated code, not the handler.

The third was the page allocator. Allocation without zeroing costs 105 instructions and a free costs 103. Allocation with zeroing costs 2184, so clearing 4 KiB dominates (`results/bench-summary.txt`). Under TCG the zeroing allocation took 1622 to 1645 ns, less than its icount instruction count, because QEMU runs that store loop faster than one guest instruction per nanosecond.

The fourth was timer interrupt latency, from the programmed deadline to the C handler. Under icount the median was 100 ns and the mean 85 ns, idle or busy, which is within one timebase tick (`results/bench-summary.txt`). Under plain TCG the median was between 2333000 and 2614000 ns in all six runs, idle and busy. That is a steady offset of a few milliseconds rather than noise, and the same file shows it at load averages of 129.23 and up. My guess, which I have not verified, is that it comes from how QEMU arms its host timer on macOS. The icount numbers show the kernel's own path is short.

The fifth was whether icount is reproducible. Two icount benchmark runs gave identical cycles/op in every column (`results/icount-determinism.txt`). The timer test saw exactly 10 ticks in a 100 ms busy wait under icount (`results/selftest-icount.txt`).

The last was why the lock matters. Four threads increment a shared counter, 80000 increments in total, with a short delay between each read and write. Under the spinlock the count is exactly 80000. The unlocked control loses updates when a preemption lands inside its window. Under icount it lost 3605 updates. In the plain TCG run it lost 0 (`results/selftest-icount.txt`, `results/selftest-tcg.txt`), so the control is informational and not an assertion. One plausible reason for the TCG result is that a tick arriving while the lock is held is deferred until `spin_unlock` turns interrupts back on, just before the unlocked read. That biases preemption toward the edge of the racy window, but I have not measured it.

### Results

- `make test` passes. That is 22 self-tests under TCG (including the UART receive test with typed input), 21 under icount, and four crash scenarios exiting with code 3 (`results/selftest-tcg.txt`, `results/selftest-icount.txt`, `results/crash-*.txt`).
- The kernel boots through OpenSBI, parses the device tree, enables Sv39 and reaches its first thread in 16061 microseconds of virtual time under icount (`results/selftest-icount.txt`). That figure includes OpenSBI's own startup.
- In the demo, four workers are preempted between 4 and 7 times each, their shared tally ends at the expected 16, and the boot takes 48 context switches before it powers off (`results/boot-log.txt`).
- Cost per operation in guest instructions is 34 per `swtch`, 190 per scheduler switch, 140 per trap and 291 per SBI call (`results/bench-summary.txt`).
- The interactive monitor works over UART receive interrupts, with 10 interrupts for the scripted session (`results/monitor-session.txt`).

### What I would change

I would make the kernel higher-half before starting user mode, so user programs can own the low addresses without special cases. I would give each hart its own run queue and add the "still on CPU" flag that `thread_join` needs before it frees a stack under SMP. I would write `stimecmp` directly when Sstc is present, and measure whether it removes the TCG timer offset. The page allocator should use the gap between OpenSBI's reserved range and the kernel image, and its contiguous allocator should stop scanning from zero. The heap should return empty slab pages. UART output should be interrupt driven so a long `kprintf` does not hold interrupts off. Finally, I would build the fault-injection tests in from the start. The exit-code race and the tick race were both found by running the same boot many times, so that should be a standard test mode, not an afterthought.

### Verification

On 2026-09-26 an independent clean rebuild (`make clean`, `make -j2`, `make test`) passed every self-test and all four deliberate crash reports.
