# 16-os, a RISC-V kernel from scratch

A small operating system kernel for 64-bit RISC-V (rv64gc), written in freestanding C and assembly. It boots on the QEMU `virt` machine under OpenSBI, runs in supervisor mode, turns on Sv39 paging, handles traps and interrupts, and schedules preemptive kernel threads. Every boot can run a built-in self-test suite that powers QEMU off with a pass or fail exit code, so `make test` works headless.

v0 is kernel only. There is no user mode yet.

### What it does

- Boots at `0x80200000` from OpenSBI, parks secondary harts, zeroes BSS and enters C on a per-hart boot stack.
- Reads the device tree blob (passed in `a1`) for RAM size, reserved ranges, UART, PLIC, the test device, the timebase and the kernel command line. Nothing about the machine is hard-coded.
- Console output goes through the SBI debug console until the UART is found, then through an NS16550A driver. `kprintf` supports `%d %i %u %x %X %p %s %c %%`, the `l`/`ll`/`z` modifiers, width, `0` padding and `-` alignment.
- Panics print file, line, message and a symbolized frame-pointer backtrace.
- Physical memory is a bitmap page allocator with double-free detection. The kernel heap has power-of-two size classes (32 to 2048 bytes) and uses contiguous pages for larger requests.
- Sv39 page tables map text R-X, rodata R, data and bss RW, and leave a guard page below every kernel stack.
- The trap vector saves all 31 registers plus the CSRs, decodes `scause`, and prints readable reports for page faults (including a page-table walk) and illegal instructions. It detects kernel stack overflow before it recurses.
- A 100 Hz SBI timer, and the PLIC with UART receive interrupts, so typed characters echo.
- Kernel threads with a direct thread-to-thread context switch, round-robin preemption, sleep/wakeup, spinlocks that disable interrupts, and sleep locks.
- 22 boot-time self-tests, 4 crash-report tests and in-kernel micro-benchmarks.

### Milestone status

| Milestone | Status |
|---|---|
| Boot, linker script, per-hart stacks, BSS, park other harts | done |
| Console (SBI then UART), printf, panic with backtrace | done |
| Device tree parsing, bitmap page allocator, double-free tests | done |
| Sv39 kernel page table, W^X permissions, stack guard pages, kmalloc | done |
| Trap vector, scause decoding, fault reports, SBI timer, PLIC and UART RX | done |
| Kernel threads, context switch, preemptive round robin, sleep/wakeup, locks | done |
| Self-test suite with QEMU exit codes, `make test` | done |
| Measurements inside QEMU (results/) | done |
| User mode, syscalls, user shell | planned |
| Processes with fork/exec/wait and ELF loading | planned |
| virtio-blk driver, filesystem with buffer cache and logging | planned |
| SMP (all harts running the scheduler) | planned |
| Pipes and file descriptors | planned |
| Copy-on-write fork | planned |
| Port to AArch64 or real hardware (VisionFive 2) | maybe |

### Install the toolchain (macOS, Apple silicon)

Apple's clang has no RISC-V backend, so the build uses Homebrew LLVM.

```sh
brew install qemu llvm lld
make toolchain        # runs tools/check-toolchain.sh and prints versions
```

This project was built with Homebrew clang 23.1.2, LLD 23.1.2 and QEMU 11.1.1. QEMU ships the OpenSBI firmware that `-bios default` loads.

### Build and run

```sh
make                  # build/kernel.elf
make run              # boot the demo in QEMU; quit with Ctrl-A then X
```

`make run` starts four worker threads that print interleaved lines while the timer preempts them. It then opens a small monitor on the UART. Commands are `help`, `ps`, `mem`, `ticks`, `stats`, `pf` (a recovered page fault), `panic` and `poweroff`.

The kernel command line (QEMU `-append`) selects a mode. `mode=demo` is the default. `mode=test` runs the self-tests, and `only=<text>` runs only the tests whose name contains the text. `mode=bench` runs the benchmarks. `crash=panic|pagefault|illegal|stackoverflow` crashes on purpose. `autoexit` powers off after the demo.

### Test

```sh
make test
```

This boots QEMU headless several times, each run under a timeout (`TIMEOUT=120` seconds by default).

1. The self-test suite under plain TCG, with `ping` typed into the UART to test receive interrupts.
2. The same suite under `-icount shift=0,sleep=off`, where virtual time is deterministic.
3. Four deliberate crashes. Each must print the expected report and exit with code 3.

The kernel exits through SBI system reset when everything passes. It writes the failure code to the `sifive,test0` device otherwise, so QEMU's exit status is the test result. Logs go to `build/test-logs/`. `make results` regenerates everything in `results/` (tests, benchmarks, boot log, monitor session, code size).

### Benchmarks

```sh
make bench
```

This runs the benchmarks three times under plain TCG and twice under icount, and writes `results/bench-summary.txt`. These are QEMU timings, not hardware timings. Under icount a "nanosecond" is one guest instruction. Headline numbers from `results/bench-summary.txt`:

| Operation | icount (guest instructions) | TCG wall clock (ns, min to max of 3 runs) |
|---|---|---|
| `swtch`, one direction | 34 | 50 to 52 |
| yield, thread to thread through the scheduler | 190 | 570 to 601 |
| trap round trip (`ebreak`) | 140 | 1158 to 1225 |
| SBI `ecall` round trip | 291 | 1520 to 1583 |
| page alloc (no zeroing) / free | 105 / 103 | 302 to 322 / 354 to 374 |
| `kmalloc(64)` + `kfree` | 222 | 850 to 868 |

### Debug

```sh
make debug            # QEMU halted at reset, gdb stub on port 1234 (-s -S)
make lldb             # in a second terminal: Apple lldb, attached to 127.0.0.1:1234
```

Apple's `lldb` understands riscv64 ELF and the gdb remote protocol. I tested it with breakpoints on `kmain` and `thread_start`, with source lines and a backtrace. Useful commands are `breakpoint set --name kmain`, `continue`, `bt`, `register read a0 a1` and `x/4gx $sp`. If you prefer gdb, run `brew install riscv64-elf-gdb` and then `make gdb`. `make disasm` writes `build/kernel.asm` with interleaved source.

### Layout

```
kernel/boot     entry.S (first instructions), kernel.ld (memory layout)
kernel/arch     trapvec.S, switch.S, probe.S (fault probes), sbi.c
kernel/core     main.c, trap.c, thread.c (scheduler), timer.c, spinlock.c, panic.c
kernel/mm       pmm.c (page frames), vm.c (Sv39), kmalloc.c (heap)
kernel/dev      fdt.c (device tree), uart.c, plic.c, console.c
kernel/lib      printf.c, string.c
kernel/test     selftest.c, bench.c, demo.c (and monitor), crash.c
tools           check-toolchain.sh, run-tests.sh, run-bench.sh, gensyms.sh, timeout.sh, monitor-session.sh
results         raw logs of every run quoted in DEVLOG.md
```

DESIGN.md explains the memory map, boot sequence, trap flow, scheduler and locking rules. DEVLOG.md is the build log.

### Known issues

- Only one hart runs kernel code. The locking is written for SMP, but `thread_join` frees a zombie's stack without checking that no other hart is still on it.
- The 1.6 MiB between OpenSBI's reserved range and the kernel image is never used.
- Slab pages are never given back to the page allocator, and `pmm_alloc_contig` is a linear first-fit scan.
- `kfree` on a wild pointer reads the header in front of it and can fault instead of returning an error.
- UART output is polled. A long `kprintf` keeps interrupts off while it writes.
- Under plain TCG the timer interrupt arrives about 2.4 ms late on this host (see DEVLOG.md). Under icount it arrives within one 100 ns timebase tick.
