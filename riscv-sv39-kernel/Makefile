# 16-os: a small RISC-V 64 kernel for QEMU virt.
# Toolchain: Homebrew LLVM clang (cross-compiles to riscv64) + ld.lld + qemu.

BREW   ?= $(shell command -v brew 2>/dev/null || echo /opt/homebrew/bin/brew)
LLVM   ?= $(shell $(BREW) --prefix llvm 2>/dev/null || echo /opt/homebrew/opt/llvm)
CC      = $(LLVM)/bin/clang
LD      = $(shell command -v ld.lld 2>/dev/null || echo $(LLVM)/bin/ld.lld)
NM      = $(LLVM)/bin/llvm-nm
OBJDUMP = $(LLVM)/bin/llvm-objdump
QEMU   ?= qemu-system-riscv64

BUILD  := build
KERNEL := $(BUILD)/kernel.elf

# The machine is rv64gc, but the kernel itself is compiled without the F/D
# extensions so it never touches FP registers (which traps do not save).
ARCHFLAGS = --target=riscv64-unknown-elf -march=rv64imac_zicsr_zifencei -mabi=lp64 \
            -mcmodel=medany -mno-relax
CFLAGS = $(ARCHFLAGS) -std=gnu17 -O2 -g -Wall -Wextra -Werror -ffreestanding -fno-builtin \
         -nostdlib -fno-omit-frame-pointer -fno-optimize-sibling-calls \
         -fno-stack-protector -fno-pic -fno-asynchronous-unwind-tables \
         -fno-unwind-tables -Ikernel/include -MMD -MP
ASFLAGS = $(ARCHFLAGS) -Ikernel/include -MMD -MP
LDFLAGS = -nostdlib -static -T kernel/boot/kernel.ld --no-relax

SRCS_C := $(wildcard kernel/*/*.c)
SRCS_S := $(wildcard kernel/*/*.S)
OBJS   := $(patsubst %.c,$(BUILD)/%.o,$(SRCS_C)) $(patsubst %.S,$(BUILD)/%.o,$(SRCS_S))

# QEMU virt: 1 GiB address map, OpenSBI as M-mode firmware (-bios default).
SMP  ?= 2
MEM  ?= 128M
QEMUFLAGS = -machine virt -cpu rv64 -smp $(SMP) -m $(MEM) -nographic \
            -bios default -kernel $(KERNEL)
TIMEOUT ?= 120

.PHONY: all run test bench debug gdb lldb clean toolchain boot-log disasm results stats monitor-session

all: $(KERNEL)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/%.o: %.S
	@mkdir -p $(dir $@)
	@echo "  AS      $<"
	@$(CC) $(ASFLAGS) -c $< -o $@

# Two-pass link so panics can print function names: pass 1 links with an
# empty symbol table, then the text symbols of pass 1 become a table in
# .rodata for pass 2. .rodata sits after .text, so text addresses cannot move;
# the cmp below proves it.
$(BUILD)/ksyms_empty.c:
	@mkdir -p $(BUILD)
	printf '#include "types.h"\nstruct ksym { u64 addr; const char *name; };\nconst struct ksym ksyms[1];\nconst u64 ksyms_count = 0;\n' > $@

$(BUILD)/ksyms_empty.o: $(BUILD)/ksyms_empty.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/kernel.pass1.elf: $(OBJS) $(BUILD)/ksyms_empty.o kernel/boot/kernel.ld
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(BUILD)/ksyms_empty.o

$(BUILD)/ksyms.c: $(BUILD)/kernel.pass1.elf tools/gensyms.sh
	NM=$(NM) sh tools/gensyms.sh $< > $@

$(BUILD)/ksyms.o: $(BUILD)/ksyms.c
	$(CC) $(CFLAGS) -c $< -o $@

$(KERNEL): $(OBJS) $(BUILD)/ksyms.o kernel/boot/kernel.ld
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(BUILD)/ksyms.o
	@$(NM) -n $(BUILD)/kernel.pass1.elf | grep -i ' t ' > $(BUILD)/text1.txt
	@$(NM) -n $@ | grep -i ' t ' > $(BUILD)/text2.txt
	@cmp -s $(BUILD)/text1.txt $(BUILD)/text2.txt || (echo "error: text symbols moved between link passes"; exit 1)
	@$(OBJDUMP) -h $@ | awk '/text|rodata|data|bss|bootstacks/ {print "  " $$2 " size=0x" $$3 " vma=0x" $$4}'

run: $(KERNEL)
	$(QEMU) $(QEMUFLAGS) -append "mode=demo"

# Headless self-test plus crash-report checks. Exit status 0 means all passed.
TESTLOGS ?= $(BUILD)/test-logs
test: $(KERNEL)
	QEMU=$(QEMU) KERNEL=$(KERNEL) TIMEOUT=$(TIMEOUT) SMP=$(SMP) MEM=$(MEM) OUT=$(TESTLOGS) sh tools/run-tests.sh

bench: $(KERNEL)
	QEMU=$(QEMU) KERNEL=$(KERNEL) SMP=$(SMP) MEM=$(MEM) bash tools/run-bench.sh

# Boot log for results/: the demo with autoexit instead of the monitor.
boot-log: $(KERNEL)
	@mkdir -p results
	sh tools/timeout.sh $(TIMEOUT) $(QEMU) $(QEMUFLAGS) -append "mode=demo autoexit" < /dev/null | tee results/boot-log.txt

# Scripted session with the interactive monitor (input typed over the UART).
monitor-session: $(KERNEL)
	@mkdir -p results
	QEMU=$(QEMU) KERNEL=$(KERNEL) sh tools/monitor-session.sh > results/monitor-session.txt
	@sed -n '/kernel monitor ready/,$$p' results/monitor-session.txt

# Code size and line counts.
stats: $(KERNEL)
	@mkdir -p results
	@{ echo "# section sizes of $(KERNEL)"; $(OBJDUMP) -h $(KERNEL); \
	  echo "# lines per directory (C, assembly, headers, linker script)"; \
	  for d in kernel/boot kernel/arch kernel/core kernel/mm kernel/dev kernel/lib kernel/include kernel/test; do \
	    printf '%-16s %6s\n' $$d "$$(cat $$d/* | wc -l | tr -d ' ')"; done; \
	  printf '%-16s %6s\n' total "$$(cat kernel/*/* | wc -l | tr -d ' ')"; \
	  printf '%-16s %6s\n' tools "$$(cat tools/* Makefile | wc -l | tr -d ' ')"; } > results/code-size.txt
	@cat results/code-size.txt

# Regenerate every file in results/ (the numbers quoted in DEVLOG.md).
results: $(KERNEL)
	$(MAKE) test TESTLOGS=results
	$(MAKE) bench
	$(MAKE) boot-log
	$(MAKE) monitor-session
	$(MAKE) stats

# QEMU waits for a debugger on localhost:1234 before running any code.
debug: $(KERNEL)
	@echo "QEMU halted, gdb stub on :1234. In another terminal: make lldb (or make gdb)"
	$(QEMU) $(QEMUFLAGS) -append "mode=demo" -s -S

lldb: $(KERNEL)
	lldb $(KERNEL) -o "gdb-remote 127.0.0.1:1234"

gdb: $(KERNEL)
	riscv64-elf-gdb $(KERNEL) -ex "target remote 127.0.0.1:1234"

disasm: $(KERNEL)
	$(OBJDUMP) -d -S $(KERNEL) > $(BUILD)/kernel.asm
	@echo "wrote $(BUILD)/kernel.asm"

toolchain:
	sh tools/check-toolchain.sh

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)
