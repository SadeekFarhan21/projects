#!/bin/sh
# Verify the cross toolchain and emulator are installed and print versions.
# Install with: brew install qemu llvm lld
ok=1
BREW=$(command -v brew || echo /opt/homebrew/bin/brew)
LLVM=$($BREW --prefix llvm 2>/dev/null || echo /opt/homebrew/opt/llvm)

check() { # name path-or-command version-args
    if command -v "$2" >/dev/null 2>&1 || [ -x "$2" ]; then
        printf '%-22s %s\n' "$1" "$("$2" $3 2>&1 | head -1)"
    else
        printf '%-22s MISSING (%s)\n' "$1" "$2"; ok=0
    fi
}

check "clang (Homebrew LLVM)" "$LLVM/bin/clang" --version
check "ld.lld" "$(command -v ld.lld || echo $LLVM/bin/ld.lld)" --version
check "llvm-nm" "$LLVM/bin/llvm-nm" --version
check "llvm-objdump" "$LLVM/bin/llvm-objdump" --version
check "qemu-system-riscv64" qemu-system-riscv64 --version
if command -v lldb >/dev/null 2>&1; then check "lldb (optional)" lldb --version; fi
if command -v riscv64-elf-gdb >/dev/null 2>&1; then check "riscv64-elf-gdb (optional)" riscv64-elf-gdb --version; fi

if [ -x "$LLVM/bin/clang" ]; then
    if "$LLVM/bin/clang" -print-targets 2>/dev/null | grep -q riscv64; then
        echo "clang riscv64 target: yes"
    else
        echo "clang riscv64 target: NO"; ok=0
    fi
fi
if command -v qemu-system-riscv64 >/dev/null 2>&1; then
    # OpenSBI firmware ships with QEMU and is loaded by -bios default
    fw=$(ls "$(dirname "$(command -v qemu-system-riscv64)")"/../share/qemu/opensbi-riscv64-generic-fw_dynamic.bin 2>/dev/null)
    echo "OpenSBI firmware: ${fw:-not found next to qemu (QEMU may still find it)}"
fi

if [ $ok = 1 ]; then echo "toolchain OK"; else echo "toolchain incomplete: brew install qemu llvm lld"; exit 1; fi
