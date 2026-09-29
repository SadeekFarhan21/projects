#!/bin/sh
# Turn the text symbols of a linked kernel into a sorted C table for
# ksym_lookup(). Usage: NM=llvm-nm gensyms.sh kernel.pass1.elf > ksyms.c
set -e
NM=${NM:-llvm-nm}
echo '#include "types.h"'
echo 'struct ksym { u64 addr; const char *name; };'
echo 'const struct ksym ksyms[] = {'
"$NM" -n --defined-only "$1" | awk '$2 ~ /^[tT]$/ && $3 !~ /^\.L/ { printf "    {0x%s, \"%s\"},\n", $1, $3; n++ } END { if (n == 0) print "    {0, \"none\"}," }'
echo '};'
echo 'const u64 ksyms_count = sizeof(ksyms) / sizeof(ksyms[0]);'
