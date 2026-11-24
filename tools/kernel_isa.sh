#!/usr/bin/env bash
# One OpenCL kernel through a given Intel ocloc, for its spills and size: the scratch (spill) messages in its GPU
# assembly, the compiler's own spill line, and the instruction count. The Linux counterpart of tools/kernel_isa.ps1,
# for the users' compiler: since 2026-11-23 every DG2 (A770) / BMG (B-series) spill check runs on the compute
# runtime the users have (26.40.40196.8, IGC 2.42.6), not oneAPI's ocloc (26.24 on Windows), which called a 32 x 48
# GEMM tile spill-free in its K loop where 26.40 spills it (skipzx-lab's A770, #45: 1,440 B a thread, -18.6 % at 4K).
#   tools/kernel_isa.sh DEVICE FILE.cl KERNEL "BUILD OPTIONS" [LABEL]
#   e.g. tools/kernel_isa.sh dg2-g10 zero/kernels/gemm_xw8.cl gemm_xw8 "-cl-intel-256-GRF-per-thread -DXW_SLM ..."
# OCLOC_ROOT: the unpacked runtime's root (default ~/intel-ocloc-26.40/root: usr/bin/ocloc-26.40.1 and its
# libraries under usr/lib/x86_64-linux-gnu and usr/local/lib), or OCLOC: an ocloc on the PATH's way.
set -u
dev=$1; file=$2; kern=$3; opts=$4; label=${5:-$kern}
root=${OCLOC_ROOT:-$HOME/intel-ocloc-26.40/root}
if [ -n "${OCLOC:-}" ]; then oc=$OCLOC
elif [ -x "$root/usr/bin/ocloc-26.40.1" ]; then
  oc=$root/usr/bin/ocloc-26.40.1
  export LD_LIBRARY_PATH=$root/usr/lib/x86_64-linux-gnu:$root/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
else
  oc=$(command -v ocloc) || { echo "no ocloc: set OCLOC_ROOT or OCLOC"; exit 2; }
fi
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
log=$("$oc" compile -device "$dev" -file "$file" -options "$opts" -output k -out_dir "$out" 2>&1)
bin=$(find "$out" -maxdepth 1 -name "*.bin" | head -1)
if [ -z "$bin" ]; then
  echo "== $label: build failed"
  printf '%s\n' "$log" | grep -iE "error" | head -5
  exit 1
fi
( cd "$out" && "$oc" disasm -file "$bin" -device "$dev" -dump dis >/dev/null 2>&1 )
asm=$(find "$out" -name "*$kern*.asm" | head -1)  # (named .text.<kernel>.asm: a dot file, so not a plain glob)
[ -n "$asm" ] || { echo "== $label: no assembly for $kern"; exit 1; }
sc=$(grep -ci "scratch" "$asm")  # (grep -c prints 0 and fails on no match: no '|| echo' here)
ins=$(grep -cE "^\s+\(?[A-Za-z0-9~&|.]*\)?\s*[a-z][a-z0-9_.]* \(" "$asm")
spl=$(printf '%s\n' "$log" | grep -ioE "allocated [0-9]+ regs and spilled[^,;]*" | head -1)
echo "== $label: scratch messages $sc, instructions ~$ins${spl:+ | compiler: $spl}"
