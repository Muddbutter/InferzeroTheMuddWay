#!/usr/bin/env bash
# dec_attn_kl.sh: the Linux counterpart of bench/dec_attn_kl.ps1 (2026-11-23, #45): the long-context decode attention's
# switches (ARC_QWEN_DSPLIT=1, ARC_ATTN_DEC_SMAX=512, both) against the default GPU path, token by token. The default
# path's own log-probs at every decoded position are the reference (runz ARC_KL_SAVE); each switch's KL against them
# (runz --kl-decode). The first CTX tokens are the prompt (ARC_KL_FROM), the next STEPS decoded one by one.
#   bench/dec_attn_kl.sh MODEL.gguf [CTX 16384] [STEPS 256]
# The limit for these switches (AGENTS.md): max KL <= 0.015, the same top token at >= 98 % of the positions,
# against the default GPU path (not f32: the default's own KL against f32 adds to it).
set -euo pipefail
model=${1:?usage: dec_attn_kl.sh MODEL.gguf [CTX] [STEPS]}
ctx=${2:-16384}
steps=${3:-256}
root=$(cd "$(dirname "$0")/.." && pwd)
runz=${RUNZ:-$root/bin/runz}
[ -x "$runz" ] || { echo "runz not found: $runz"; exit 1; }
[ -f "$model" ] || { echo "model not found: $model"; exit 1; }
if pgrep -x serverz >/dev/null || pgrep -x runz >/dev/null; then
  echo "serverz / runz is running: stop it first, two GPU users at once can fault the card"; exit 1
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# the text: the docs, repeated until it holds enough tokens (~4 characters a token; a margin of 2x)
need=$(( (ctx + steps) * 8 ))
: > "$work/text.txt"
while [ "$(wc -c < "$work/text.txt")" -lt "$need" ]; do
  for d in "$root"/docs/*.md; do cat "$d" >> "$work/text.txt"; printf '\n\n' >> "$work/text.txt"; done
done
"$runz" "$model" --text-file "$work/text.txt" --dump-ids "$work/ids.txt" > /dev/null
n=$((ctx + steps))
have=$(tr ',' '\n' < "$work/ids.txt" | grep -c .)
[ "$have" -ge "$n" ] || { echo "the text gave $have tokens, fewer than $n"; exit 1; }
# (awk reads to the end: head would close the pipe early and tr's SIGPIPE ends the script under pipefail, exit 141;
#  skipzx-lab, #45)
tr ',' '\n' < "$work/ids.txt" | awk -v n="$n" 'NR <= n' | paste -sd, - > "$work/ids_cut.txt"
ctxarg=$(( (n + 256 + 255) / 256 * 256 ))
echo "dec_attn_kl: $(basename "$model"), $ctx prompt tokens, $steps decoded one by one (--ctx $ctxarg)"
t0=$(date +%s)
ARC_KL_FROM=$ctx ARC_KL_SAVE="$work/ref.bin" "$runz" "$model" --tokens-file "$work/ids_cut.txt" --ctx "$ctxarg" --kl-decode 2>&1 | grep -E "ARC_KL_SAVE|error" | sed 's/^/  default: /'
echo "  (the reference run took $(( $(date +%s) - t0 )) s)"
run() {  # NAME VAR=VALUE...
  local name=$1; shift
  printf '  %-24s ' "$name"
  env "$@" ARC_KL_FROM="$ctx" "$runz" "$model" --tokens-file "$work/ids_cut.txt" --ctx "$ctxarg" --kl-ref "$work/ref.bin" --kl-decode 2>&1 | grep -E "KL decode path|error" | tr '\n' ' '
  echo
}
run "ARC_QWEN_DSPLIT=1" ARC_QWEN_DSPLIT=1
run "ARC_ATTN_DEC_SMAX=512" ARC_ATTN_DEC_SMAX=512
run "both" ARC_QWEN_DSPLIT=1 ARC_ATTN_DEC_SMAX=512
echo "dec_attn_kl: against the default GPU path; the limit: max KL <= 0.015 and the same top token at >= 98 % of the positions"
