#!/usr/bin/env bash
# The options that move the logits, each against the f32 CPU reference (tools/ref_forward), on your GPU, in one table:
# the KL mean and max and the same top token over every position (bench/kl_options.ps1's twin; the user, 2026-11-21:
# "make the user test KL against f32 reference for the new options"). The rows that apply to the model, found from its
# first run's start-up lines:
#   - the default against its reference;
#   - the two QSA models (the QSA 27B, Flash Next): the other attention mode (ARC_QSA=1 / ARC_QSA=0) against its own
#     reference and against the default's, the selection cut to 64 keys on both sides (ARC_QSA_TOPK / REF_TOPK);
#   - where H8 runs: ARC_H8_Q2=1 (More precise prompt reading);
#   - Flash Next: its fast mode (ARC_FN_PRECISE=0; precise mode is the default since 2026-11-22);
#   - with an MTP block: ARC_MTP_VOCAB=32768 (Faster drafting) must give the same text as plain --mtp.
# The reference reads the weights row by row from the file: a few GB of RAM besides the file's pages, and CPU time
# (about 30 minutes for 480 tokens of the 27B on a laptop). Each reference is made once and kept (--fresh: again).
#   bash bench/kl_options.sh --model <model.gguf> [--tokens 480] [--fresh]
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 1
MODEL="" TOKENS=480 FRESH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2;;
    --tokens) TOKENS=$2; shift 2;;
    --fresh) FRESH=1; shift;;
    *) echo "kl_options: unknown option $1 (--model F, --tokens N, --fresh)" >&2; exit 2;;
  esac
done
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "kl_options: --model <model.gguf> (a file)" >&2; exit 2; }
RUNZ=$ROOT/bin/runz REF=$ROOT/bin/ref_forward
for e in "$RUNZ" "$REF"; do [ -x "$e" ] || { echo "kl_options: $e not found (an inferzero package, or ./build.sh)" >&2; exit 1; }; done
if pgrep -x serverz >/dev/null || pgrep -x runz >/dev/null; then echo "kl_options: a serverz / runz is running: stop it first" >&2; exit 1; fi
# (the file's name, other characters as _, and its size: a folder per model file, so no model reads another's cached
#  references. Until 2026-11-22 tr -c 'A-Za-z0-9._-\n' took "_-\n" for a reverse range and failed, and every model
#  shared ~/.inferzero/kl-options/ itself: Daleang, #73)
WORK="$HOME/.inferzero/kl-options/$(basename "$MODEL" | sed 's/[^A-Za-z0-9._-]/_/g')-$(stat -c %s "$MODEL")"
[ "$FRESH" = 1 ] && rm -rf "$WORK"
mkdir -p "$WORK"
stamp() { echo "$(date +%H:%M:%S) $*" >&2; }
text_hash() { awk '/^----- text -----$/ { on = 1; next } /^----------------$/ { on = 0 } on' | sha256sum | cut -c1-16; }
# --- the tokens: the first $TOKENS of USAGE.md after its first 2,000 characters
IDS="$WORK/tokens$TOKENS.ids"
if [ ! -f "$IDS" ]; then
  tail -c +2001 "$ROOT/USAGE.md" | head -c 40000 > "$WORK/usage.txt"
  "$RUNZ" "$MODEL" --text-file "$WORK/usage.txt" --dump-ids "$WORK/usage.ids" >/dev/null 2>&1
  n=$(tr ',' '\n' < "$WORK/usage.ids" | grep -c .)
  [ "$n" -ge "$TOKENS" ] || { echo "kl_options: only $n tokens from USAGE.md" >&2; exit 1; }
  tr ',' '\n' < "$WORK/usage.ids" | grep . | head -n "$TOKENS" | paste -sd, - > "$IDS"
fi
# --- what the model has
PROBE=$("$RUNZ" "$MODEL" --tokens-file "$IDS" --n 1 2>&1)
H8=0 FN=0 Q27=0
grep -q "H8 on for the prompt GEMMs" <<<"$PROBE" && H8=1
grep -q "Flash Next with" <<<"$PROBE" && FN=1
grep -q "QSA indexer present" <<<"$PROBE" && Q27=1
stamp "model: $(basename "$MODEL") | H8 $([ $H8 = 1 ] && echo on || echo off) | $([ $FN = 1 ] && echo 'Flash Next (sparse by default)' || { [ $Q27 = 1 ] && echo 'QSA 27B (full by default)' || echo 'no QSA indexer'; })"
get_ref() {  # name env...: the reference file (made once)
  local name=$1; shift
  local f="$WORK/$name.bin"
  if [ ! -f "$f" ]; then
    stamp "f32 reference '$name' (CPU, minutes to an hour)..."
    local t0=$SECONDS
    env REF_ALL=1 "$@" "$REF" "$MODEL" "@$IDS" "$f" > "$WORK/$name.log" 2>&1
    [ -f "$f" ] || { echo "kl_options: the reference '$name' failed: see $WORK/$name.log" >&2; exit 1; }
    stamp "  done in $(( (SECONDS - t0) / 60 )) min"
  fi
  printf '%s' "$f"
}
REF_FULL=$(get_ref ref_full REF_DENSE=1) || exit 1
REF_SPARSE=""
if [ $FN = 1 ] || [ $Q27 = 1 ]; then REF_SPARSE=$(get_ref ref_sparse64 REF_TOPK=64) || exit 1; fi
ROWS=()
add_row() {  # option against ref env...
  local opt=$1 against=$2 ref=$3; shift 3
  local o m
  o=$(env "$@" "$RUNZ" "$MODEL" --tokens-file "$IDS" --kl-ref "$ref" --n 1 2>&1)
  m=$(grep -oE 'KL\(ref \|\| inferzero\) = [0-9.]+ \(mean over [0-9]+ positions, max [0-9.]+\), top token agrees [0-9]+ / [0-9]+' <<<"$o" | tail -n 1)
  # (the share without the reference's near-ties, the rule of 2026-11-24: runz prints it after the strict one)
  local nt
  nt=$(grep -oE 'without near-ties \(ref top-2 margin < [0-9.]+\): [0-9]+ / [0-9]+' <<<"$o" | tail -n 1 | sed -E 's/.*: ([0-9]+ \/ [0-9]+)$/\1/')
  [ -n "$nt" ] || nt="-"
  if [ -n "$m" ]; then
    ROWS+=("| $opt | $against | $(sed -E 's/.*= ([0-9.]+) \(.*/\1/' <<<"$m") | $(sed -E 's/.*max ([0-9.]+)\).*/\1/' <<<"$m") | $(sed -E 's/.*agrees ([0-9]+ \/ [0-9]+).*/\1/' <<<"$m") | $nt |")
  else
    ROWS+=("| $opt | $against | failed | - | - | - |")
  fi
  stamp "$opt against $against done"
}
if [ $FN = 1 ]; then
  add_row "default (sparse attention)" "sparse reference" "$REF_SPARSE" ARC_QSA_TOPK=64
  add_row "ARC_QSA=0 (full attention)" "full reference" "$REF_FULL" ARC_QSA=0
  add_row "ARC_QSA=0 (full attention)" "sparse reference (how far from the default)" "$REF_SPARSE" ARC_QSA=0
  add_row "ARC_FN_PRECISE=0 (fast mode)" "sparse reference" "$REF_SPARSE" ARC_FN_PRECISE=0 ARC_QSA_TOPK=64
  add_row "ARC_FN_PRECISE=0, ARC_QSA=0 (fast mode)" "full reference" "$REF_FULL" ARC_FN_PRECISE=0 ARC_QSA=0
elif [ $Q27 = 1 ]; then
  add_row "default (full attention)" "full reference" "$REF_FULL"
  add_row "ARC_QSA=1 (sparse attention)" "sparse reference" "$REF_SPARSE" ARC_QSA=1 ARC_QSA_TOPK=64
  add_row "ARC_QSA=1 (sparse attention)" "full reference (how far from the default)" "$REF_FULL" ARC_QSA=1 ARC_QSA_TOPK=64
else
  add_row "default" "reference" "$REF_FULL"
fi
if [ $H8 = 1 ]; then
  if [ $FN = 1 ]; then add_row "ARC_H8_Q2=1 (more precise prompt reading)" "the default's reference" "$REF_SPARSE" ARC_H8_Q2=1 ARC_QSA_TOPK=64
  else add_row "ARC_H8_Q2=1 (more precise prompt reading)" "the default's reference" "$REF_FULL" ARC_H8_Q2=1; fi
fi
# --- Faster drafting: the same text as plain drafting
DRAFT="not run (no MTP block)"
Q="$WORK/question.txt"
printf '%s' $'<|im_start|>user\nWrite a Python function that merges two sorted lists into one sorted list, with a docstring.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' > "$Q"
A=$("$RUNZ" "$MODEL" --text-file "$Q" --n 128 --mtp --show 2>&1)
if grep -q "MTP:" <<<"$A" && ! grep -q "no MTP block" <<<"$A"; then
  B=$(ARC_MTP_VOCAB=32768 "$RUNZ" "$MODEL" --text-file "$Q" --n 128 --mtp --show 2>&1)
  ha=$(text_hash <<<"$A") hb=$(text_hash <<<"$B")
  if [ "$ha" = "$hb" ]; then DRAFT="the same text as --mtp ($ha)"; else DRAFT="TEXT DIFFERS: --mtp $ha, ARC_MTP_VOCAB=32768 $hb"; fi
fi
GPU=$(grep -m1 "GPU 0 (" <<<"$PROBE" | sed 's/^ *//')
echo
echo "kl_options: $(basename "$MODEL"), $TOKENS tokens of USAGE.md, ${GPU:-GPU 0}"
echo "release: $(head -n 1 "$ROOT/bin/PREBUILT.txt" 2>/dev/null || git -C "$ROOT" log -1 --format=%h 2>/dev/null)"
echo
echo "| option | against | KL mean | KL max | same top token | without near-ties |"
echo "|---|---|---|---|---|---|"
printf '%s\n' "${ROWS[@]}"
echo "(without near-ties: positions whose reference top two are within 0.08 nats left out, ARC_KL_TIE; the KL counts every position)"
echo
echo "Faster drafting (ARC_MTP_VOCAB=32768): $DRAFT"
[ $FN = 1 ] || [ $Q27 = 1 ] && echo "(the sparse rows with the selection cut to 64 keys on both sides, ARC_QSA_TOPK / REF_TOPK, so that these tokens are sparse)"
echo "paste everything from 'kl_options:' into the GitHub issue"
