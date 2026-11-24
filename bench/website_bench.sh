#!/usr/bin/env bash
# bench/website_bench.ps1 is the same script for Windows; the two must stay the same: change both together.
# Benchmarks for the website: per model, prompt processing and generation from short prompts to 131K tokens, the
# generation modes (plain, MTP, DFlash2), and the KL against an f32 CPU reference. Writes one JSON object per
# measurement to <out>/results.jsonl and the tables to <out>/summary.md.
#   bash bench/website_bench.sh [--models a.gguf b.gguf] [--sizes 128 512 2048 8192 32768 131072] [--out bench/results/website]
#   [--skip-kl] [--skip-modes] [--gen 64] [--cooldown 20]   (--sizes 128,512 works too)
# A. Prompt sizes: runz --bench-pp N --n GEN. Prompt tok/s (best of 2 runs up to 8K, one run above), time to the first
#    token (N / prompt tok/s), and generation tok/s at that depth (GEN tokens after the N-token prompt), plain decoding.
# B. Generation modes, short context: a 256-token coding answer, plain / --mtp / --dflash (when the model has an MTP
#    block or a DFlash2 drafter is in the Hugging Face cache), best of 2.
# C. KL: 480 tokens of USAGE.md through bin/ref_forward (f32 activations, dequantized weights; kept in
#    <out>/ref so later runs reuse it) and runz --kl-ref: mean / max KL and how often the top token agrees.
# Without --models: the usual models found in the Hugging Face cache, the data folder's zero/models ($ARC_DATA_DIR,
# else ~/.inferzero) and the folders the server reads (serverz --list-models).
# Every run checks for other GPU programs first (a second program on the GPU spoils the numbers).
# Programs: bin/runz, bin/ref_forward, bin/serverz, bin/mgpu_probe. A relative --out is taken from the inferzero folder,
# as in the .ps1. Bash and the system's tools only (awk, sed, grep, coreutils): no Python, no jq.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNZ="$ROOT/bin/runz"
REF="$ROOT/bin/ref_forward"
SERVERZ="$ROOT/bin/serverz"
PROBE="$ROOT/bin/mgpu_probe"
GPU_PROGRAMS="serverz runz zero zdec ref_forward"

usage() {
  cat <<'EOF'
Benchmarks for the website: prompt sizes, generation modes and KL against an f32 reference (the Linux version of
bench/website_bench.ps1). Writes <out>/results.jsonl and <out>/summary.md.

  bash bench/website_bench.sh [options]

  --models A.gguf B.gguf ...  the models (.gguf), up to the next option (default: the usual models found in the
                              Hugging Face cache, the data folder and the folders the server reads)
  --sizes 128 512 ...         prompt sizes in tokens (default 128 512 2048 8192 32768 131072; 128,512 works too)
  --out DIR                   the results folder (default bench/results/website, from the inferzero folder)
  --gen N                     tokens generated after each prompt size (default 64)
  --cooldown N                seconds of rest before every run (default 20)
  --skip-kl                   no KL against the f32 reference (phase C)
  --skip-modes                no generation modes (phase B)
  --help                      this text
EOF
}

die() { printf '%s\n' "$*" >&2; exit 2; }

# ---- the options
MODELS=()
SIZES=()
SIZES_GIVEN=0
OUT_ARG="bench/results/website"
GEN=64
COOLDOWN=20
SKIP_KL=0
SKIP_MODES=0
add_sizes() {  # (--sizes 128 512 and --sizes 128,512 both work, as -Sizes 128,512 does in the .ps1)
  local v x parts
  IFS=',' read -r -a parts <<< "$1"
  for x in "${parts[@]}"; do
    v="${x//[[:space:]]/}"
    [ -z "$v" ] && continue
    [[ "$v" =~ ^[0-9]+$ ]] || die "--sizes: not a number: $x"
    SIZES+=("$((10#$v))")
  done
}
while [ $# -gt 0 ]; do
  case "$1" in
    --models)
      shift
      n0=${#MODELS[@]}
      while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do MODELS+=("$1"); shift; done
      [ ${#MODELS[@]} -gt "$n0" ] || die "--models: name at least one model"
      ;;
    --sizes)
      shift
      [ "$SIZES_GIVEN" = 1 ] || SIZES=()
      SIZES_GIVEN=1
      n0=0
      while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do add_sizes "$1"; shift; n0=1; done
      [ "$n0" = 1 ] || die "--sizes: name at least one size"
      ;;
    --out)
      [ $# -ge 2 ] || die "--out: a folder is needed"
      OUT_ARG="$2"; shift 2 ;;
    --gen)
      [ $# -ge 2 ] && [[ "$2" =~ ^-?[0-9]+$ ]] || die "--gen: a number is needed"
      GEN=$((10#${2#-})); [ "${2:0:1}" = "-" ] && GEN=$((-GEN)); shift 2 ;;
    --cooldown)
      [ $# -ge 2 ] && [[ "$2" =~ ^-?[0-9]+$ ]] || die "--cooldown: a number is needed"
      COOLDOWN=$((10#${2#-})); [ "${2:0:1}" = "-" ] && COOLDOWN=$((-COOLDOWN)); shift 2 ;;
    --skip-kl) SKIP_KL=1; shift ;;
    --skip-modes) SKIP_MODES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (bash bench/website_bench.sh --help)" ;;
  esac
done
[ "$SIZES_GIVEN" = 1 ] || SIZES=(128 512 2048 8192 32768 131072)

if [ ! -e "$RUNZ" ]; then printf 'bin/runz not found\n' >&2; exit 1; fi
# (model paths as given, from the current folder, else from the inferzero folder, where the .ps1 runs;
#  --out from the inferzero folder)
if [ ${#MODELS[@]} -gt 0 ]; then
  for i in "${!MODELS[@]}"; do
    m="${MODELS[$i]}"
    if [ "${m:0:1}" != "/" ]; then
      if [ -e "$PWD/$m" ] || [ ! -e "$ROOT/$m" ]; then MODELS[$i]="$PWD/$m"; else MODELS[$i]="$ROOT/$m"; fi
    fi
  done
fi
if [ "${OUT_ARG:0:1}" = "/" ]; then OUT="$OUT_ARG"; else OUT="$ROOT/$OUT_ARG"; fi
mkdir -p "$OUT/ref" || exit 1
JSONL="$OUT/results.jsonl"

hub_dir() {  # the Hugging Face cache, as the engine (src/platform.h hf_home_hub) and zero-launcher.sh find it
  if [ -n "${HF_HUB_CACHE:-}" ]; then printf '%s' "$HF_HUB_CACHE"
  elif [ -n "${HF_HOME:-}" ]; then printf '%s/hub' "$HF_HOME"
  elif [ "${XDG_CACHE_HOME:-}" != "" ] && [ "${XDG_CACHE_HOME:0:1}" = "/" ]; then printf '%s/huggingface/hub' "$XDG_CACHE_HOME"
  else printf '%s/.cache/huggingface/hub' "${HOME:-.}"; fi
}
app_dir() {  # the data folder as the engine finds it (src/app_dir.h): ARC_DATA_DIR, else ~/.inferzero
  if [ -n "${ARC_DATA_DIR:-}" ]; then printf '%s' "$ARC_DATA_DIR"; else printf '%s/.inferzero' "${HOME:-.}"; fi
}
HUB=$(hub_dir)
PKG="$(app_dir)/zero/models"

TMP="${TMPDIR:-/tmp}"
Q="$TMP/inferzero-website-q.txt"
KLTEXT="$TMP/inferzero-website-kl.txt"
RUNOUT=$(mktemp "$TMP/inferzero-website-run.XXXXXX") || exit 1
ROWS=$(mktemp "$TMP/inferzero-website-rows.XXXXXX") || exit 1  # (this run's rows, for summary.md)
LISTED=$(mktemp "$TMP/inferzero-website-listed.XXXXXX") || exit 1  # (the server's models: path TAB drafter)
trap 'rm -f "$Q" "$KLTEXT" "$RUNOUT" "$ROWS" "$LISTED"' EXIT

# ---- numbers: as .NET formats them (the .ps1's [string]::Format with the invariant culture), so the lines and
#      tables read the same. The numbers are kept as the decimal text runz printed, rounded as text.
AWK_NUM='
function valid(s) { return s ~ /^([0-9]+\.?[0-9]*|\.[0-9]+)$/ }
function pad(s, w) {
  if (w > 0) { while (length(s) < w) s = " " s }
  else if (w < 0) { while (length(s) < -w) s = s " " }
  return s
}
# {i,w:Fd}: d decimals, midpoints away from zero; nothing (padded) for no value
function fmtf(s, d, w,    neg, p, ip, fp, keep, out, i, c, carry) {
  if (s == "") return pad("", w)
  neg = 0
  if (substr(s, 1, 1) == "-") { neg = 1; s = substr(s, 2) }
  if (s ~ /[eE]/) s = sprintf("%.15f", s + 0)
  p = index(s, ".")
  if (p) { ip = substr(s, 1, p - 1); fp = substr(s, p + 1) } else { ip = s; fp = "" }
  if (ip == "") ip = "0"
  while (length(fp) < d + 1) fp = fp "0"
  keep = ip substr(fp, 1, d)
  if (substr(fp, d + 1, 1) + 0 >= 5) {
    out = ""; carry = 1
    for (i = length(keep); i >= 1; i--) {
      c = substr(keep, i, 1) + carry
      if (c == 10) { c = 0; carry = 1 } else carry = 0
      out = c out
    }
    if (carry) out = "1" out
    keep = out
  }
  ip = substr(keep, 1, length(keep) - d); fp = substr(keep, length(keep) - d + 1)
  sub(/^0+/, "", ip); if (ip == "") ip = "0"
  out = d > 0 ? ip "." fp : ip
  if (neg && out ~ /[1-9]/) out = "-" out
  return pad(out, w)
}
# a number as text: json = 1 as a JSON number (80.0, 1e-05, as Python writes a float), else as "{0}" writes it
# (80, 1E-05); nothing for no value
function gnum(s, json,    p, ip, fp, z, rest, m, e) {
  if (s == "") return ""
  p = index(s, ".")
  if (p) { ip = substr(s, 1, p - 1); fp = substr(s, p + 1) } else { ip = s; fp = "" }
  sub(/^0+/, "", ip); if (ip == "") ip = "0"
  sub(/0+$/, "", fp)
  if (ip == "0" && fp ~ /^0000/ && fp ~ /[1-9]/) {
    match(fp, /^0+/); z = RLENGTH; rest = substr(fp, z + 1); e = z + 1
    m = substr(rest, 1, 1); if (length(rest) > 1) m = m "." substr(rest, 2)
    return m (json ? "e" : "E") "-" (e < 10 ? "0" e : e)
  }
  if (fp == "") return json ? ip ".0" : ip
  return ip "." fp
}
'
fmt_f() { awk -v s="$1" -v d="$2" -v w="${3:-0}" "$AWK_NUM"'BEGIN { printf "%s", fmtf(s, d + 0, w + 0) }'; }
fmt_g() { awk -v s="$1" "$AWK_NUM"'BEGIN { printf "%s", gnum(s, 0) }'; }
jnum() { if [ -z "$1" ]; then printf 'null'; else awk -v s="$1" "$AWK_NUM"'BEGIN { printf "%s", gnum(s, 1) }'; fi; }
# a JSON string: quotes, backslashes and control characters escaped
jstr() {
  JS="$1" LC_ALL=C awk 'BEGIN {
    for (i = 1; i < 32; i++) esc[sprintf("%c", i)] = sprintf("\\u%04x", i)
    esc["\""] = "\\\""; esc["\\"] = "\\\\"; esc["\n"] = "\\n"; esc["\r"] = "\\r"; esc["\t"] = "\\t"
    esc["\b"] = "\\b"; esc["\f"] = "\\f"
    s = ENVIRON["JS"]; out = ""
    for (i = 1; i <= length(s); i++) { c = substr(s, i, 1); out = out ((c in esc) ? esc[c] : c) }
    printf "\"%s\"", out
  }'
}
jstr_or_null() { if [ -z "$1" ]; then printf 'null'; else jstr "$1"; fi; }
# the first group of the extended regex ere in file, or nothing (case matters unless flags is "i")
grp() {
  local file="$1" ere="$2" flags="${3:-}" m
  if [ "$flags" = i ]; then m=$(grep -a -o -i -m1 -E "$ere" "$file" | head -n 1)
  else m=$(grep -a -o -m1 -E "$ere" "$file" | head -n 1); fi
  [ -n "$m" ] || return 0
  if [ "$flags" = i ]; then m=$(printf '%s\n' "$m" | sed -E "s#^${ere}\$#\\1#I")
  else m=$(printf '%s\n' "$m" | sed -E "s#^${ere}\$#\\1#"); fi
  printf '%s' "$m"
}
# a captured number, or nothing when it does not read as one
num() {
  local v
  v=$(grp "$@")
  if [ -n "$v" ] && awk -v s="$v" "$AWK_NUM"'BEGIN { exit valid(s) ? 0 : 1 }'; then printf '%s' "$v"; fi
}
truthy() { [ -n "$1" ] && awk -v s="$1" 'BEGIN { exit (s + 0 != 0) ? 0 : 1 }'; }

# ---- the models
# (also every folder the server reads: model-dirs.txt, e.g. /srv/hf, through serverz --list-models; GitHub #4: models
#  outside the Hugging Face cache were not found and the script stopped at its summary. Its JSON is read by a small
#  scanner here: each entry's "path", "supported" and "dflash_path", one line "path TAB dflash_path" per supported
#  model)
if [ -e "$SERVERZ" ]; then
  "$SERVERZ" --list-models 2>/dev/null | LC_ALL=C awk '
    function hexval(h,    i, v, c) {
      v = 0
      for (i = 1; i <= length(h); i++) { c = index("0123456789abcdef", tolower(substr(h, i, 1))); v = v * 16 + c - 1 }
      return v
    }
    function readstr(    out, c, e, h, v) {  # at the opening quote; leaves i after the closing one
      out = ""; i++
      while (i <= n) {
        c = substr(s, i, 1)
        if (c == "\"") { i++; return out }
        if (c == "\\") {
          e = substr(s, i + 1, 1); i += 2
          if (e == "n") out = out "\n"; else if (e == "t") out = out "\t"; else if (e == "r") out = out "\r"
          else if (e == "b") out = out "\b"; else if (e == "f") out = out "\f"
          else if (e == "u") {
            h = substr(s, i, 4); i += 4; v = hexval(h)
            out = out ((v > 0 && v < 128) ? sprintf("%c", v) : "\\u" h)
          } else out = out e
          continue
        }
        out = out c; i++
      }
      return out
    }
    function setval(v) {
      if (key == "path") path = v; else if (key == "supported") sup = v; else if (key == "dflash_path") dfl = v
      afterkey = 0
    }
    { buf = buf $0 "\n" }
    END {
      s = buf; n = length(s); i = 1; depth = 0; afterkey = 0
      while (i <= n && substr(s, i, 1) ~ /[ \t\r\n]/) i++
      c = substr(s, i, 1)
      if (c == "[") ed = 2; else if (c == "{") ed = 1; else exit
      while (i <= n) {
        c = substr(s, i, 1)
        if (c == "\"") {
          str = readstr()
          if (depth == ed) {
            if (afterkey) setval("s:" str)
            else {
              j = i; while (j <= n && substr(s, j, 1) ~ /[ \t\r\n]/) j++
              if (substr(s, j, 1) == ":") { key = str; afterkey = 1; i = j + 1 }
            }
          }
          continue
        }
        if (c == "{" || c == "[") {
          depth++; afterkey = 0
          if (depth == ed && c == "{") { path = ""; sup = ""; dfl = "" }
          i++; continue
        }
        if (c == "}" || c == "]") {
          if (depth == ed && c == "}" && sup == "l:true" && path != "s:")
            if (substr(path, 1, 2) == "s:") print substr(path, 3) "\t" (substr(dfl, 1, 2) == "s:" ? substr(dfl, 3) : "")
          depth--; i++; continue
        }
        if (c ~ /[A-Za-z0-9.+-]/) {
          lit = ""
          while (i <= n && substr(s, i, 1) ~ /[A-Za-z0-9.+-]/) { lit = lit substr(s, i, 1); i++ }
          if (depth == ed && afterkey) setval("l:" lit)
          continue
        }
        i++
      }
    }' > "$LISTED" 2>/dev/null
fi

lower() { printf '%s' "${1,,}"; }
has_flash_next() {  # a model of the list is a Flash Next file already
  local x
  [ ${#MODELS[@]} -gt 0 ] || return 1
  for x in "${MODELS[@]}"; do case "$(lower "$x")" in *flash-next*) return 0 ;; esac; done
  return 1
}
if [ ${#MODELS[@]} -eq 0 ]; then
  names=("Qwen3.5-4B-UD-Q4_K_XL.gguf" "Qwen3.5-9B-UD-Q4_K_XL.gguf" "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
         "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf"
         "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" "gemma-4-E4B_q4_0-it.gguf"
         "gemma-4-12b-it-qat-q4_0.gguf" "gemma-4-26b-a4b-it-qat-q4_0.gguf" "gemma-4-26B_q4_0-it.gguf"
         "gemma-4-31B_q4_0-it.gguf"
         # (DeepSeek-V4.1-Flash, ISTA's IQ2_XS: the first of six parts; runz finds the others next to it)
         "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf")
  # (every .gguf of the data folder, then of the Hugging Face cache, in name order: the first one of a name is taken,
  #  in any case, as Get-ChildItem -Recurse -Filter does)
  found=$( { [ -d "$PKG" ] && find "$PKG" -iname '*.gguf' 2>/dev/null | LC_ALL=C sort
             [ -d "$HUB" ] && find "$HUB" -iname '*.gguf' 2>/dev/null | LC_ALL=C sort; } )
  for nm in "${names[@]}"; do
    lnm=$(lower "$nm")
    f=$(printf '%s\n' "$found" | awk -v want="$lnm" '{ b = $0; sub(/.*\//, "", b); if (tolower(b) == want) { print; exit } }')
    if [ -z "$f" ] && [ -s "$LISTED" ]; then
      f=$(awk -F '\t' -v want="$lnm" '{ b = $1; sub(/.*[\/\\]/, "", b); if (tolower(b) == want) { print $1; exit } }' "$LISTED")
    fi
    # (one Flash Next file only: the single file or the first of its two parts)
    if [ -n "$f" ]; then
      case "$(lower "$f")" in *flash-next*) has_flash_next && continue ;; esac
      MODELS+=("$f")
    fi
  done
fi
if [ ${#MODELS[@]} -eq 0 ]; then
  printf 'No model found: none of the usual model files is in the Hugging Face cache, %s\n' "$PKG"
  printf 'or a folder the server reads (model-dirs.txt). Name them: bash bench/website_bench.sh --models /path/model.gguf ...\n'
  exit 1
fi

# DFlash2 drafters in the Hugging Face cache, by model
DRAFTER_KEYS=("27B" "35B-A3B" "Flash-Next")
DRAFTER_REPOS=("models--z-lab--Qwen3.8-27B-DFlash2" "models--incoai--Qwen3.6-35B-A3B-DFlash2"
               "models--incoai--Qwen3.8-Flash-Next-DFlash2")
drafter() {
  local model="$1" lm i k sd d
  lm=$(lower "$model")
  for i in "${!DRAFTER_KEYS[@]}"; do
    k=$(lower "${DRAFTER_KEYS[$i]}")
    case "$lm" in
      *"$k"*)
        sd="$HUB/${DRAFTER_REPOS[$i]}/snapshots"
        d=$(find "$sd" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | LC_ALL=C sort | head -n 1)
        if [ -n "$d" ] && [ -e "$d/config.json" ]; then printf '%s' "$d"; return 0; fi
        ;;
    esac
  done
  # (else the drafter the server found for that model, in any folder it reads)
  [ -s "$LISTED" ] || return 0
  d=$(awk -F '\t' -v want="$model" '$1 == want && $2 != "" { print $2; exit }' "$LISTED")
  if [ -n "$d" ] && [ -e "$d/config.json" ]; then printf '%s' "$d"; fi
  return 0
}

gpu_programs_running() {  # another GPU program running, by /proc/<pid>/comm
  local c p n
  for c in /proc/[0-9]*/comm; do
    [ -r "$c" ] || continue
    IFS= read -r n 2>/dev/null < "$c" || continue
    n="${n,,}"
    for p in $GPU_PROGRAMS; do [ "$n" = "$p" ] && return 0; done
  done
  return 1
}
wait_free() {
  while gpu_programs_running; do sleep 30; done
  if [ "$COOLDOWN" -gt 0 ]; then sleep "$COOLDOWN"; fi
}
# run VAR=VALUE... -- ARGS...: one runz call once the GPU is free, its output (stdout and stderr) in $RUNOUT; the
# script's own variables (ARC_BENCH_RUNS, REF_ALL) are set for that call only
run() {
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  wait_free
  env -u ARC_BENCH_RUNS -u REF_ALL ${envs[@]+"${envs[@]}"} "$RUNZ" "$@" > "$RUNOUT" 2>&1
  return 0
}
save() { printf '%s\n' "$1" >> "$JSONL"; }
# a row for summary.md: model test key pp ttft tg fits acc kl kmax top (empty for no value)
row() { local IFS=$'\t'; printf '%s\n' "$*" >> "$ROWS"; }

text=$'<|im_start|>user\nWrite a Python module with a class LRUCache (get, put, a capacity, O(1) operations using '
text+=$'a dict and a doubly linked list), docstrings, and unit tests with unittest.<|im_end|>\n<|im_start|>assistant\n'
text+=$'<think>\n\n</think>\n\n'
printf '%s' "$text" > "$Q"
# (the first 2600 characters of USAGE.md, not bytes: a UTF-8 locale for the cut)
(
  export LC_ALL=C.UTF-8
  usage_text=$(cat "$ROOT/USAGE.md"; printf x)
  usage_text="${usage_text%x}"
  printf '%s' "${usage_text:0:2600}"
) > "$KLTEXT" 2>/dev/null

gpu="unknown GPU"
if [ -e "$PROBE" ]; then
  g=$("$PROBE" 2>&1 | grep -a -i -m1 -E '^[[:space:]]+GPU 0:')
  g="${g%$'\r'}"
  g="${g#"${g%%[![:space:]]*}"}"; g="${g%"${g##*[![:space:]]}"}"
  [ -n "$g" ] && gpu="$g"
fi
# (the commit: a package's bin/PREBUILT.txt names it; a git folder asks git)
commit=""
if [ -e "$ROOT/bin/PREBUILT.txt" ]; then
  commit=$(grep -a -o -m1 -E 'commit [0-9a-f]+' "$ROOT/bin/PREBUILT.txt" | head -n 1)
  commit="${commit#commit }"
fi
if [ -z "$commit" ] && [ -e "$ROOT/.git" ] && command -v git > /dev/null 2>&1; then
  commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null | head -n 1)
fi
[ -n "$commit" ] || commit="unknown"
stamp=$(date +%Y-%m-%dT%H:%M:%S)
printf '# website benchmark %s, commit %s, %s\n' "$stamp" "$commit" "$gpu" >> "$OUT/log.txt"
J_HEAD="\"time\":$(jstr "$stamp"),\"commit\":$(jstr "$commit"),\"gpu\":$(jstr "$gpu")"

for m in "${MODELS[@]}"; do
  if [ ! -e "$m" ]; then printf 'skipped (not found): %s\n' "$m"; continue; fi
  b="${m##*/}"; name="${b%.*}"
  printf '=== %s\n' "$name"
  J_MODEL="$J_HEAD,\"model\":$(jstr "$name")"
  # (a run that is not counted first: a model's first run compiles its kernels (and builds or warms its expert file),
  #  which landed in the first size's generation: a B60's 4B generated 12.8 tok/s at 128 tokens, 61 at 512, issue #4)
  run ARC_BENCH_RUNS=1 -- "$m" --bench-pp 128 --n 8 --must-fit
  # ---- A. prompt sizes
  for n in "${SIZES[@]}"; do
    if [ "$n" -le 8192 ]; then runs=2; else runs=1; fi
    # (--must-fit: a context beyond the GPU share stops at load, so the size is marked "does not fit" instead of
    #  hours of paging: Qwen3.8 27B at 131K on a 16.5 GiB share ran at 8.6 instead of ~120 tok/s)
    run "ARC_BENCH_RUNS=$runs" -- "$m" --bench-pp "$n" --n "$GEN" --must-fit
    nf=$(num "$RUNOUT" 'does not fit: ([0-9.]+) GiB' i)
    if [ -n "$nf" ]; then
      save "{$J_MODEL,\"test\":\"size\",\"tokens\":$n,\"pp\":null,\"tg\":null,\"ttft_s\":null,\"fits\":false,\"need_gib\":$(jnum "$nf"),\"error\":\"does not fit in the GPU share\"}"
      row "$name" size "$n" "" "" "" false "" "" "" ""
      printf '  %7s tokens: does not fit in the GPU share (%s GiB needed)\n' "$n" "$nf"
      continue
    fi
    # (every pass's prompt speed, the best kept; as in the .ps1, no pass or one pass at 0 gives no prompt speed)
    pp=$(grep -a -o -E 'pp[0-9]+ run [0-9]+: [0-9.]+ ms[[:space:]]+->[[:space:]]+[0-9.]+ tokens/s' "$RUNOUT" |
         sed -E 's#.*->[[:space:]]+([0-9.]+) tokens/s$#\1#' |
         awk "$AWK_NUM"'valid($0) { c++; if (c == 1 || $0 + 0 > best + 0) best = $0 }
                        END { if (c > 1 || (c == 1 && best + 0 != 0)) print best }')
    tg=$(num "$RUNOUT" 'tg[0-9]+: [0-9.]+ ms[[:space:]]+->[[:space:]]+([0-9.]+) tokens/s')
    err=""
    ttft=""
    if truthy "$pp"; then
      ttft=$(awk -v n="$n" -v p="$pp" 'BEGIN { printf "%.2f", n / p }')
    else
      err=$(grep -a -i -m1 -E 'error|cannot|failed|exception|out of' "$RUNOUT" | head -n 1)
      err="${err%$'\r'}"
    fi
    save "{$J_MODEL,\"test\":\"size\",\"tokens\":$n,\"pp\":$(jnum "$pp"),\"tg\":$(jnum "$tg"),\"ttft_s\":$(jnum "$ttft"),\"error\":$(jstr_or_null "$err")}"
    row "$name" size "$n" "$pp" "$ttft" "$tg" "" "" "" "" ""
    if truthy "$pp"; then first=$(awk -v n="$n" -v p="$pp" 'BEGIN { printf "%.15g", n / p }'); else first=0; fi
    printf '  %7s tokens: prompt %s tok/s, first token %s s, generation %s tok/s %s\n' "$n" "$(fmt_f "$pp" 1 8)" \
      "$(fmt_f "$first" 2 7)" "$(fmt_f "$tg" 2 6)" "$err"
  done
  # ---- B. generation modes
  if [ "$SKIP_MODES" = 0 ]; then
    d=$(drafter "$m")
    mode_names=(plain mtp)
    mode_args=("" "--mtp")
    if [ -n "$d" ]; then
      mode_names+=(dflash mtp+dflash)
      mode_args+=("--dflash"$'\n'"$d"$'\n'"--draft"$'\n'"dflash" "--dflash"$'\n'"$d"$'\n'"--draft"$'\n'"auto")
    fi
    for mi in "${!mode_names[@]}"; do
      k="${mode_names[$mi]}"
      extra=()
      [ -n "${mode_args[$mi]}" ] && mapfile -t extra <<< "${mode_args[$mi]}"
      best=""
      acc=""
      for _ in 1 2; do
        run -- "$m" --text-file "$Q" --n 256 ${extra[@]+"${extra[@]}"}
        t=$(num "$RUNOUT" 'generation: [0-9]+ tokens in [0-9.]+ ms \(([0-9.]+) t/s\)')
        if truthy "$t" && { [ -z "$best" ] || awk -v a="$t" -v b="$best" 'BEGIN { exit (a + 0 > b + 0) ? 0 : 1 }'; }; then
          best="$t"
          acc=$(num "$RUNOUT" '\(([0-9.]+) %\), [0-9.]+ tokens per main-model pass')
        fi
      done
      if [ "$k" = mtp ] && [ -z "$best" ]; then continue; fi  # (no MTP block)
      save "{$J_MODEL,\"test\":\"mode\",\"mode\":$(jstr "$k"),\"tg\":$(jnum "$best"),\"accepted_pct\":$(jnum "$acc")}"
      row "$name" mode "$k" "" "" "$best" "" "$acc" "" "" ""
      if truthy "$acc"; then accs="($(fmt_g "$acc") % of drafts accepted)"; else accs=""; fi
      printf '  %-11s %s tok/s %s\n' "$k" "$(fmt_f "$best" 2 6)" "$accs"
    done
  fi
  # ---- C. KL against f32
  if [ "$SKIP_KL" = 0 ] && [ -e "$REF" ]; then
    ids="$OUT/ref/$name.ids.txt"
    rb="$OUT/ref/$name.f32.bin"
    if [ ! -e "$rb" ]; then
      run -- "$m" --text-file "$KLTEXT" --dump-ids "$ids"
      if [ -e "$ids" ]; then
        t=$(cat "$ids")
        t="${t#$'\xef\xbb\xbf'}"
        t="${t#"${t%%[![:space:]]*}"}"; t="${t%"${t##*[![:space:]]}"}"
        t=$(printf '%s\n' "$t" | cut -d, -f1-480)
        printf '%s' "$t" > "$ids"
        wait_free
        env -u ARC_BENCH_RUNS REF_ALL=1 "$REF" "$m" "@$ids" "$rb" > /dev/null 2>&1
      fi
    fi
    if [ -e "$rb" ]; then
      run -- "$m" --tokens-file "$ids" --kl-ref "$rb"
      kl=$(num "$RUNOUT" 'KL\(ref \|\| inferzero\) = ([0-9.]+)')
      kmax=$(num "$RUNOUT" 'max ([0-9.]+)\)')
      agree=$(grep -a -o -m1 -E 'top token agrees [0-9]+ / [0-9]+' "$RUNOUT" | head -n 1)
      top=""
      if [ -n "$agree" ]; then
        top=$(printf '%s\n' "$agree" | awk '{ if ($6 + 0 != 0) printf "%.1f", 100.0 * $4 / $6 }')
      fi
      # (without the reference's near-ties, the rule of 2026-11-24: runz prints it after the strict share)
      topnt=""
      ntl=$(grep -a -o -m1 -E 'without near-ties \(ref top-2 margin < [0-9.]+\): [0-9]+ / [0-9]+' "$RUNOUT" | head -n 1)
      if [ -n "$ntl" ]; then
        topnt=$(printf '%s\n' "$ntl" | awk '{ if ($NF + 0 != 0) printf "%.1f", 100.0 * $(NF - 2) / $NF }')
      fi
      save "{$J_MODEL,\"test\":\"kl\",\"kl\":$(jnum "$kl"),\"kl_max\":$(jnum "$kmax"),\"top_agree_pct\":$(jnum "$top"),\"top_agree_nt_pct\":$(jnum "$topnt")}"
      row "$name" kl "" "" "" "" "" "" "$kl" "$kmax" "$top" "$topnt"
      printf '  KL against f32: mean %s, max %s, top token agrees %s %% (%s %% without near-ties)\n' "$(fmt_g "$kl")" "$(fmt_g "$kmax")" "$(fmt_g "$top")" "$(fmt_g "$topnt")"
    else
      printf '  KL: no f32 reference (ref_forward failed or does not take this model)\n'
    fi
  fi
done
rm -f "$Q" "$KLTEXT"

# ---- summary.md from every row of this run
SUM_GPU="$gpu" SUM_COMMIT="$commit" SUM_STAMP="$stamp" awk -F '\t' -v sizes="${SIZES[*]}" "$AWK_NUM"'
  function truthy(s) { return s != "" && s + 0 != 0 }
  {
    model = $1; test = $2; key = $3
    if (!(model in seen)) { seen[model] = 1; models[++nm] = model }
    id = model SUBSEP test SUBSEP key
    if (test != "kl" && !(id in has)) {
      has[id] = 1; pp[id] = $4; ttft[id] = $5; tg[id] = $6; fits[id] = $7; acc[id] = $8
    }
    if (test == "kl") { nk++; km[nk] = model; kl[nk] = $9; kx[nk] = $10; kt[nk] = $11; kn[nk] = $12 }
  }
  END {
    ns = split(sizes, sz, " ")
    head = "| Model |"; rule = "|---|"
    for (i = 1; i <= ns; i++) { head = head " " sz[i] " |"; rule = rule (i > 1 ? "|" : "") "---" }
    rule = rule "|"
    print "# inferzero benchmarks"; print ""
    print ENVIRON["SUM_GPU"] "; commit " ENVIRON["SUM_COMMIT"] "; " ENVIRON["SUM_STAMP"] "."; print ""
    print "## Prompt processing (tok/s) and time to first token"; print ""; print head; print rule
    for (j = 1; j <= nm; j++) {
      line = "| " models[j] " |"
      for (i = 1; i <= ns; i++) {
        id = models[j] SUBSEP "size" SUBSEP sz[i]
        if ((id in has) && truthy(pp[id])) c = fmtf(pp[id], 0, 0) " (" fmtf(ttft[id], 1, 0) " s)"
        else if ((id in has) && fits[id] == "false") c = "does not fit"
        else c = "-"
        line = line " " c " |"
      }
      print line
    }
    print ""; print "## Generation (tok/s) at that context depth, plain decoding"; print ""; print head; print rule
    for (j = 1; j <= nm; j++) {
      line = "| " models[j] " |"
      for (i = 1; i <= ns; i++) {
        id = models[j] SUBSEP "size" SUBSEP sz[i]
        if ((id in has) && truthy(tg[id])) c = fmtf(tg[id], 1, 0)
        else if ((id in has) && fits[id] == "false") c = "does not fit"
        else c = "-"
        line = line " " c " |"
      }
      print line
    }
    print ""; print "## Generation modes (256-token coding answer, tok/s)"; print ""
    print "| Model | plain | MTP | DFlash2 | MTP + DFlash2 |"; print "|---|---|---|---|---|"
    nmo = split("plain mtp dflash mtp+dflash", mo, " ")
    for (j = 1; j <= nm; j++) {
      line = "| " models[j] " |"
      for (i = 1; i <= nmo; i++) {
        id = models[j] SUBSEP "mode" SUBSEP mo[i]
        if ((id in has) && truthy(tg[id])) c = fmtf(tg[id], 1, 0) (truthy(acc[id]) ? " (" gnum(acc[id], 0) " %)" : "")
        else c = "-"
        line = line " " c " |"
      }
      print line
    }
    print ""; print "## Accuracy: KL against an f32 reference (480 tokens of USAGE.md)"; print ""
    print "| Model | KL mean | KL max | top token agrees | without near-ties |"; print "|---|---|---|---|---|"
    for (i = 1; i <= nk; i++)
      print "| " km[i] " | " fmtf(kl[i], 5, 0) " | " fmtf(kx[i], 4, 0) " | " fmtf(kt[i], 1, 0) " % | " (kn[i] == "" ? "-" : fmtf(kn[i], 1, 0) " %") " |"
    print ""; print "(without near-ties: positions whose reference top two are within 0.08 nats left out, ARC_KL_TIE; the KL counts every position)"
  }' "$ROWS" > "$OUT/summary.md"
printf 'summary: %s\n' "$OUT_ARG/summary.md"
exit 0  # (not runz's last exit code: 3 when the last size did not fit)
