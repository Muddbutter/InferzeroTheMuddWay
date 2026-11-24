#!/usr/bin/env bash
# inferzero on Linux: checks the system, lists the catalog's models (downloaded or not), downloads one into the Hugging
# Face cache with curl, starts bin/serverz and opens the web UI. Bash and the system's tools only (curl, sha256sum):
# no Python, as the rest of inferzero. The Windows launcher is zero-launcher.cmd.
#   ./zero-launcher.sh                    the menu
#   ./zero-launcher.sh --check            what the system has and lacks (GPU runtime, device access, CPU, RAM)
#   ./zero-launcher.sh --list             the catalog, with what is downloaded
#   ./zero-launcher.sh --download 4b      download a catalog model (no server)
#   ./zero-launcher.sh --model 4b         download if needed, then start the server (also a path to a .gguf)
#   ./zero-launcher.sh --model 4b --install-service   a systemd user service for it instead of starting it here
#   ./zero-launcher.sh --options          the speed and accuracy options (scripts/options.tsv) and what is chosen
#   ./zero-launcher.sh --opt copy_draft --no-opt xcpu --opt budget=12 --model 27b   choose some (kept for later starts)
# Options: --port N (8090), --host H (127.0.0.1), --ctx N (a context of N tokens; default auto, the largest that fits),
# --gpus N, --mtp (with --model FILE), --think-budget N (8192; -1 no limit), --session-cache-gb N (8; 0 off),
# --chunk N (default: the server's), --no-auto (not scripts/best-setups.tsv's best measured settings for this PC),
# --no-browser, --yes (no questions), --log-dir D (the server's log folder, default ~/.inferzero/logs; also ARC_LOG_DIR), --logs (the log's
# last lines), --model-dirs / --add-model-dir D / --remove-model-dir D (folders searched for .gguf models),
# --service-name NAME (inferzero), --replace-service, --opt ID[=VALUE] / --no-opt ID / --default-opt ID
# (comma lists too; ~/.inferzero/options.json, the same as zero-server's on Windows; --no-saved-options: only the
# ones given, nothing saved); anything after `--` goes to
# serverz as it is (e.g. -- --slots 2; not --port / --host, which the launcher must know); ./bin/serverz --help all lists
# the server's options.
# The catalog below follows scripts/model-catalog.ps1 (tests/check_catalog_sync.sh checks that they agree). Flash
# Next's AutoRound import and the speech models are setup.ps1's on Windows and not offered here.
# (A symlink to this file, e.g. in ~/bin, is followed to the package; a relative --model path stays relative to where
#  you ran it.)

SELF=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || printf '%s' "${BASH_SOURCE[0]}")
ROOT=$(cd "$(dirname "$SELF")" && pwd)
SERVERZ="$ROOT/bin/serverz"

# ---- the catalog: key|name|repo|GB|disk GB|RAM GB|mtp|files (space-separated)|drafter repo|drafter file|extra repo|extra file|extra into|words
# (words: the model's, as serverz --list-models "opts" gives them (dense moe moe_std qsa flashnext mtp h8) and vision, for the
#  switches; df=NAME: a DFlash2 drafter for it is a cache repository whose name holds NAME and DFlash)
CATALOG=(
"4b|Qwen3.5 4B (fastest; MTP, reads images)|unsloth/Qwen3.5-4B-MTP-GGUF|3.7||16|1|Qwen3.5-4B-UD-Q4_K_XL.gguf mmproj-F16.gguf||||||dense mtp h8 vision"
"9b|Qwen3.5 9B (MTP, reads images)|unsloth/Qwen3.5-9B-MTP-GGUF|7.1||16|1|Qwen3.5-9B-UD-Q4_K_XL.gguf mmproj-F16.gguf||||||dense mtp h8 vision"
"35b|Qwen3.6 35B-A3B (fast mixture of experts; MTP, reads images)|unsloth/Qwen3.6-35B-A3B-MTP-GGUF|19.1||32|1|Qwen3.6-35B-A3B-UD-IQ4_XS.gguf mmproj-F16.gguf||||||moe moe_std mtp vision df=Qwen3.6-35B"
"27b|Qwen3.8 27B (best answers, slowest; MTP, reads images)|ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF|13.1||32|1|Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf mmproj-Qwen3.8-27B-BF16.gguf||||||dense mtp h8 vision df=Qwen3.8-27B"
"flash|Qwen3.8 Flash Next (512 experts on the SSD; reads images)|ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF|84.5|141|32|0|IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf mmproj-Qwen3.8-Flash-Next-f16.gguf|||bartowski/Qwen3.8-Flash-Next-GGUF|mmproj-Qwen3.8-Flash-Next-f16.gguf|IQ3_S|moe qsa flashnext vision df=Qwen3.8-Flash-Next"
"g12|Gemma 4 12B (reads images; drafter)|google/gemma-4-12B-it-qat-q4_0-gguf|7.4||32|1|gemma-4-12b-it-qat-q4_0.gguf mmproj-gemma-4-12b-it-qat-q4_0.gguf mtp-gemma-4-12B-it.gguf|unsloth/gemma-4-12B-it-qat-GGUF|mtp-gemma-4-12B-it.gguf||||dense mtp vision"
"g31|Gemma 4 31B (Google's largest dense Gemma; slow; drafter)|google/gemma-4-31B-it-qat-q4_0-gguf|19.1||32|1|gemma-4-31B_q4_0-it.gguf gemma-4-31B-it-mmproj.gguf mtp-gemma-4-31B-it.gguf|unsloth/gemma-4-31B-it-qat-GGUF|mtp-gemma-4-31B-it.gguf||||dense mtp vision"
"e4b|Gemma 4 E4B (images, audio; drafter)|google/gemma-4-E4B-it-qat-q4_0-gguf|6.2||16|1|gemma-4-E4B_q4_0-it.gguf gemma-4-E4B-it-mmproj.gguf mtp-gemma-4-E4B-it.gguf|unsloth/gemma-4-E4B-it-qat-GGUF|mtp-gemma-4-E4B-it.gguf||||dense mtp vision"
)
# the cache variables the server and the downloader read (a systemd unit gets them too); never HF_TOKEN
CACHE_ENV=(HF_HUB_CACHE HF_HOME XDG_CACHE_HOME HF_TOKEN_PATH)

say() { printf '%s\n' "$*"; }
err() { printf '%s\n' "$*" >&2; }

# m_field KEY N: field N (0-based) of a catalog entry
m_field() {
  local e
  for e in "${CATALOG[@]}"; do
    if [ "${e%%|*}" = "$1" ]; then IFS='|' read -r -a _f <<< "$e"; printf '%s' "${_f[$2]}"; return 0; fi
  done
  return 1
}
m_known() { m_field "$1" 0 >/dev/null; }

hub_dir() {  # the Hugging Face cache, as the engine (src/platform.h hf_home_hub) finds it
  if [ -n "${HF_HUB_CACHE:-}" ]; then printf '%s' "$HF_HUB_CACHE"
  elif [ -n "${HF_HOME:-}" ]; then printf '%s/hub' "$HF_HOME"
  elif [ "${XDG_CACHE_HOME:-x}" != x ] && [ "${XDG_CACHE_HOME:0:1}" = "/" ]; then printf '%s/huggingface/hub' "$XDG_CACHE_HOME"
  else printf '%s/.cache/huggingface/hub' "$HOME"; fi
}
repo_root() { printf '%s/models--%s' "$(hub_dir)" "${1//\//--}"; }

snapshots() {  # the repository's snapshot folders: the one refs/main points at first, then the others
  local root main d
  root=$(repo_root "$1")
  if [ -f "$root/refs/main" ]; then
    main="$root/snapshots/$(tr -d '[:space:]' < "$root/refs/main")"
    [ -d "$main" ] && printf '%s\n' "$main"
  fi
  [ -d "$root/snapshots" ] || return 0
  for d in "$root"/snapshots/*/; do
    d=${d%/}
    [ -d "$d" ] && [ "$d" != "${main:-}" ] && printf '%s\n' "$d"
  done
}

has_file() {  # snapshot, path: there, or (a file copied in from another repository) by its name anywhere inside
  [ -f "$1/$2" ] && return 0
  [ -n "$(find "$1" -name "$(basename "$2")" -type f -print -quit 2>/dev/null)" ]
}

model_path() {  # the first file of a downloaded catalog model (a snapshot that holds all its files), or nothing
  local key=$1 repo files d f ok
  repo=$(m_field "$key" 2); files=$(m_field "$key" 7)
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    ok=1
    for f in $files; do has_file "$d" "$f" || { ok=0; break; }; done
    if [ $ok -eq 1 ]; then printf '%s/%s' "$d" "${files%% *}"; return 0; fi
  done < <(snapshots "$repo")
  return 1
}

ram_gb() { awk '/^MemTotal:/ { printf "%d", $2 / 1048576 }' /proc/meminfo 2>/dev/null || printf '0'; }

have_lib() {  # a shared library the dynamic loader finds
  ldconfig -p 2>/dev/null | grep -q "$1" && return 0
  local d
  for d in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/lib ${LD_LIBRARY_PATH//:/ }; do
    [ -e "$d/$1" ] && return 0
  done
  return 1
}

# check: one line per need, "ok    ..." or "MISS  ..."; returns 1 when something is missing
CHECK_LINES=()
run_check() {
  CHECK_LINES=()
  local bad=0 node nodes=() n
  _c() { if [ "$1" -eq 0 ]; then CHECK_LINES+=("  ok    $2"); else CHECK_LINES+=("  MISS  $2"); bad=1; fi; }
  [ "$(uname -s)" = Linux ]; _c $? "Linux (this launcher is for Linux; on Windows use zero-launcher.cmd)"
  [ -x "$SERVERZ" ]; _c $? "bin/serverz built ($SERVERZ; ./build.sh builds it)"
  grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | grep -qw avx2 && grep -m1 '^flags' /proc/cpuinfo | grep -qw fma &&
    grep -m1 '^flags' /proc/cpuinfo | grep -qw f16c
  _c $? "CPU with AVX2, FMA and F16C"
  have_lib libze_loader.so.1; _c $? "libze_loader.so.1 (sudo apt install libze1)"
  have_lib libOpenCL.so.1; _c $? "libOpenCL.so.1 (sudo apt install ocl-icd-libopencl1)"
  have_lib libgomp.so.1; _c $? "libgomp.so.1 (sudo apt install libgomp1)"
  ls /etc/OpenCL/vendors/ 2>/dev/null | grep -qi intel
  _c $? "Intel's OpenCL driver (/etc/OpenCL/vendors/intel*.icd; sudo apt install intel-opencl-icd libze-intel-gpu1, or Intel's packages for Battlemage cards: docs/linux-port.md)"
  for n in /dev/dri/renderD*; do [ -e "$n" ] && nodes+=("$n"); done
  [ ${#nodes[@]} -eq 0 ] && [ -e /dev/dxg ] && nodes=(/dev/dxg)  # (WSL2: the Windows driver's GPU, no render node)
  [ ${#nodes[@]} -gt 0 ]; _c $? "a GPU render node (/dev/dri/renderD*, or /dev/dxg in WSL2)"
  if [ ${#nodes[@]} -gt 0 ]; then
    local acc=1
    for node in "${nodes[@]}"; do [ -r "$node" ] && [ -w "$node" ] && acc=0; done
    _c $acc "access to the render node (sudo usermod -aG render \$USER, then log in again)"
  fi
  local r; r=$(ram_gb)
  [ "${r:-0}" -ge 15 ]; _c $? "RAM ${r} GB (16 GB for the 4B / 9B / E4B, 32 GB for the others)"
  return $bad
}
print_check() { local rc=0; run_check || rc=1; printf '%s\n' "${CHECK_LINES[@]}"; return $rc; }

# ---- downloads, with curl into the Hugging Face cache layout (snapshots/<commit>/<file>, refs/main); each file's
# size and, for large (LFS) files, its SHA-256 from the repository's own headers, checked before it takes its name
hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then printf '%s' "$HF_TOKEN"; return; fi
  local f="${HF_HOME:-$HOME/.cache/huggingface}/token"
  [ -f "$f" ] && tr -d '[:space:]' < "$f"
}
# hf_head REPO REV FILE: the headers of the file's first answer (not followed), lower-cased names
hf_head() {
  local tok a=(-sS -I --max-time 60 --noproxy '' -A 'inferzero-launcher')
  tok=$(hf_token)
  [ -n "$tok" ] && a+=(-H "Authorization: Bearer $tok")
  curl "${a[@]}" "https://huggingface.co/$1/resolve/$2/$3" 2>/dev/null | tr -d '\r' | sed 's/^\([^:]*\):/\L\1:/'
}
hdr() { sed -n "s/^$1: *//p" | head -1 | tr -d '"'; }

safe_rel() {  # a repository path that stays inside the snapshot (no .., no absolute path)
  case "/$1/" in */../*|*//*) return 1;; esac
  [ "${1:0:1}" != "/" ]
}

# hf_get REPO FILE...: downloads into the cache; prints the snapshot folder on the last line
hf_get() {
  local repo=$1; shift
  local h commit f size sha snap dst part have tok need=0 free
  h=$(hf_head "$repo" main "$1")
  commit=$(printf '%s\n' "$h" | hdr x-repo-commit)
  if [ -z "$commit" ]; then
    case "$(printf '%s\n' "$h" | head -1)" in
      *" 401"*|*" 403"*) err "$repo needs you to sign in and accept its terms on huggingface.co first, then a token: set HF_TOKEN (a read token from https://huggingface.co/settings/tokens) and run this again";;
      *) err "cannot reach huggingface.co for $repo (or it has no file $1): check the internet connection, then run this again";;
    esac
    return 1
  fi
  snap="$(repo_root "$repo")/snapshots/$commit"
  mkdir -p "$snap" "$(repo_root "$repo")/refs" || return 1
  printf '%s' "$commit" > "$(repo_root "$repo")/refs/main"
  local -A SIZE SHA
  for f in "$@"; do
    safe_rel "$f" || { err "$f is not a path inside the repository"; return 1; }
    h=$(hf_head "$repo" "$commit" "$f")
    size=$(printf '%s\n' "$h" | hdr x-linked-size); [ -n "$size" ] || size=$(printf '%s\n' "$h" | hdr content-length)
    sha=$(printf '%s\n' "$h" | hdr x-linked-etag)
    [ -n "$size" ] || { err "$repo has no file $f"; return 1; }
    SIZE[$f]=$size; SHA[$f]=$sha
    dst="$snap/$f"; have=0
    if [ -f "$dst" ]; then have=$(stat -c %s "$dst"); elif [ -f "$dst.part" ]; then have=$(stat -c %s "$dst.part"); fi
    [ "$have" -lt "$size" ] && need=$((need + size - have))
  done
  free=$(df -PB1 "$snap" 2>/dev/null | awk 'NR == 2 { print $4 }')
  if [ -n "$free" ] && [ "$need" -gt "$free" ]; then
    err "$(awk -v n="$need" -v f="$free" -v p="$snap" 'BEGIN { printf "these files need %.1f GB more, and %s has %.1f GB free: make room, or set HF_HOME to a folder on a larger drive", n / 1e9, p, f / 1e9 }')"
    return 1
  fi
  [ "$need" -gt 0 ] && awk -v n="$need" -v f="${free:-0}" 'BEGIN { printf "    %.1f GB to download (%.1f GB free)\n", n / 1e9, f / 1e9 }' >&2
  tok=$(hf_token)
  for f in "$@"; do
    dst="$snap/$f"; part="$dst.part"; size=${SIZE[$f]}
    mkdir -p "$(dirname "$dst")"
    if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ]; then err "    $f is there"; continue; fi
    # (one download of a file at a time: a second one writing the same .part appended its bytes to the first's, a
    #  16.67 GB file grew to 17.8 GB and was downloaded again from nothing, 2026-11-21. flock on <file>.part.lock,
    #  let go when the process ends, however it ends; without flock, no lock)
    local lk=""
    if command -v flock >/dev/null; then
      exec {lk}>"$part.lock"
      if ! flock -n "$lk"; then
        exec {lk}>&-
        err "$f is being downloaded by another window or program now: let that one finish, then run this again"; return 1
      fi
    fi
    have=0; [ -f "$part" ] && have=$(stat -c %s "$part")
    if [ "$have" -gt "$size" ]; then rm -f "$part"; have=0; fi
    if [ "$have" -lt "$size" ]; then
      awk -v f="$f" -v s="$size" -v h="$have" 'BEGIN { printf "    %s (%.2f GB%s)\n", f, s / 1e9, h ? sprintf(", resuming at %.2f GB", h / 1e9) : "" }' >&2
      # (the tries made here, not by curl's --retry: with -C - curl's own retry resumed from the offset it found at its
      #  start, so the bytes between it and a cut were written twice. Each try resumes from the .part's length measured
      #  just before it. --speed-time: a connection that stalls ends with an error instead of waiting forever)
      local code=0 try a
      for try in 1 2 3 4 5 6; do
        have=0; [ -f "$part" ] && have=$(stat -c %s "$part")
        [ "$have" -ge "$size" ] && break
        a=(-L --fail --speed-limit 1 --speed-time 120 --progress-bar -A 'inferzero-launcher' -o "$part")
        [ "$have" -gt 0 ] && a+=(-C "$have")
        [[ "${ARC_HF_RATE:-}" =~ ^[0-9]+[KkMmGg]?$ ]] && a+=(--limit-rate "$ARC_HF_RATE")
        [ -n "$tok" ] && a+=(-H "Authorization: Bearer $tok")
        code=0; curl "${a[@]}" "https://huggingface.co/$repo/resolve/$commit/$f" || code=$?
        [ "$code" -eq 0 ] && break
        # (22: --fail's HTTP error, such as a 401 / 404: trying again does not help)
        { [ "$code" -eq 22 ] || [ "$try" -eq 6 ]; } && break
        err "    the download stopped (curl exit $code): trying again in 3 s ($try of 5)"
        sleep 3
      done
      if [ "$code" -ne 0 ]; then
        [ -n "$lk" ] && { exec {lk}>&-; rm -f "$part.lock"; }
        err "downloading $f failed (curl exit $code): run this again, it goes on where it stopped"; return 1
      fi
    fi
    [ -n "$lk" ] && { exec {lk}>&-; rm -f "$part.lock"; }
    have=$(stat -c %s "$part")
    [ "$have" = "$size" ] || { err "$f is $have bytes, the repository says $size: run this again"; return 1; }
    if [ ${#SHA[$f]} -eq 64 ]; then
      err "    checking $f (SHA-256)..."
      if [ "$(sha256sum "$part" | cut -d' ' -f1)" != "${SHA[$f],,}" ]; then
        rm -f "$part"; err "$f arrived damaged (SHA-256 differs): run this again to download it anew"; return 1
      fi
    fi
    mv -f "$part" "$dst"
  done
  printf '%s\n' "$snap"
}

copy_into() {  # atomic: a copy stopped halfway never looks like the finished file
  cp -f "$1" "$2.incomplete" && mv -f "$2.incomplete" "$2"
}

download() {  # KEY YES
  local key=$1 yes=$2 name repo files gb own f snap dr df xr xf xi xs src
  name=$(m_field "$key" 1); repo=$(m_field "$key" 2); gb=$(m_field "$key" 3); files=$(m_field "$key" 7)
  dr=$(m_field "$key" 8); df=$(m_field "$key" 9); xr=$(m_field "$key" 10); xf=$(m_field "$key" 11); xi=$(m_field "$key" 12)
  command -v curl >/dev/null || { err "curl not found (sudo apt install curl)"; return 1; }
  if [ "$yes" != 1 ]; then
    local ans; read -r -p "Download $name (~$gb GB) into $(hub_dir)? [y/N] " ans || return 1
    [ "${ans,,}" = y ] || return 1
  fi
  own=()
  for f in $files; do [ "$f" = "$df" ] || [ "$f" = "$xf" ] || own+=("$f"); done
  snap=$(hf_get "$repo" "${own[@]}" | tail -1) || return 1
  [ -d "$snap" ] || return 1
  # (a file from another repository copied next to the model, where the engine looks for it: Gemma's drafter, Flash
  #  Next's image projector)
  if [ -n "$dr" ]; then
    xs=$(hf_get "$dr" "$df" | tail -1) || return 1
    copy_into "$xs/$df" "$snap/$(basename "$df")" || return 1
  fi
  if [ -n "$xr" ]; then
    xs=$(hf_get "$xr" "$xf" | tail -1) || return 1
    mkdir -p "$snap/$xi"; copy_into "$xs/$xf" "$snap/$xi/$(basename "$xf")" || return 1
  fi
  if ! model_path "$key" >/dev/null; then
    err "$name: some of its files are still missing (a new commit of the repository during the download?); run it again"
    return 1
  fi
}

# ---- the server
exposed() { case "$1" in ''|127.0.0.1|localhost|::1) return 1;; esac; return 0; }
port_busy() {  # HOST PORT: something already answers there (another server would answer our /health)
  local h=${1:-127.0.0.1}
  [ "$h" = 0.0.0.0 ] && h=127.0.0.1
  [ "$h" = "::" ] && h=::1
  (exec 3<>"/dev/tcp/$h/$2") 2>/dev/null
}

# ---- speed and accuracy options (QUEUE O3, 2026-11-21): the catalog scripts/options.tsv (one row an option: id, kind
# env / flag, its switch, the on and off values, the default, a plain title, what it does, the PCs it is for, its cost,
# same output or KL ...) and the choices in options.json, the same file zero-server.ps1 keeps on Windows
# ({"copy_draft": "on", "xcpu": "off", "budget": "12"}; an option not in it keeps the engine's default). An env option
# goes into the server's environment, a flag onto its command line; --install-service writes both into the unit
STATE_DIR=${ARC_DATA_DIR:-$HOME/.inferzero}
# (the server's log: <logs folder>/server.log, earlier runs next to it (src/server_log.h); --log-dir or ARC_LOG_DIR)
LOG_DIR=${ARC_LOG_DIR:-}
logs_dir() { printf '%s' "${LOG_DIR:-$STATE_DIR/logs}"; }
OPT_TSV="$ROOT/scripts/options.tsv"
declare -A OPTC=()
OPT_ENV=(); OPT_FLAGS=(); OPT_WORDS=()
opt_ids() { [ -f "$OPT_TSV" ] && awk -F'\t' 'NR > 1 && $1 != "" { print $1 }' "$OPT_TSV"; }
opt_known() { [ -n "$1" ] && opt_ids | grep -qxF -- "$1"; }
opt_col() {  # ID COLUMN: that option's field, by the header's column name
  awk -F'\t' -v id="$1" -v col="$2" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == col) c = i; next }
    $1 == id { if (c) print $c; exit }' "$OPT_TSV" | tr -d '\r'
}
opt_load() {  # options.json into OPTC (a flat object of strings: split on commas and braces, then "key": "value")
  OPTC=()
  local f="$STATE_DIR/options.json" k v
  [ -f "$f" ] || return 0
  while IFS=$'\t' read -r k v; do
    [ -n "$k" ] && OPTC[$k]=$v
  done < <(tr ',{}' '\n\n\n' < "$f" | sed -n 's/^[[:space:]]*"\([^"\\]*\)"[[:space:]]*:[[:space:]]*"\([^"\\]*\)"[[:space:]]*$/\1\t\2/p')
}
opt_save() {  # OPTC to options.json (atomic: a stopped write never leaves half a file)
  local f="$STATE_DIR/options.json" tmp k first=1
  mkdir -p "$STATE_DIR" || return 1
  tmp=$(mktemp "$STATE_DIR/.options-XXXXXX") || return 1
  {
    printf '{'
    for k in $(printf '%s\n' "${!OPTC[@]}" | sort); do
      [ $first = 1 ] || printf ','
      first=0
      printf '\n    "%s":  "%s"' "$k" "${OPTC[$k]}"
    done
    printf '\n}\n'
  } > "$tmp" && mv -f "$tmp" "$f" || { rm -f "$tmp"; return 1; }
}
opt_set() {  # ID CHOICE: on, off, a value, or "" (the default); refuses what the catalog does not allow
  local id=$1 c=$2 on
  opt_known "$id" || { err "zero-launcher: no option \"$id\" (the options: $(opt_ids | tr '\n' ' '); --options)"; return 2; }
  on=$(opt_col "$id" on)
  if [ -z "$c" ]; then unset "OPTC[$id]"; return 0; fi
  case "$c" in *[\"\\]*|*[[:cntrl:]]*) err "zero-launcher: --opt $id: a value without quotes, backslashes or control characters"; return 2;; esac
  if [ "$c" = on ] && [[ "$on" == "<"* ]]; then err "zero-launcher: --opt $id needs its value, e.g. --opt $id=12 ($on)"; return 2; fi
  OPTC[$id]=$c
}
opt_effect() {  # the choices as OPT_ENV (NAME=value), OPT_FLAGS (serverz arguments) and OPT_WORDS (for the log line)
  OPT_ENV=(); OPT_FLAGS=(); OPT_WORDS=()
  local id c kind name on off title v
  for id in $(printf '%s\n' "${!OPTC[@]}" | sort); do
    opt_known "$id" || continue  # (a choice for an option no longer in the catalog: left out)
    # (a catalog model's start: an option that does not apply to it is left out, as the menu greys it; the
    #  Researcher's review of df7379c)
    if [ "${SW_SKIP:-0}" = 1 ] && [ -n "${CT_title[$id]+x}" ] && [ -n "$(sw_reason "$id")" ]; then continue; fi
    c=${OPTC[$id]}; kind=$(opt_col "$id" kind); name=$(opt_col "$id" name)
    on=$(opt_col "$id" on); off=$(opt_col "$id" off); title=$(opt_col "$id" title)
    if [ "$c" = off ]; then
      if [ "$kind" = env ] && [ -n "$off" ]; then OPT_ENV+=("$name=$off"); OPT_WORDS+=("$title: off"); fi
      continue
    fi
    v=$c; [ "$c" = on ] && v=$on
    if [ "$kind" = env ]; then OPT_ENV+=("$name=$v")
    elif [ -n "$on" ]; then OPT_FLAGS+=("$name" "$v")
    else OPT_FLAGS+=("$name"); fi
    if [ "$c" = on ]; then OPT_WORDS+=("$title"); else OPT_WORDS+=("$title ($c)"); fi
  done
}
opt_now() {  # ID: what is chosen now, in words
  if [ -n "${OPTC[$1]+x}" ]; then printf '%s' "${OPTC[$1]}"; else printf 'default'; fi
}
opt_list() {  # every option: the choice, the default, what it does and the PCs it is for
  local id n=0
  [ -f "$OPT_TSV" ] || { say "no option catalog ($OPT_TSV)"; return 1; }
  for id in $(opt_ids); do
    n=$((n + 1))
    printf ' %2d %-12s %s  [now: %s]\n' "$n" "$id" "$(opt_col "$id" title)" "$(opt_now "$id")"
    printf '      %s\n' "$(opt_col "$id" what)"
    printf '      default: %s | for: %s | cost: %s\n' "$(opt_col "$id" default)" "$(opt_col "$id" best_for | tr ';' ' ')" "$(opt_col "$id" cost)"
  done
  say "choices: $STATE_DIR/options.json"
}
opt_menu() {  # the menu's options screen: a number to change, Enter to go back
  local s ids id on c
  while :; do
    say ""; say "speed and accuracy options (the same choices as zero-server on Windows):"; say ""
    opt_list || return 1
    read -r -p $'\noption number to change (Enter: back): ' s || return 0
    [ -z "$s" ] && return 0
    mapfile -t ids < <(opt_ids)
    if ! [[ "$s" =~ ^[0-9]+$ ]] || [ "$s" -lt 1 ] || [ "$s" -gt ${#ids[@]} ]; then say "no option \"$s\""; continue; fi
    id=${ids[$((s - 1))]}; on=$(opt_col "$id" on)
    if [[ "$on" == "<"* ]]; then
      read -r -p "$id: a value ($on), off, or Enter for the default: " c || return 0
    else
      read -r -p "$id: 1 on, 2 off, Enter the default: " c || return 0
      case "$c" in 1) c=on;; 2) c=off;; '') ;; *) say "1, 2 or Enter"; continue;; esac
    fi
    opt_set "$id" "$c" && opt_save && say "$id: $(opt_now "$id")"
  done
}


# ---- the switches (QUEUE row 15 step 3, docs/launcher-design.md): for a model and this PC, which options of the
# catalog apply, the one-line reason when one does not, and each one's tip. The same rules as the Windows launcher's
# scripts/launcher-switches.ps1; tests/test_zero_launcher.sh checks both against tests/launcher_switch_cases.tsv.
# The state: SW_WORDS (the model's words: dense moe moe_std qsa flashnext mtp h8), SW_PC (this PC's: b-series alchemist
# igpu-140t igpu-xe2 igpu-xe-lpg multi-gpu; empty: not known, nothing greyed for the PC), SW_GPUS, SW_DRAFTING (1 when
# Drafting is on) and the choices in OPTC
declare -A CT_title=() CT_applies=() CT_relates=() CT_default=() CT_on=() CT_off=() CT_variant=() CT_tip=() CT_output=() CT_cost=()
CT_IDS=()
SW_WORDS=""; SW_PC=""; SW_GPUS=1; SW_DRAFTING=0; SW_KIND=""
cat_load() {  # the catalog into CT_* (one awk for the whole file, not one per field)
  CT_IDS=()
  [ -f "$OPT_TSV" ] || return 1
  local id title applies relates def on off variant tip output cost
  while IFS=$'\x1f' read -r id title applies relates def on off variant tip output cost; do
    CT_IDS+=("$id"); CT_title[$id]=$title; CT_applies[$id]=$applies; CT_relates[$id]=$relates; CT_default[$id]=$def
    CT_on[$id]=$on; CT_off[$id]=$off; CT_variant[$id]=$variant; CT_tip[$id]=$tip; CT_output[$id]=$output; CT_cost[$id]=$cost
  done < <(tr -d '\r' < "$OPT_TSV" | awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    $1 != "" { printf "%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\n", $c["id"], $c["title"], $c["applies"],
      $c["relates"], $c["default"], $c["on"], $c["off"], $c["variant"], $c["tip"], $c["output"], $c["cost"] }')
}
sw_list() {  # LIST: its ;-separated items, trimmed, one a line (no other process: the menu calls it per option per key, #81)
  local -a p; local s
  IFS=';' read -r -a p <<< "$1"
  for s in "${p[@]}"; do
    s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}
    [ -n "$s" ] && printf '%s\n' "$s"
  done
  return 0
}
sw_list_a() {  # ARRAY LIST: sw_list's items into the array named ARRAY, without a subshell
  local -n _out=$1; local -a p; local s
  _out=()
  IFS=';' read -r -a p <<< "$2"
  for s in "${p[@]}"; do
    s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}
    [ -n "$s" ] && _out+=("$s")
  done
  return 0
}
sw_has() { case " $1 " in *" $2 "*) return 0;; esac; return 1; }  # LIST WORD
sw_or() {  # words joined "a, b or c"
  local n=$# i=0 out="" w
  for w in "$@"; do
    i=$((i + 1))
    if [ $i -eq 1 ]; then out=$w; elif [ $i -eq $n ]; then out+=" or $w"; else out+=", $w"; fi
  done
  printf '%s' "$out"
}
sw_mword() { case "$1" in dense) printf 'dense models';; moe) printf 'MoE models';; moe_std) printf 'standard MoE models';; qsa) printf 'QSA models';; flashnext) printf 'Flash Next';;
  mtp) printf 'models with MTP';; h8) printf 'dense Qwen3.5 / 3.8 models';; esac; }
sw_pword() { case "$1" in b-series) printf 'Arc B-series';; alchemist) printf 'Arc A-series';; igpu-140t) printf 'Arc 140T';;
  igpu-xe2) printf 'Lunar Lake / Panther Lake';; igpu-xe-lpg) printf 'Meteor Lake';; esac; }
sw_title() { case "$1" in mtp) printf 'Drafting';; gpus2) printf '2 or more GPUs';; *) printf '%s' "${CT_title[$1]:-$1}";; esac; }
sw_value() { [[ "${CT_on[$1]}" == "<"* ]]; }  # (a value option, such as budget: not a switch)
sw_default() {  # on / off, or auto when the engine picks it by the PC or the model
  local d=${CT_default[$1]}
  if [[ "$d" =~ ^off([^a-z]|$) ]]; then printf off
  elif [[ "$d" =~ ^on([^a-z:]|$) ]] && ! [[ "$d" =~ ^on[[:space:]]*: ]]; then printf on
  else printf auto; fi
}
sw_on_v() {  # ID: sw_on's answer in SW_ONV, without a subshell
  SW_ONV=""
  case "$1" in mtp) SW_ONV=$SW_DRAFTING; return;; gpus2) [ "$SW_GPUS" -ge 2 ] && SW_ONV=1 || SW_ONV=0; return;; esac
  if [ -n "${OPTC[$1]+x}" ]; then [ "${OPTC[$1]}" = off ] && SW_ONV=0 || SW_ONV=1; return; fi
  local d=${CT_default[$1]}
  if [[ "$d" =~ ^off([^a-z]|$) ]]; then SW_ONV=0
  elif [[ "$d" =~ ^on([^a-z:]|$) ]] && ! [[ "$d" =~ ^on[[:space:]]*: ]]; then SW_ONV=1; fi
  return 0
}
sw_on() { sw_on_v "$1"; printf '%s' "$SW_ONV"; }  # ID: 1 on, 0 off, empty when the engine picks
sw_choices() {  # ID: "on:Sparse off:Full" for a choose-one option (its two variants are its on and off values)
  local vs=() e suf val out=() w
  sw_list_a vs "${CT_variant[$1]}"
  [ ${#vs[@]} -eq 2 ] && [ -n "${CT_on[$1]}" ] && [ -n "${CT_off[$1]}" ] || return 1
  for e in "${vs[@]}"; do
    suf=${e%%:*}; val=${e#*:}; w=${suf#-}
    if [ "$val" = "${CT_on[$1]}" ]; then out+=("on:${w^}"); elif [ "$val" = "${CT_off[$1]}" ]; then out+=("off:${w^}"); else return 1; fi
  done
  printf '%s\n' "${out[*]}"
}
sw_reason() {  # ID: the reason it does not apply now, or nothing; SW_KIND says which rule (model, pc, gpus, not, needs)
  sw_reason_v "$1"; printf '%s' "$SW_WHY"
}
sw_names_v() {  # KIND WORD...: the words' names joined "a, b or c" in SW_NAMES, without a subshell (the menu, #81)
  local kind=$1 n=$(($# - 1)) i=0 w nm out=""; shift
  for w in "$@"; do
    i=$((i + 1))
    if [ "$kind" = m ]; then
      case "$w" in dense) nm='dense models';; moe) nm='MoE models';; moe_std) nm='standard MoE models';; qsa) nm='QSA models';; flashnext) nm='Flash Next';;
        mtp) nm='models with MTP';; h8) nm='dense Qwen3.5 / 3.8 models';; *) nm=$w;; esac
    else
      case "$w" in b-series) nm='Arc B-series';; alchemist) nm='Arc A-series';; igpu-140t) nm='Arc 140T';;
        igpu-xe2) nm='Lunar Lake / Panther Lake';; igpu-xe-lpg) nm='Meteor Lake';; *) nm=$w;; esac
    fi
    if [ $i -eq 1 ]; then out=$nm; elif [ $i -eq $n ]; then out+=" or $nm"; else out+=", $nm"; fi
  done
  SW_NAMES=$out
}
sw_title_v() { case "$1" in mtp) SW_T='Drafting';; gpus2) SW_T='2 or more GPUs';; *) SW_T=${CT_title[$1]:-$1};; esac; }
sw_reason_v() {  # ID: sw_reason's reason in SW_WHY (empty: it applies) and its rule in SW_KIND, in this shell
  local id=$1 ap=() mw=() pw=() rel=() w e t multi=0 any
  SW_KIND=""; SW_WHY=""
  sw_list_a ap "${CT_applies[$id]}"
  for w in "${ap[@]}"; do
    case "$w" in dense|moe|moe_std|qsa|flashnext|mtp|h8) mw+=("$w");; b-series|alchemist|igpu-140t|igpu-xe2|igpu-xe-lpg) pw+=("$w");; multi-gpu) multi=1;; esac
  done
  if [ ${#mw[@]} -gt 0 ]; then
    any=0; for w in "${mw[@]}"; do sw_has "$SW_WORDS" "$w" && any=1; done
    if [ $any = 0 ]; then sw_names_v m "${mw[@]}"; SW_KIND=model; SW_WHY="for $SW_NAMES only"; return 0; fi
  fi
  if [ -n "$SW_PC" ] && [ ${#pw[@]} -gt 0 ]; then
    any=0; for w in "${pw[@]}"; do sw_has "$SW_PC" "$w" && any=1; done
    if [ $any = 0 ]; then sw_names_v p "${pw[@]}"; SW_KIND=pc; SW_WHY="needs $SW_NAMES"; return 0; fi
  fi
  sw_list_a rel "${CT_relates[$id]}"
  for e in "${rel[@]}"; do [ "$e" = needs:gpus2 ] && multi=1; done
  if [ $multi = 1 ] && [ "$SW_GPUS" -lt 2 ]; then
    SW_KIND=gpus
    if sw_has "$SW_PC" multi-gpu; then SW_WHY='choose a Multi-GPU split first'; else SW_WHY='needs 2 or more GPUs'; fi
    return 0
  fi
  for e in "${rel[@]}"; do
    [[ "$e" == not:* ]] || continue
    t=${e#not:}; sw_on_v "$t"
    [ "$SW_ONV" = 1 ] && { sw_title_v "$t"; SW_KIND=not; SW_WHY="not with $SW_T"; return 0; }
  done
  for e in "${rel[@]}"; do
    [[ "$e" == needs:* ]] || continue
    t=${e#needs:}; [ "$t" = gpus2 ] && continue
    sw_on_v "$t"
    [ "$SW_ONV" = 0 ] && { sw_title_v "$t"; SW_KIND=needs; SW_WHY="turn $SW_T on first"; return 0; }
  done
  return 0
}
sw_tipline() {  # ID: what it needs, goes with or excludes, the output, the cost (up to its first bracket or ;)
  local id=$1 rel=() e n=() h=() x=() p=() c
  mapfile -t rel < <(sw_list "${CT_relates[$id]}")
  for e in "${rel[@]}"; do
    case "$e" in
      needs:mtp) n+=("Drafting on (MTP)");; needs:*) n+=("$(sw_title "${e#needs:}")");;
      has:*) h+=("$(sw_title "${e#has:}")");; not:*) x+=("$(sw_title "${e#not:}")");;
    esac
  done
  [ ${#n[@]} -gt 0 ] && p+=("Needs: $(IFS=,; printf '%s' "${n[*]}" | sed 's/,/, /g').")
  [ ${#h[@]} -gt 0 ] && p+=("Goes with: $(IFS=,; printf '%s' "${h[*]}" | sed 's/,/, /g').")
  [ ${#x[@]} -gt 0 ] && p+=("Not with: $(IFS=,; printf '%s' "${x[*]}" | sed 's/,/, /g').")
  [ ${#p[@]} -eq 0 ] && p+=("Works on its own.")
  if [ "${CT_output[$id]}" = same ]; then p+=("Same output."); else p+=("Output checked against full precision."); fi
  c=$(printf '%s' "${CT_cost[$id]}" | sed 's/[[:space:]]*[(;].*//')
  [ -n "$c" ] && [ "$c" != none ] && p+=("Costs: $c.")
  printf '%s' "${p[*]}"
}
sw_pc_kinds() {  # this PC's words, from the PCI names of its Intel GPUs (lspci); nothing when lspci is missing
  local names l k="" discrete=0
  names=$(lspci 2>/dev/null | grep -Ei 'vga|display|3d' | grep -i intel) || true
  while IFS= read -r l; do
    [ -n "$l" ] || continue
    if [[ "$l" =~ B3[0-9]0|140V|130V|Lunar|Panther ]]; then k+=" igpu-xe2"; [[ "$l" =~ B3[0-9]0|Panther ]] && k+=" igpu-ptl"
    elif [[ "$l" =~ 140T|Arrow\ Lake ]]; then k+=" igpu-140t"
    elif [[ "$l" =~ DG2|Alchemist|A[0-9]{3}[^0-9] ]]; then k+=" alchemist"; discrete=$((discrete + 1))
    elif [[ "$l" =~ Battlemage|BMG|B[0-9]{2,3}[^0-9] ]]; then k+=" b-series"; discrete=$((discrete + 1))
    elif [[ "$l" =~ Meteor|Arc ]]; then k+=" igpu-xe-lpg"; fi
  done <<< "$names"
  [ $discrete -gt 1 ] && k+=" multi-gpu"
  # (multi-same: those cards all of one PCI device ID, the only ones tensor parallel takes together)
  if [ $discrete -gt 1 ]; then
    local ids
    ids=$(lspci -nn 2>/dev/null | grep -Ei 'vga|display|3d' | grep -i intel | grep -Evi 'B3[0-9]0|140V|130V|140T|Lunar|Panther|Arrow Lake|Meteor' |
      grep -Ei 'DG2|Alchemist|A[0-9]{3}[^0-9]|Battlemage|BMG|B[0-9]{2,3}[^0-9]' | grep -o '8086:[0-9a-fA-F]\{4\}' | sort -u | wc -l)
    [ "$ids" = 1 ] && k+=" multi-same"
  fi
  printf '%s' "$(printf '%s\n' $k | sort -u | tr '\n' ' ' | sed 's/ $//')"
}
# a catalog model's words (its field 13) and whether a DFlash2 drafter for it is in the cache
m_words() { local w; w=$(m_field "$1" 13); printf '%s' "${w%% df=*}" | sed 's/ *df=[^ ]*//'; }
m_dflash() {  # KEY: a DFlash2 drafter's repository for the model in the Hugging Face cache (field 13's df= name)
  local pat d
  pat=$(m_field "$1" 13 | sed -n 's/.*df=\([^ ]*\).*/\1/p')
  [ -n "$pat" ] || return 1
  for d in "$(hub_dir)"/models--*"$pat"*[Dd][Ff]lash*; do [ -d "$d" ] && return 0; done
  return 1
}

# ---- the full-screen menu (docs/launcher-design.md, Linux): the models on the left, the selected one's switches on
# the right, the tip of the focused switch under them. Plain text, no colour: [x] [ ] [-] and > carry the state. In a
# terminal of 80 columns or more; otherwise, and when not a terminal, the numbered menu
declare -A TUI_DRAFT=() TUI_IMG=()
# (the menu's per-model facts, found once when it opens instead of at every key: a redraw ran model_path's find and
#  ~100 subshells, ~1 s a key on d-dorok's NixOS over SSH, #81. TUI_HAVE downloaded 1 / 0, TUI_NAME the short name,
#  TUI_WORDS the words, TUI_DCH the drafting choices)
declare -A TUI_HAVE=() TUI_NAME=() TUI_WORDS=() TUI_DCH=()
tui_cache() {
  local k
  for k in "${TUI_KEYS[@]}"; do
    model_path "$k" >/dev/null 2>&1 && TUI_HAVE[$k]=1 || TUI_HAVE[$k]=0
    TUI_NAME[$k]=$(m_field "$k" 1 | sed 's/ (.*//')
    TUI_WORDS[$k]=$(m_words "$k")
    TUI_DCH[$k]=$(tui_draft_choices "$k")
  done
}
tui_draft_choices() {  # KEY: off mtp both dflash, the ones this model has (cached for the menu: TUI_DCH)
  local k=$1
  [ -n "${TUI_DCH[$k]+x}" ] && { printf '%s' "${TUI_DCH[$k]}"; return; }
  sw_has "$(m_words "$k")" mtp || { printf 'off'; return; }
  if m_dflash "$k"; then printf 'off mtp both dflash'; else printf 'off mtp'; fi
}
tui_draft_default() { local c; c=$(tui_draft_choices "$1"); case "$c" in *both*) printf both;; *mtp*) printf mtp;; *) printf off;; esac; }
tui_draft_label() { case "$1" in off) printf 'Off';; mtp) printf 'MTP';; both) printf 'MTP+DF';; dflash) printf 'DFlash2';; esac; }
tui_state() {  # KEY: SW_* for that model with its Drafting
  local k=$1
  if [ -n "${TUI_WORDS[$k]+x}" ]; then SW_WORDS=${TUI_WORDS[$k]}; else SW_WORDS=$(m_words "$k"); fi; SW_GPUS=$GPUS; [ "$SW_GPUS" -ge 1 ] 2>/dev/null || SW_GPUS=1
  [ "${TUI_DRAFT[$k]:-$(tui_draft_default "$k")}" = off ] && SW_DRAFTING=0 || SW_DRAFTING=1
}
tui_cut() { local s=$1 n=$2; [ ${#s} -le "$n" ] && printf '%-*s' "$n" "$s" || printf '%s~' "${s:0:$((n - 1))}"; }
tui_wrap() { printf '%s\n' "$1" | fold -s -w "${2:-78}" | sed 's/^/ /'; }  # TEXT [WIDTH]
tui_items() {  # KEY: the right pane's items (draft, img, attn, the switches; the ones that do not apply folded)
  local k=$1 id why
  TUI_ITEMS=(draft img); TUI_NA=0
  [ -n "${CT_title[qsa]+x}" ] && TUI_ITEMS+=(attn)
  for id in "${CT_IDS[@]}"; do
    sw_value "$id" && continue
    sw_choices "$id" >/dev/null && continue
    sw_reason_v "$id"; why=$SW_WHY
    if [ -n "$why" ]; then TUI_NA=$((TUI_NA + 1)); [ "$TUI_SHOWNA" = 1 ] || continue; fi
    TUI_ITEMS+=("$id")
  done
  [ $TUI_NA -gt 0 ] && TUI_ITEMS+=(na)
}
tui_item_line() {  # KEY ITEM: its line in the right pane (44 columns)
  local k=$1 it=$2 why c d
  case "$it" in
    draft)
      c=$(tui_draft_choices "$k"); d=${TUI_DRAFT[$k]:-$(tui_draft_default "$k")}
      if [ "$c" = off ]; then printf '[-] Drafting  none for this model'
      else local w out=""; for w in $c; do if [ "$w" = "$d" ]; then out+="($(tui_draft_label "$w")) "; else out+="$(tui_draft_label "$w") "; fi; done; printf 'Drafting  %s' "$out"; fi;;
    img)
      if sw_has "$(m_words "$k")" vision; then
        [ "${TUI_IMG[$k]:-1}" = 1 ] && printf '[x] Images' || printf '[ ] Images'
      else printf '[-] Images  this model reads none'; fi;;
    attn)
      sw_reason_v qsa; why=$SW_WHY
      if [ -n "$why" ]; then printf '[-] Attention  %s' "$why"
      else case "${OPTC[qsa]:-}" in on) printf 'Attention  default (Sparse) Full';; off) printf 'Attention  default Sparse (Full)';; *) printf 'Attention  (default) Sparse Full';; esac; fi;;
    na) [ "$TUI_SHOWNA" = 1 ] && printf '    a: hide the %s that do not apply' "$TUI_NA" || printf '    a: show %s more that do not apply' "$TUI_NA";;
    *)
      sw_reason_v "$it"; why=$SW_WHY
      if [ -n "$why" ]; then printf '[-] %s  %s' "${CT_title[$it]}" "$why"
      else case "$(sw_on "$it")" in 1) printf '[x] %s' "${CT_title[$it]}";; 0) printf '[ ] %s' "${CT_title[$it]}";; *) printf '[~] %s (auto)' "${CT_title[$it]}";; esac; fi;;
  esac
}
tui_tip() {  # KEY ITEM: the focused item's tip, for the lines under the lists
  case "$2" in
    draft) printf 'Drafting: fast guesses of the next tokens, each checked by the model, so the text stays the same and comes faster. MTP: the model'"'"'s own draft block; DFlash2: a separate drafter found in the cache; MTP+DF: both.';;
    img) printf 'Images: the image encoder is loaded (it takes GPU memory from the context). Off: --no-mmproj, a longer context.';;
    attn) printf '%s %s' "${CT_tip[qsa]}" "$(sw_tipline qsa)";;
    na) printf 'Options that do not apply to this model or this PC, each with the reason.';;
    *) printf '%s: %s %s' "${CT_title[$2]}" "${CT_tip[$2]}" "$(sw_tipline "$2")";;
  esac
}
tui_changed() {  # KEY: what was changed from the defaults
  local k=$1 x=() id c
  [ "${TUI_DRAFT[$k]:-$(tui_draft_default "$k")}" != "$(tui_draft_default "$k")" ] && x+=("Drafting: $(tui_draft_label "${TUI_DRAFT[$k]}")")
  [ "${TUI_IMG[$k]:-1}" = 0 ] && sw_has "$(m_words "$k")" vision && x+=("Images: off")
  for id in "${CT_IDS[@]}"; do
    [ -n "${OPTC[$id]+x}" ] || continue
    sw_value "$id" && continue
    c=${OPTC[$id]}
    if [ "$id" = qsa ]; then [ "$c" = on ] && x+=("Attention: Sparse") || x+=("Attention: Full")
    elif [ "$c" = off ]; then x+=("${CT_title[$id]}: off"); else x+=("${CT_title[$id]}"); fi
  done
  if [ ${#x[@]} -eq 0 ]; then printf 'nothing'; else (IFS=';'; printf '%s' "${x[*]}" | sed 's/;/; /g'); fi
}
# (#81, d-dorok: "use the whole terminal", "mark what differs from the defaults", "stack the panes in a narrow
#  terminal", "set the keys apart". The width and height are the terminal's (TUI_W / TUI_H, tui_ok): 80 columns or
#  more side by side, the panes sized from the width; 40-79 the models above the options. The rows the lists leave go
#  to the tip, up to 6 lines. A * column marks an option changed from its default; the keys in brackets, bold where
#  the terminal has it)
tui_changed_item() {  # KEY ITEM: 0 when the item differs from its default
  local k=$1 it=$2
  case "$it" in
    draft) [ -n "${TUI_DRAFT[$k]+x}" ] && [ "${TUI_DRAFT[$k]}" != "$(tui_draft_default "$k")" ];;
    img) [ "${TUI_IMG[$k]:-1}" = 0 ];;
    attn) [ -n "${OPTC[qsa]+x}" ];;
    na) return 1;;
    *) [ -n "${OPTC[$it]+x}" ];;
  esac
}
tui_keys_line() {  # WIDTH: the keys, set apart: bold where the terminal has it, else in brackets alone
  local w=$1 b="${TUI_BOLD:-}" n="${TUI_NORM:-}" ud="Up/Down" a=""
  case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in *UTF-8*|*utf8*|*UTF8*|*utf-8*) ud=$'↑/↓';; esac
  # (the a key with the others where options are folded or shown, #81: "more" shows them, "fewer" folds them again)
  if [ "${TUI_NA:-0}" -gt 0 ]; then [ "$TUI_SHOWNA" = 1 ] && a="  $b[a]$n fewer" || a="  $b[a]$n more"; fi
  local k1="$b[$ud]$n move  $b[Tab]$n models/options  $b[Space]$n switch" k2="$b[Enter]$n start$a  $b[?]$n help  $b[q]$n quit  * changed"
  if [ "$w" -ge 107 ]; then printf ' %s  %s\n' "$k1" "$k2"
  elif [ "$w" -ge 57 ]; then printf ' %s\n %s\n' "$k1" "$k2"
  else printf ' %s\n %s\n %s\n' "$b[$ud]$n move  $b[Tab]$n panes" "$b[Space]$n switch  $b[Enter]$n start" "$b[?]$n help  $b[q]$n quit$a  * changed"; fi
}
tui_keys_n() { if [ "$1" -ge 107 ]; then echo 1; elif [ "$1" -ge 57 ]; then echo 2; else echo 3; fi; }  # WIDTH: its lines
# (after a toggle the list is filtered again: an option whose reason changed comes or goes, and the focus by index
#  landed on another option, so the next Space switched that one; #81, d-dorok. The focus follows the item's id)
tui_keep_focus() {  # ID
  local k=${TUI_KEYS[$TUI_SEL]} i
  tui_state "$k"; tui_items "$k"
  for i in "${!TUI_ITEMS[@]}"; do [ "${TUI_ITEMS[$i]}" = "$1" ] && { TUI_FOC=$i; return; }; done
  [ "$TUI_FOC" -ge ${#TUI_ITEMS[@]} ] && TUI_FOC=$((${#TUI_ITEMS[@]} - 1))
  return 0
}
tui_draw() {
  local k=${TUI_KEYS[$TUI_SEL]} i left right lines=() rows tip
  [ -n "${TUI_NAME[$k]+x}" ] || tui_cache  # (a draw before tui() opened the menu: the tests)
  tui_state "$k"; tui_items "$k"
  # (a tall terminal: the options that do not apply shown too, with their reasons, until the user chooses with a; #81)
  if [ "${TUI_NA_SET:-0}" = 0 ] && [ "$TUI_SHOWNA" = 0 ] && [ "$TUI_NA" -gt 0 ] &&
     [ $((${#TUI_ITEMS[@]} + TUI_NA)) -le $(( ${TUI_H:-24} - 14 )) ] && [ "${TUI_W:-80}" -ge 80 ]; then
    TUI_SHOWNA=1; tui_items "$k"
  fi
  [ "$TUI_FOC" -ge ${#TUI_ITEMS[@]} ] && TUI_FOC=$((${#TUI_ITEMS[@]} - 1))
  local W=${TUI_W:-80} H=${TUI_H:-24}
  [ "$W" -lt 40 ] && W=40  # (no upper cap: the whole terminal, #81)
  local stack=0; [ "$W" -lt 80 ] && stack=1
  local nk=${#TUI_KEYS[@]} ni=${#TUI_ITEMS[@]} keyl; keyl=$(tui_keys_n "$W")
  local sep; sep=" $(printf '%*s' $((W - 2)) '' | tr ' ' '-')"
  lines+=(" inferzero  $(tui_cut "${TUI_NAME[$k]}" $((W - 22)))  ~$(m_field "$k" 3) GB")
  lines+=("$sep")
  # (at most the terminal's height: the fixed lines around the lists, then as many list rows as fit; a longer options
  #  list scrolls, its window following the focus. A 24-row terminal scrolled with every option shown, and the redraw
  #  from the top then wrote over the wrong lines; the Researcher's review)
  local fixed avail mrows=$nk ms=0
  if [ $stack = 1 ]; then
    # (stacked: title, rule, two headers, two rules, the changed line, a rule and the keys, then the two lists and the
    #  tip (2 lines at least). Short of rows, the models become a window around the selection and the options get the rest)
    fixed=$((8 + keyl)); local left=$((H - fixed - 2))
    if [ $((nk + 3)) -gt "$left" ]; then mrows=$((left - 3)); [ $mrows -lt 2 ] && mrows=2; fi
    [ $mrows -gt "$nk" ] && mrows=$nk
    ms=$((TUI_SEL - mrows / 2)); [ $((ms + mrows)) -gt "$nk" ] && ms=$((nk - mrows)); [ $ms -lt 0 ] && ms=0
    avail=$((left - mrows)); [ $avail -lt 1 ] && avail=1
  else fixed=$((7 + 3 + keyl)); avail=$((H - fixed)); [ $avail -lt $nk ] && avail=$nk; fi
  local off=${TUI_OFF:-0}
  [ "$TUI_FOC" -lt "$off" ] && off=$TUI_FOC
  [ "$TUI_FOC" -ge $((off + avail)) ] && off=$((TUI_FOC - avail + 1))
  [ $((off + avail)) -gt "$ni" ] && off=$((ni - avail))
  [ "$off" -lt 0 ] && off=0
  TUI_OFF=$off
  local orows=$ni; [ "$orows" -gt "$avail" ] && orows=$avail
  local more=""; [ "$ni" -gt "$orows" ] && more="  ($((off + 1))-$((off + orows)) of $ni)"
  # (a model's row; an option's row: the focus mark, the * of a changed option, the item)
  model_row() {
    local mk=${TUI_KEYS[$1]} mark=" " cur=" "
    [ "${TUI_HAVE[$mk]:-0}" = 1 ] && mark="*"
    [ "$1" -eq "$TUI_SEL" ] && cur=">"
    printf '%s%d %s %s' "$cur" $(($1 + 1)) "$mark" "$(tui_cut "${TUI_NAME[$mk]}" $(($2 - 6)))"
  }
  opt_row() {
    local it=${TUI_ITEMS[$1]} f=" " c=" "
    [ "$TUI_PANE" = opts ] && [ "$1" -eq "$TUI_FOC" ] && f=">"
    tui_changed_item "$k" "$it" && c="*"
    printf '%s%s%s' "$f" "$c" "$(tui_cut "$(tui_item_line "$k" "$it")" $(($2 - 2)))"
  }
  if [ $stack = 1 ]; then
    local mmore=""; [ "$mrows" -lt "$nk" ] && mmore="  ($((ms + 1))-$((ms + mrows)) of $nk)"
    lines+=("   $(tui_cut "Models  (* downloaded)$mmore" $((W - 4)))")
    for ((i = ms; i < ms + mrows; i++)); do lines+=(" $(model_row $i $((W - 2)))"); done
    lines+=("$sep")
    lines+=("   $(tui_cut "Options$more" $((W - 4)))")
    for ((i = 0; i < orows; i++)); do lines+=(" $(opt_row $((i + off)) $((W - 2)))"); done
  else
    local LW=$((W / 3)); [ $LW -lt 32 ] && LW=32; [ $LW -gt 44 ] && LW=44
    local RW=$((W - LW - 3))
    rows=$nk; [ "$orows" -gt "$rows" ] && rows=$orows
    lines+=("   $(tui_cut "Models" $((LW - 2)))Options for the selected model$more")
    for ((i = 0; i < rows; i++)); do
      left=""; right=""
      [ $i -lt "$nk" ] && left=$(model_row $i "$LW")
      [ $i -lt "$orows" ] && right=$(opt_row $((i + off)) "$RW")
      lines+=(" $(tui_cut "$left" "$LW") $right")
    done
  fi
  lines+=("$sep")
  if [ "$TUI_PANE" = opts ]; then tip=$(tui_tip "$k" "${TUI_ITEMS[$TUI_FOC]}")
  else tip="$(m_field "$k" 1), ~$(m_field "$k" 3) GB, wants $(m_field "$k" 5) GB of RAM. $([ "${TUI_HAVE[$k]:-0}" = 1 ] && printf 'Downloaded.' || printf 'Enter downloads it first.')"; fi
  # (the tip: 3 lines, more where the terminal has rows to spare, up to 6)
  local tipn=$((H - ${#lines[@]} - 3 - keyl)) tipmin=3; [ $stack = 1 ] && tipmin=2
  [ $tipn -lt $tipmin ] && tipn=$tipmin; [ $tipn -gt 6 ] && tipn=6
  mapfile -t -O ${#lines[@]} lines < <(tui_wrap "$tip" $((W - 2)) | head -"$tipn")
  lines+=(" Changed from the defaults: $(tui_changed "$k" | cut -c1-$((W - 29 > 10 ? W - 29 : 10)))")
  lines+=("$sep")
  mapfile -t -O ${#lines[@]} lines < <(tui_keys_line "$W")
  printf '\033[H'
  printf '%s\033[K\n' "${lines[@]}"
  printf '\033[J'
}
tui_toggle() {  # the focused item: Space
  local k=${TUI_KEYS[$TUI_SEL]} it=${TUI_ITEMS[$TUI_FOC]} c d w nx="" first="" take=0
  case "$it" in
    draft)
      c=$(tui_draft_choices "$k"); d=${TUI_DRAFT[$k]:-$(tui_draft_default "$k")}
      for w in $c; do [ -z "$first" ] && first=$w; if [ $take = 1 ]; then nx=$w; take=2; fi; [ "$w" = "$d" ] && [ $take = 0 ] && take=1; done
      TUI_DRAFT[$k]=${nx:-$first};;
    img) sw_has "$(m_words "$k")" vision && { [ "${TUI_IMG[$k]:-1}" = 1 ] && TUI_IMG[$k]=0 || TUI_IMG[$k]=1; };;
    attn) [ -z "$(sw_reason qsa)" ] && case "${OPTC[qsa]:-}" in on) OPTC[qsa]=off;; off) unset 'OPTC[qsa]';; *) OPTC[qsa]=on;; esac;;
    na) TUI_NA_SET=1; [ "$TUI_SHOWNA" = 1 ] && TUI_SHOWNA=0 || TUI_SHOWNA=1;;
    *)
      [ -n "$(sw_reason "$it")" ] && return
      d=$(sw_default "$it")
      if [ "$d" = auto ]; then case "${OPTC[$it]:-}" in on) OPTC[$it]=off;; off) unset "OPTC[$it]";; *) OPTC[$it]=on;; esac
      else case "$(sw_on "$it")" in 1) c=off;; *) c=on;; esac; if [ "$c" = "$d" ]; then unset "OPTC[$it]"; else OPTC[$it]=$c; fi; fi;;
  esac
}
tui_help() {
  printf '\033[H\033[J'
  say " inferzero: keys"; say ""
  say "   up / down     move in the list that has the > mark"
  say "   tab           switch between the models and the selected model's options"
  say "   1-9           choose a model by its number"
  say "   space         turn the focused switch on or off, or step a choice (Drafting, Attention)"
  say "                 [x] on   [ ] off   [~] the engine picks   [-] does not apply (the reason next to it)"
  say "                 * next to an option: changed from its default; * next to a model: downloaded"
  say "   a             show or hide the options that do not apply"
  say "   enter         start the server with the selected model (downloads it first if needed)"
  say "   r             reset the switches to the defaults"
  say "   c             check the system"
  say "   q             quit"
  say ""; say " The switches are kept in $STATE_DIR/options.json, the same choices as zero-server on Windows;"
  say " each model's Drafting and Images in $STATE_DIR/launcher-models."
  say " Press a key to go back."
  IFS= read -rsn1 _ || true
}
tui() {  # returns 0 after the server ran or on q; the key to start is in TUI_START
  local key rest saved
  TUI_KEYS=(); local e; for e in "${CATALOG[@]}"; do TUI_KEYS+=("${e%%|*}"); done
  TUI_SEL=0; TUI_FOC=0; TUI_PANE=models; TUI_SHOWNA=0; TUI_START=""
  local lp; lp=$(cat "$STATE_DIR/launcher-last" 2>/dev/null)
  for ((e = 0; e < ${#TUI_KEYS[@]}; e++)); do [ "${TUI_KEYS[$e]}" = "$lp" ] && TUI_SEL=$e; done
  saved=$(stty -g 2>/dev/null)
  trap 'stty "$saved" 2>/dev/null; printf "\033[?25h"' EXIT
  # (a resized window: the trap takes its size and asks for a full redraw. Bash 5.2 holds a WINCH that arrives during
  #  read until the read returns, so the key read waits a second at a time (read -t 1), and at each timeout a resize
  #  that came is drawn; a shell whose read a signal interrupts returns above 128 the same way. Neither leaves the menu
  #  (the Researcher's review of e4b395e))
  TUI_RESIZED=0
  trap 'TUI_W=$(tput cols 2>/dev/null || echo 80); TUI_H=$(tput lines 2>/dev/null || echo 24); TUI_RESIZED=1' WINCH
  TUI_BOLD=$(tput bold 2>/dev/null || true); TUI_NORM=$(tput sgr0 2>/dev/null || true)
  stty -echo -icanon 2>/dev/null
  printf '\033[?25l\033[H\033[J'
  tui_cache
  tui_mem_load
  local rc need=1
  while :; do
    # (keys already waiting, a held arrow key or a paste: handled first, one redraw after them; #81)
    if [ $need = 1 ]; then
      if [ "$TUI_RESIZED" = 1 ]; then TUI_RESIZED=0; printf '\033[H\033[J'; fi
      read -t 0 2>/dev/null || { tui_draw; need=0; }
    fi
    IFS= read -rsn1 -t 1 key; rc=$?
    if [ $rc -gt 128 ]; then [ "$TUI_RESIZED" = 1 ] && need=1; continue; fi  # (a timeout or a signal: no key)
    [ $rc -ne 0 ] && break  # (the end of the input)
    need=1
    if [ "$key" = $'\e' ]; then
      IFS= read -rsn2 -t 0.05 rest; rc=$?
      [ $rc -gt 128 ] && { IFS= read -rsn2 -t 0.05 rest; rc=$?; }  # (a resize inside the escape sequence: once more)
      [ $rc -eq 0 ] || rest=""
      key="ESC$rest"
    fi
    case "$key" in
      'ESC[A') if [ "$TUI_PANE" = models ]; then [ $TUI_SEL -gt 0 ] && TUI_SEL=$((TUI_SEL - 1)); TUI_FOC=0; else [ $TUI_FOC -gt 0 ] && TUI_FOC=$((TUI_FOC - 1)); fi;;
      'ESC[B') if [ "$TUI_PANE" = models ]; then [ $TUI_SEL -lt $((${#TUI_KEYS[@]} - 1)) ] && TUI_SEL=$((TUI_SEL + 1)); TUI_FOC=0; else [ $TUI_FOC -lt $((${#TUI_ITEMS[@]} - 1)) ] && TUI_FOC=$((TUI_FOC + 1)); fi;;
      $'\t'|'ESC[C'|'ESC[D') [ "$TUI_PANE" = models ] && TUI_PANE=opts || TUI_PANE=models;;
      [1-9]) [ "$key" -le ${#TUI_KEYS[@]} ] && { TUI_SEL=$((key - 1)); TUI_PANE=models; TUI_FOC=0; };;
      ' ') if [ "$TUI_PANE" = opts ]; then
             local fid=${TUI_ITEMS[$TUI_FOC]:-}; tui_toggle; opt_save; tui_mem_save; tui_keep_focus "$fid"
           else TUI_PANE=opts; fi;;
      a|A) local fid=${TUI_ITEMS[$TUI_FOC]:-}; TUI_NA_SET=1; [ "$TUI_SHOWNA" = 1 ] && TUI_SHOWNA=0 || TUI_SHOWNA=1
           [ "$TUI_PANE" = opts ] && tui_keep_focus "$fid";;
      r|R) local k=${TUI_KEYS[$TUI_SEL]} id; unset "TUI_DRAFT[$k]" "TUI_IMG[$k]"
           for id in "${CT_IDS[@]}"; do sw_value "$id" || unset "OPTC[$id]"; done; opt_save; tui_mem_save;;
      '?') tui_help; printf '\033[H\033[J';;
      c|C) printf '\033[H\033[J'; print_check; say ""; say " Press a key to go back."; IFS= read -rsn1 _ || true; printf '\033[H\033[J';;
      q|Q|'ESC') break;;
      '') TUI_START=${TUI_KEYS[$TUI_SEL]}; break;;
    esac
  done
  stty "$saved" 2>/dev/null; printf '\033[?25h\033[H\033[J'; trap - EXIT WINCH
  [ -n "$TUI_START" ] || return 0
  mkdir -p "$STATE_DIR" 2>/dev/null && printf '%s' "$TUI_START" > "$STATE_DIR/launcher-last"
  SEL_DRAFT=${TUI_DRAFT[$TUI_START]:-$(tui_draft_default "$TUI_START")}; SEL_IMG=${TUI_IMG[$TUI_START]:-1}
  # (MTP where the cache showed no DFlash2 drafter: plain --mtp, not --draft mtp, so a drafter the server finds itself
  #  is still used; the Researcher's review)
  case " $(tui_draft_choices "$TUI_START") " in *" both "*) ;; *) [ "$SEL_DRAFT" = mtp ] && SEL_DRAFT=both;; esac
  start_key "$TUI_START"
}
tui_ok() {  # a terminal the full-screen menu can draw in: input and output a terminal, not dumb, 40 x 20 or more
  [ -t 0 ] && [ -t 1 ] || return 1
  case "${TERM:-dumb}" in dumb|unknown|'') return 1;; esac
  local c r; c=$(tput cols 2>/dev/null || echo 80); r=$(tput lines 2>/dev/null || echo 24)
  TUI_H=${r:-24}; TUI_W=${c:-80}  # (tui_draw keeps to them: side by side from 80 columns, stacked below; #81)
  [ "${c:-0}" -ge 40 ] && [ "${r:-0}" -ge 20 ]
}
# (#81: the Drafting and Images choices per model, kept between runs as the switches are in options.json:
#  launcher-models, a line per model "KEY DRAFT IMG", only the ones changed from their defaults)
tui_mem_load() {
  local f="$STATE_DIR/launcher-models" key d im
  [ -f "$f" ] || return 0
  while read -r key d im; do
    [ -n "$key" ] || continue
    case "$d" in off|mtp|both|dflash) TUI_DRAFT[$key]=$d;; esac
    case "$im" in 0|1) TUI_IMG[$key]=$im;; esac
  done < "$f"
}
tui_mem_save() {
  local f="$STATE_DIR/launcher-models" tmp key
  mkdir -p "$STATE_DIR" 2>/dev/null || return 1
  tmp=$(mktemp "$STATE_DIR/.launcher-models-XXXXXX") || return 1
  for key in $(printf '%s\n' "${!TUI_DRAFT[@]}" "${!TUI_IMG[@]}" | sort -u); do
    printf '%s %s %s\n' "$key" "${TUI_DRAFT[$key]:--}" "${TUI_IMG[$key]:--}"
  done > "$tmp" && mv -f "$tmp" "$f" || { rm -f "$tmp"; return 1; }
}
# --opt ID with --model KEY: an option that cannot apply to that model is refused with the menu's reason line (the
# PC's reasons only warn: lspci's names are a guess, and a wrong guess must not stop a start)
CLI_ON=()
cli_check() {  # KEY
  local k=$1 id why bad=0
  [ ${#CLI_ON[@]} -gt 0 ] && [ ${#CT_IDS[@]} -gt 0 ] || return 0
  SW_WORDS=$(m_words "$k"); SW_GPUS=${GPUS:-1}; [ "$SW_GPUS" -ge 1 ] 2>/dev/null || SW_GPUS=1
  SW_DRAFTING=$(m_field "$k" 6)
  for id in "${CLI_ON[@]}"; do
    [ -n "${CT_title[$id]+x}" ] || continue
    why=$(sw_reason "$id")
    [ -n "$why" ] || continue
    sw_reason "$id" >/dev/null  # (again outside the subshell: for SW_KIND)
    why=${why//turn Drafting on first/turn --mtp on first}
    if [ "$SW_KIND" = pc ]; then err "zero-launcher: note: --opt $id: $why (as far as lspci tells)"; else err "zero-launcher: --opt $id: $why"; bad=1; fi
  done
  return $bad
}

extra_has() {  # NAME...: one of these options is among the arguments after -- (EXTRA)
  local a n
  for a in "${EXTRA[@]}"; do
    for n in "$@"; do [ "$a" = "$n" ] || [ "${a%%=*}" = "$n" ] && return 0; done
  done
  return 1
}

server_cmd() {  # PATH MTP: fills CMD (the menu's Drafting and Images: SEL_DRAFT off / mtp / both / dflash, SEL_IMG 0 / 1)
  # (the launcher's own --ctx auto / --think-budget / --session-cache-gb only where the arguments after -- give none:
  #  the user's then stand alone in the command, #39; serverz takes the last --ctx in any case)
  CMD=("$SERVERZ" -m "$1" --port "$PORT")
  extra_has --ctx -c || CMD+=(--ctx "${CTX:-auto}")  # (--ctx N of the launcher, else auto)
  extra_has --think-budget || CMD+=(--think-budget "${THINK:-8192}")
  extra_has --session-cache-gb || CMD+=(--session-cache-gb "${SESS_GB:-8}")
  [ -n "${CHUNK:-}" ] && CMD+=(--chunk "$CHUNK")
  [ -n "${LOG_DIR:-}" ] && CMD+=(--log-dir "$LOG_DIR")  # (the logs folder: also in a unit made by --install-service)
  case "${SEL_DRAFT:-}" in
    off) ;;
    mtp) CMD+=(--mtp --draft mtp);;
    dflash) CMD+=(--mtp --draft dflash);;
    both) CMD+=(--mtp);;
    *) [ "$2" = 1 ] && CMD+=(--mtp);;
  esac
  [ "${SEL_IMG:-1}" = 0 ] && CMD+=(--no-mmproj)
  [ -n "$HOST" ] && CMD+=(--host "$HOST")
  [ "${GPUS:-0}" -gt 1 ] && CMD+=(--gpus "$GPUS")
  opt_effect
  CMD+=("${OPT_FLAGS[@]}")
  CMD+=("${EXTRA[@]}")
}

run_server() {
  local host=${HOST:-127.0.0.1} shown url pid opened=0 rc
  case "$host" in 0.0.0.0) shown=127.0.0.1;; ::) shown=::1;; *) shown=$host;; esac
  case "$shown" in *:*) url="http://[$shown]:$PORT/";; *) url="http://$shown:$PORT/";; esac
  if port_busy "$host" "$PORT"; then say "port $PORT is in use (another server?): stop it, or choose another with --port"; return 4; fi  # (4, as zero-server: not 1, which Docker restarts in a loop)
  exposed "$HOST" && say "note: --host $HOST makes the server reachable from your network; it has no password of its own"
  [ ${#OPT_WORDS[@]} -gt 0 ] && say "options: $(IFS=';'; printf '%s' "${OPT_WORDS[*]}" | sed 's/;/; /g')"
  say "starting: ${OPT_ENV[*]}${OPT_ENV[*]:+ }${CMD[*]}"
  (cd "$ROOT" && exec env "${OPT_ENV[@]}" "${CMD[@]}") &
  pid=$!
  # (passed on: Ctrl+C reaches the server from the terminal too, which is harmless; a SIGINT or SIGTERM sent to the
  #  launcher alone (kill, an IDE's stop, systemd) must reach it as well. As TERM: a background command of a
  #  non-interactive shell starts with SIGINT and SIGQUIT ignored, so an INT never stopped the server (d-dorok,
  #  issue #37: SigIgn 0x1006))
  trap 'kill -TERM $pid 2>/dev/null' INT
  trap 'kill -TERM $pid 2>/dev/null' TERM
  while kill -0 "$pid" 2>/dev/null; do
    if [ $opened -eq 0 ] && curl -s --noproxy '*' --max-time 2 -o /dev/null -w '%{http_code}' "${url}health" 2>/dev/null | grep -q '^200$'; then
      opened=1
      say ""; say "ready: $url  (Ctrl+C stops the server)"
      if [ "$NO_BROWSER" != 1 ] && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; } && command -v xdg-open >/dev/null; then
        # (a session of its own: Ctrl+C or a closed terminal must not take the browser with it)
        setsid xdg-open "$url" >/dev/null 2>&1 < /dev/null &
      fi
    fi
    sleep 1
  done
  wait "$pid"; rc=$?
  trap - INT TERM
  # (the GPU failsafe, docs/gpu-failsafe.md: 3 a GPU hang, 4 a lost GPU; the marker refuses GPU work until a restart)
  if [ "$rc" = 3 ] || [ "$rc" = 4 ]; then
    if [ -f "$STATE_DIR/gpu_hang.txt" ]; then
      say "the GPU hung or was lost (serverz exit $rc): restart the computer before using the GPU again. The marker ($STATE_DIR/gpu_hang.txt):"
      sed 's/^/  /' "$STATE_DIR/gpu_hang.txt" >&2
    fi
  fi
  return $rc
}

# ---- the systemd user service
service_unit() {  # prints the unit's path, or fails with a message when the name is not a plain unit name
  local name=${SERVICE_NAME%.service} base
  if ! [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$ ]]; then
    err "--service-name: use a simple unit name, without a path or whitespace"; return 1
  fi
  base=${XDG_CONFIG_HOME:-}
  [ "${base:0:1}" = "/" ] || base="$HOME/.config"
  printf '%s/systemd/user/%s.service' "$base" "$name"
}
service_refused() {  # a reason not to write the unit (checked before a download), or nothing
  local unit
  unit=$(service_unit) || return 0
  if { [ -e "$unit" ] || [ -L "$unit" ]; } && [ "$REPLACE_SERVICE" != 1 ]; then
    printf '%s already exists; left unchanged. Use --service-name NAME for a separate service, or --replace-service to replace this unit explicitly.' "$unit"
  fi
  return 0
}
# systemd's own quoting, not the shell's: each word in double quotes with \ and " escaped and its % specifiers doubled;
# in ExecStart= its $ variables too, which systemd expands there and not in Environment=
sq() { local c=$1; c=${c//\\/\\\\}; c=${c//\"/\\\"}; c=${c//%/%%}; [ "${2:-1}" = 1 ] && c=${c//\$/\$\$}; printf '"%s"' "$c"; }

install_service() {
  local unit why name d tmp existed=0 c k v exec_line="" env_lines="" nl=$'\n' cr=$'\r'
  why=$(service_refused)
  if [ -n "$why" ]; then say "$why"; return 1; fi
  unit=$(service_unit) || return 1
  name=$(basename "$unit" .service)
  for c in "${CMD[@]}"; do
    case "$c" in *"$nl"*|*"$cr"*) say "--install-service: a path, option or cache environment value holds a line break"; return 1;; esac
    exec_line+="$(sq "$c") "
  done
  for k in "${CACHE_ENV[@]}"; do
    v=${!k:-}
    [ -n "$v" ] || continue
    case "$v" in *"$nl"*|*"$cr"*) say "--install-service: a path, option or cache environment value holds a line break"; return 1;; esac
    env_lines+="Environment=$(sq "$k=$v" 0)$nl"
  done
  for v in "${OPT_ENV[@]}"; do  # (the speed and accuracy options' switches: catalog names and checked values)
    env_lines+="Environment=$(sq "$v" 0)$nl"
  done
  d=$(dirname "$unit")
  mkdir -p "$d" || return 1
  { [ -e "$unit" ] || [ -L "$unit" ]; } && existed=1
  tmp=$(mktemp "$d/.inferzero-XXXXXX") || return 1
  # (KillMode=control-group, the default: the server's children (terminal tool, MCP servers) end with it;
  #  WorkingDirectory: the package, where the server finds its kernels and web UI)
  printf '[Unit]\nDescription=inferzero server\nAfter=network.target\n\n[Service]\nWorkingDirectory=%s\n%sExecStart=%s\nRestart=on-failure\nRestartSec=5\nRestartPreventExitStatus=3 4\n\n[Install]\nWantedBy=default.target\n' \
    "$(sq "$ROOT")" "$env_lines" "${exec_line% }" > "$tmp"
  if [ "$REPLACE_SERVICE" = 1 ]; then
    mv -fT "$tmp" "$unit" || { rm -f "$tmp"; return 1; }  # (replaces a symlink itself, never writes through it)
  elif ! ln "$tmp" "$unit" 2>/dev/null; then  # (an atomic create that never replaces a file)
    if [ -e "$unit" ] || [ -L "$unit" ]; then rm -f "$tmp"; say "$unit already exists; left unchanged."; return 1; fi
    # (a file system without hard links: an exclusive create instead, still never replacing a file)
    if ! (set -C; cat "$tmp" > "$unit") 2>/dev/null; then rm -f "$tmp"; say "$unit already exists; left unchanged."; return 1; fi
    rm -f "$tmp"
  else
    rm -f "$tmp"
  fi
  if [ $existed -eq 1 ]; then say "replaced $unit"; else say "wrote $unit"; fi
  exposed "$HOST" && say "note: --host $HOST makes the server reachable from your network; it has no password of its own"
  say "start it now and at every login:"
  say "  systemctl --user daemon-reload"
  say "  systemctl --user enable $name"
  say "  systemctl --user restart $name"
  say "its log: journalctl --user -u $name -f"
  say "(to keep it running while logged out: sudo loginctl enable-linger \$USER)"
}

# the discrete Arc cards (lspci), for best-setups.tsv's Gpus=all and gpus2same
sw_discrete() {
  lspci 2>/dev/null | grep -Ei 'vga|display|3d' | grep -i intel | grep -Evi 'B3[0-9]0|140V|130V|140T|Lunar|Panther|Arrow Lake|Meteor' |
    grep -Eci 'DG2|Alchemist|A[0-9]{3}[^0-9]|Battlemage|BMG|B[0-9]{2,3}[^0-9]' || true
}
# "Automatic (best measured)" (the launcher spec B, 2026-11-22): scripts/best-setups.tsv's rows for this PC and the
# model KEY, each on a setting not chosen already (--opt / --gpus / the menu's Drafting); --no-auto: none. A row's
# needs this launcher cannot tell (notfit: the server's memory plan) leave the row out, as the engine's default
best_auto() {
  [ "${NO_AUTO:-0}" = 1 ] && return 0
  local key=$1 f="$ROOT/scripts/best-setups.tsv" line id pc models os needs settings why ok w s k v n
  [ -f "$f" ] || return 0
  n=$(sw_discrete)
  while IFS=$'\t' read -r id pc models os needs settings why _; do
    [ "$id" = id ] && continue
    [ -n "$id" ] && [ "${id:0:1}" != "#" ] || continue
    ok=1
    for w in ${pc//;/ }; do [ "$w" = "-" ] && continue; [[ " $SW_PC " == *" $w "* ]] || ok=0; done
    [[ ";$models;" == *";*;"* || ";$models;" == *";$key;"* ]] || ok=0
    [ "$os" = any ] || [ "$os" = linux ] || ok=0
    for w in ${needs//;/ }; do
      case "$w" in
        -) ;;
        gpus2same) { [ "${n:-0}" -ge 2 ] && [[ " $SW_PC " == *" multi-same "* ]]; } || ok=0;;  # (every card one device ID, sw_pc_kinds)
        *) ok=0;;
      esac
    done
    [ $ok = 1 ] || continue
    for s in ${settings//;/ }; do
      k=${s%%=*}; v=${s#*=}
      case "$k" in
        Opt) if opt_known "$v" && [ -z "${OPTC[$v]+x}" ]; then OPTC[$v]=on; say "best measured: $why ($v)"; fi;;
        Gpus) if [ "${GPUS:-0}" = 0 ] && [ "${n:-0}" -ge 2 ]; then GPUS=$([ "$v" = all ] && printf '%s' "$n" || printf '%s' "$v"); say "best measured: $why (--gpus $GPUS)"; fi;;
        NoMtp) if [ -z "${SEL_DRAFT:-}" ]; then SEL_DRAFT=off; say "best measured: $why (no drafting)"; fi;;
      esac
    done
  done < "$f"
}

start_key() {  # KEY: download if needed, then the server or the unit
  local key=$1 path why
  if [ "$INSTALL_SERVICE" = 1 ]; then  # (refused before any download, not after several GB of it)
    why=$(service_refused); [ -z "$why" ] || { say "$why"; return 1; }
    service_unit >/dev/null || return 1
  fi
  path=$(model_path "$key") || { download "$key" "$YES" || return 1; path=$(model_path "$key") || return 1; }
  best_auto "$key"  # (scripts/best-setups.tsv; --no-auto: none)
  # (the switches' state for this model: options that do not apply to it are not passed, opt_effect)
  SW_WORDS=$(m_words "$key"); SW_GPUS=${GPUS:-1}; [ "$SW_GPUS" -ge 1 ] 2>/dev/null || SW_GPUS=1
  case "${SEL_DRAFT:-}" in off) SW_DRAFTING=0;; '') SW_DRAFTING=$(m_field "$key" 6);; *) SW_DRAFTING=1;; esac
  SW_SKIP=1
  server_cmd "$path" "$(m_field "$key" 6)"
  SW_SKIP=0
  if [ "$INSTALL_SERVICE" = 1 ]; then install_service; else run_server; fi
}

list_models() {
  local e key p
  for e in "${CATALOG[@]}"; do
    IFS='|' read -r -a _f <<< "$e"; key=${_f[0]}
    if p=$(model_path "$key"); then
      printf '%-6s * %s, ~%s GB, %s GB RAM\n         %s\n' "$key" "${_f[1]}" "${_f[3]}" "${_f[5]}" "$p"
    else
      printf '%-6s   %s, ~%s GB, %s GB RAM\n' "$key" "${_f[1]}" "${_f[3]}" "${_f[5]}"
    fi
  done
}

menu() {
  local ram i=0 e mark fit s keys=()
  ram=$(ram_gb)
  say "inferzero: models (* downloaded)"; say ""
  for e in "${CATALOG[@]}"; do
    IFS='|' read -r -a _f <<< "$e"; i=$((i + 1)); keys+=("${_f[0]}")
    mark=" "; model_path "${_f[0]}" >/dev/null && mark="*"
    fit=""; [ "${ram:-0}" -gt 0 ] && [ $((ram + 1)) -lt "${_f[5]}" ] && fit="  (wants ${_f[5]} GB of RAM)"
    printf ' %2d %s %s, ~%s GB%s\n' "$i" "$mark" "${_f[1]}" "${_f[3]}" "$fit"
  done
  say ""; say "  c  check the system    o  speed and accuracy options    q  quit"
  read -r -p $'\nmodel number: ' s || return 130
  s=${s,,}
  case "$s" in
    c) print_check; return $?;;
    o) opt_menu; menu; return $?;;
    q|'') return 0;;
  esac
  if ! [[ "$s" =~ ^[0-9]+$ ]] || [ "$s" -lt 1 ] || [ "$s" -gt ${#keys[@]} ]; then say "no model \"$s\""; return 2; fi
  start_key "${keys[$((s - 1))]}"
}

# the folders searched for models besides the Hugging Face cache: ~/.inferzero/model-dirs.txt, one a line, '#'
# comments (serverz reads it, tools/server_catalog.h); --add-model-dir D / --remove-model-dir D change it (#81)
model_dirs() {  # ADD REMOVE
  local f="$STATE_DIR/model-dirs.txt" add=$1 rm=$2 d tmp
  mkdir -p "$STATE_DIR" || return 1
  [ -f "$f" ] || : > "$f"
  if [ -n "$add" ]; then
    [ -d "$add" ] || { err "zero-launcher: not a folder: $add"; return 2; }
    add=$(cd "$add" && pwd -P)
    if sed 's/^\xEF\xBB\xBF//; s/[[:space:]]*$//' "$f" | grep -Fxq -- "$add"; then say "already listed: $add"
    else printf '%s\n' "$add" >> "$f"; say "added: $add"; fi
  fi
  if [ -n "$rm" ]; then
    rm=${rm%/}
    tmp=$(mktemp "$STATE_DIR/.model-dirs-XXXXXX") || return 1
    sed 's/^\xEF\xBB\xBF//' "$f" | awk -v r="$rm" '{ t = $0; sub(/[[:space:]]+$/, "", t); sub(/\/$/, "", t); if (t != r) print }' > "$tmp"
    if cmp -s "$tmp" "$f"; then rm -f "$tmp"; say "not listed: $rm"; else mv -f "$tmp" "$f"; say "removed: $rm"; fi
  fi
  say "model folders ($f), searched with the Hugging Face cache and $STATE_DIR/models:"
  local n=0
  while IFS= read -r d || [ -n "$d" ]; do
    d=${d#$'\xEF\xBB\xBF'}; d=${d%$'\r'}
    case "$d" in ''|'#'*) continue;; esac
    n=$((n + 1))
    if [ -d "$d" ]; then say "  $d"; else say "  $d   (not found)"; fi
  done < "$f"
  [ $n -gt 0 ] || say "  none (add one: ./zero-launcher.sh --add-model-dir /path/to/models)"
  [ -n "${ARC_MODEL_DIRS:-}" ] && say "  and ARC_MODEL_DIRS: $ARC_MODEL_DIRS"
  return 0
}
usage() { sed -n '2,22p' "$SELF" | sed 's/^# \{0,1\}//'; }

main() {
  local mode="" arg_model="" arg_dl="" md_add="" md_rm=""
  PORT=8090; HOST=""; GPUS=0; NO_BROWSER=0; YES=0; MTP=0; INSTALL_SERVICE=0; SERVICE_NAME=inferzero
  THINK=8192; SESS_GB=8; CHUNK=""; CTX=""; NO_AUTO=0  # (as zero-server's defaults on Windows: the launcher spec A2 / A3, 2026-11-22)
  REPLACE_SERVICE=0; EXTRA=()
  local opt_args=() a no_saved=0
  opt_load
  cat_load; SW_PC=$(sw_pc_kinds)
  while [ $# -gt 0 ]; do
    case "$1" in
      --options) mode=options;;
      --opt|--no-opt|--default-opt) [ $# -ge 2 ] || { err "zero-launcher: $1 needs an option id (--options lists them)"; return 2; }
        opt_args+=("$1" "$2"); shift;;
      --no-saved-options) no_saved=1;;
      --check) mode=check;;
      --list) mode=list;;
      --download) arg_dl=${2:-}; shift; mode=download;;
      --model) arg_model=${2:-}; shift;;
      --mtp) MTP=1;;
      --port) PORT=${2:-}; shift;;
      --host) HOST=${2:-}; shift;;
      --gpus) GPUS=${2:-0}; shift;;
      --think-budget) THINK=${2:-}; shift;;
      --session-cache-gb) SESS_GB=${2:-}; shift;;
      --chunk) CHUNK=${2:-}; shift;;
      --ctx|-c) CTX=${2:-}; shift;;
      --log-dir) LOG_DIR=${2:-}; shift; case "$LOG_DIR" in ''|/*) ;; *) LOG_DIR="$PWD/$LOG_DIR";; esac;;  # (absolute: serverz runs in the package folder)
      --model-dirs) mode=model_dirs;;
      --add-model-dir) md_add=${2:-}; shift; mode=model_dirs;;
      --remove-model-dir) md_rm=${2:-}; shift; mode=model_dirs;;
      --logs) mode=logs;;
      --no-auto) NO_AUTO=1;;
      --no-browser) NO_BROWSER=1;;
      --yes) YES=1;;
      --install-service) INSTALL_SERVICE=1;;
      --service-name) SERVICE_NAME=${2:-}; shift;;
      --replace-service) REPLACE_SERVICE=1;;
      -h|--help) usage; return 0;;
      --) shift; EXTRA=("$@"); break;;
      *) err "zero-launcher: unknown option $1 (--help lists them)"; return 2;;
    esac
    shift
  done
  local x
  for x in "${EXTRA[@]}"; do  # (the launcher polls the port it was given: those two go through its own options)
    case "${x%%=*}" in --port|--host) err "zero-launcher: ${x%%=*} after --: give it to the launcher itself (before --)"; return 2;; esac
  done
  [[ "$PORT" =~ ^[0-9]+$ ]] && [ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || { err "zero-launcher: --port: 1..65535"; return 2; }
  [[ "$GPUS" =~ ^[0-9]+$ ]] || { err "zero-launcher: --gpus: a number"; return 2; }
  [[ "$THINK" =~ ^-?[0-9]+$ ]] || { err "zero-launcher: --think-budget: a number of tokens (-1: no limit)"; return 2; }
  [[ "$SESS_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || { err "zero-launcher: --session-cache-gb: GB (0: off)"; return 2; }
  [ -z "$CTX" ] || [ "$CTX" = auto ] || [[ "$CTX" =~ ^[0-9]+$ ]] || { err "zero-launcher: --ctx: a number of tokens, or auto"; return 2; }
  [ -z "$CHUNK" ] || [[ "$CHUNK" =~ ^[0-9]+$ ]] || { err "zero-launcher: --chunk: a number of tokens"; return 2; }
  [ "$REPLACE_SERVICE" = 1 ] && [ "$INSTALL_SERVICE" != 1 ] && { err "zero-launcher: --replace-service needs --install-service"; return 2; }
  # (--opt id[=value] / --no-opt id / --default-opt id, each a comma list too: kept in options.json for later starts,
  #  as zero-server -Opt / -NoOpt / -DefaultOpt; with --no-saved-options the choices are exactly these, from the
  #  defaults, and nothing is saved: Docker's OPTIONS, docker/entrypoint.sh)
  [ "$no_saved" = 1 ] && OPTC=()
  if [ ${#opt_args[@]} -gt 0 ]; then
    local i ids id val
    for ((i = 0; i < ${#opt_args[@]}; i += 2)); do
      IFS=',' read -r -a ids <<< "${opt_args[$((i + 1))]}"
      for a in "${ids[@]}"; do
        [ -n "$a" ] || continue
        id=${a%%=*}; val=""; [ "$id" != "$a" ] && val=${a#*=}
        case "${opt_args[$i]}" in
          --opt) opt_set "$id" "${val:-on}" || return 2; CLI_ON+=("$id");;
          --no-opt) [ -z "$val" ] || { err "zero-launcher: --no-opt $a: just the id"; return 2; }; opt_set "$id" off || return 2;;
          --default-opt) opt_set "$id" "" || return 2;;
        esac
      done
    done
    # (refused before it is saved: an option that cannot apply to --model never reaches options.json)
    if [ -n "$arg_model" ] && m_known "$arg_model"; then cli_check "$arg_model" || return 2; fi
    [ "$no_saved" = 1 ] || opt_save || { err "zero-launcher: cannot write $STATE_DIR/options.json"; return 1; }
  fi
  case "$mode" in
    options) opt_list; return $?;;
    model_dirs) model_dirs "$md_add" "$md_rm"; return $?;;
    logs)  # (the server's log: the last 40 lines, its path first)
      local lf; lf="$(logs_dir)/server.log"
      if [ ! -f "$lf" ]; then say "no log yet ($lf): the server has not been started"; return 0; fi
      say "($lf; earlier runs' next to it)"; tail -n 40 "$lf"; return 0;;
    check) print_check; return $?;;
    list) list_models; return 0;;
    download)
      m_known "$arg_dl" || { err "zero-launcher: --download: no model \"$arg_dl\" (--list shows the keys)"; return 2; }
      model_path "$arg_dl" >/dev/null || download "$arg_dl" "$YES"; return $?;;
  esac
  if ! run_check; then
    say "missing:"; printf '%s\n' "${CHECK_LINES[@]}" | grep MISS | sed 's/^  MISS  /  /'
    say "(./zero-launcher.sh --check lists everything)"
    [ -x "$SERVERZ" ] || return 1
  fi
  if [ -n "$arg_model" ]; then
    if m_known "$arg_model"; then start_key "$arg_model"; return $?; fi
    [ -f "$arg_model" ] || { err "zero-launcher: --model: no catalog model or file \"$arg_model\" (--list shows the keys)"; return 2; }
    server_cmd "$(readlink -f "$arg_model")" "$MTP"
    if [ "$INSTALL_SERVICE" = 1 ]; then install_service; else run_server; fi
    return $?
  fi
  [ "$INSTALL_SERVICE" = 1 ] && { err "zero-launcher: --install-service needs --model"; return 2; }
  if tui_ok && [ ${#CT_IDS[@]} -gt 0 ]; then tui; else menu; fi
}

# (sourced by tests/test_zero_launcher.sh for its functions; run as a program otherwise)
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
  exit $?
fi
