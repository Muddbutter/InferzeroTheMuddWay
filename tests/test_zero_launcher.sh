#!/usr/bin/env bash
# The Linux launcher's service writes and its refusal before a download: in a temporary folder, no GPU, no network,
# no systemctl. Run by CI.
#   bash tests/test_zero_launcher.sh
set -u
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../zero-launcher.sh
source "$root/zero-launcher.sh"
pass=0 fail=0
ok() { if [ "$1" -eq 0 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL: $2"; fi; }

setup() {  # a fresh config folder and the launcher's defaults for --install-service
  T=$(mktemp -d)
  export XDG_CONFIG_HOME="$T"
  unset HF_HOME HF_HUB_CACHE XDG_CACHE_HOME HF_TOKEN_PATH
  SERVICE_NAME=inferzero REPLACE_SERVICE=0 HOST=127.0.0.1 INSTALL_SERVICE=1
  UNIT="$T/systemd/user/inferzero.service"
  CMD=(/usr/bin/serverz --model /models/a.gguf)
}
teardown() { rm -rf "$T"; }
install() { OUT=$(install_service 2>&1); return $?; }
no_temp() { [ -z "$(find "$(dirname "$UNIT")" -name '.inferzero-*' 2>/dev/null)" ]; }

# 1. a fresh unit is written, no temporary file left
setup; install; rc=$?
grep -q 'ExecStart="/usr/bin/serverz"' "$UNIT" && [ $rc -eq 0 ] && no_temp; ok $? "fresh unit"; teardown
# 2. an existing unit is left unchanged, and the message names --replace-service
setup; mkdir -p "$(dirname "$UNIT")"; printf 'private Pi unit\n' > "$UNIT"; install; rc=$?
[ $rc -eq 1 ] && [ "$(cat "$UNIT")" = "private Pi unit" ] && [[ "$OUT" == *--replace-service* ]]; ok $? "existing unit unchanged"; teardown
# 3. --replace-service replaces it
setup; install; REPLACE_SERVICE=1; CMD=(/usr/bin/other); install; rc=$?
[ $rc -eq 0 ] && grep -q 'ExecStart="/usr/bin/other"' "$UNIT"; ok $? "explicit replacement"; teardown
# 4. replacing a symlinked unit replaces the link itself, never writes through it
setup; printf 'keep me' > "$T/private.txt"; mkdir -p "$(dirname "$UNIT")"; ln -s "$T/private.txt" "$UNIT"
install; rc1=$?; REPLACE_SERVICE=1; install; rc2=$?
[ $rc1 -eq 1 ] && [ $rc2 -eq 0 ] && [ ! -L "$UNIT" ] && [ "$(cat "$T/private.txt")" = "keep me" ]; ok $? "symlink not followed"; teardown
# 5. a separate name: its own unit and printed commands
setup; SERVICE_NAME=inferzero-package-test.service; install; rc=$?
[ $rc -eq 0 ] && [ ! -e "$UNIT" ] && [ -f "$T/systemd/user/inferzero-package-test.service" ] && [[ "$OUT" == *"restart inferzero-package-test"* ]]
ok $? "separate name"; teardown
# 6. names that are not plain unit names are refused, nothing created
setup; r=0
for n in '' '../pi' '/tmp/pi' $'pi\n' 'pi unit' '-pi'; do SERVICE_NAME=$n; install && r=1; done
[ $r -eq 0 ] && [ ! -e "$T/systemd" ]; ok $? "bad names refused"; teardown
# 7. a line break in the command is refused
setup; CMD=($'/bin/server\nExecStartPre=/bin/false'); install; rc=$?
[ $rc -eq 1 ] && [ ! -e "$UNIT" ]; ok $? "line break in command"; teardown
# 8. a line break in a cache variable is refused
setup; export HF_HOME=$'/cache\nExecStartPre=/bin/false'; install; rc=$?
[ $rc -eq 1 ] && [ ! -e "$UNIT" ]; ok $? "line break in cache variable"; unset HF_HOME; teardown
# 9. systemd's specifiers and variables escaped in ExecStart=
setup; CMD=(/bin/serverz 'a$b%"c\d'); install; rc=$?
[ $rc -eq 0 ] && grep -qF '"a$$b%%\"c\\d"' "$UNIT"; ok $? "ExecStart escaping"; teardown
# 10. a unit that appears while the unit is written is kept (the create never replaces a file)
setup; mkdir -p "$(dirname "$UNIT")"
ln() { printf 'other writer' > "$UNIT"; return 1; }  # (another writer wins the race)
install; rc=$?; unset -f ln
[ $rc -eq 1 ] && [ "$(cat "$UNIT")" = "other writer" ] && no_temp; ok $? "racing writer kept"; teardown
# 11. Environment= keeps $ (systemd expands it in ExecStart= only), % doubled
setup; export HF_HOME='/cache/$x%y'; install; rc=$?
[ $rc -eq 0 ] && grep -qF 'Environment="HF_HOME=/cache/$x%%y"' "$UNIT"; ok $? "Environment keeps \$"; unset HF_HOME; teardown
# 12. a file system without hard links: the exclusive create writes the unit, no temporary file left
setup; ln() { return 1; }
install; rc=$?; unset -f ln
[ $rc -eq 0 ] && grep -q 'ExecStart="/usr/bin/serverz"' "$UNIT" && no_temp; ok $? "no hard links"; teardown
# 13. --install-service with an existing unit refuses before any download
setup; mkdir -p "$(dirname "$UNIT")"; printf 'private Pi unit\n' > "$UNIT"
model_path() { return 1; }
download() { echo "DOWNLOADED"; return 0; }
OUT=$(start_key 4b 2>&1); rc=$?
unset -f model_path download; source "$root/zero-launcher.sh"
[ $rc -eq 1 ] && [[ "$OUT" != *DOWNLOADED* ]] && [ "$(cat "$UNIT")" = "private Pi unit" ]; ok $? "refused before download"; teardown

# 14. the switches' rules against tests/launcher_switch_cases.tsv, the cases scripts/launcher-switches.ps1 is
#     checked against (QUEUE row 15 step 3): the reason line each option shows, "-" when it applies
cat_load
nc=0 bad=0
while IFS=$'\t' read -r words pc gpus drafting choices id expect; do
  [ "$words" = words ] && continue
  OPTC=()
  if [ "$choices" != - ]; then IFS=',' read -r -a cs <<< "$choices"; for c in "${cs[@]}"; do OPTC[${c%%=*}]=${c#*=}; done; fi
  SW_WORDS=${words//;/ }; SW_PC=${pc//;/ }; [ "$pc" = - ] && SW_PC=""; SW_GPUS=$gpus; SW_DRAFTING=$drafting
  got=$(sw_reason "$id"); [ -z "$got" ] && got=-
  nc=$((nc + 1))
  if [ "$got" != "$expect" ]; then bad=1; echo "  $id for [$words] on [$pc], gpus $gpus, drafting $drafting, $choices: '$got', expected '$expect'"; fi
done < <(tr -d '\r' < "$root/tests/launcher_switch_cases.tsv")
[ $bad = 0 ] && [ $nc -gt 20 ]; ok $? "switch reasons ($nc cases)"
# 15. every switch the menu draws has a title, a tip and a line under it that names the output; qsa is a choice
bad=0
for id in "${CT_IDS[@]}"; do
  sw_value "$id" && continue
  [ -n "${CT_tip[$id]}" ] && [ -n "${CT_title[$id]}" ] || { bad=1; echo "  $id: no tip or title"; }
  [[ "$(sw_tipline "$id")" =~ (Same\ output|Output\ checked\ against\ full\ precision)\. ]] || { bad=1; echo "  $id: line under the tip: $(sw_tipline "$id")"; }
done
[ "$(sw_choices qsa)" = "on:Sparse off:Full" ] || { bad=1; echo "  qsa's choices: $(sw_choices qsa)"; }
[ "$(sw_default h8_exact)" = off ] && [ "$(sw_default xcpu)" = auto ] && [ "$(sw_default copy_draft)" = off ] || { bad=1; echo "  defaults"; }
[ $bad = 0 ]; ok $? "tips and choices"
# 16. the full-screen menu fits 80 columns, for every model, with the options pane focused and the folded ones shown
model_path() { return 1; }
OPTC=(); GPUS=1; SW_PC="igpu-xe-lpg"; bad=0
TUI_KEYS=(); for e in "${CATALOG[@]}"; do TUI_KEYS+=("${e%%|*}"); done
for ((s = 0; s < ${#TUI_KEYS[@]}; s++)); do
  for na in 0 1; do
    for h in 20 24; do  # (and within the terminal's height, the focus on the last option: the list scrolls)
      TUI_SEL=$s TUI_FOC=99 TUI_PANE=opts TUI_SHOWNA=$na TUI_H=$h TUI_OFF=0; n=0
      while IFS= read -r l; do
        l=${l//$'\033[K'/}; l=${l//$'\033[H'/}; l=${l//$'\033[J'/}
        n=$((n + 1))
        [ ${#l} -le 80 ] || { bad=1; echo "  ${TUI_KEYS[$s]}: a line of ${#l} columns: $l"; }
      done < <(tui_draw)
      [ $n -le $h ] || { bad=1; echo "  ${TUI_KEYS[$s]}: $n lines for a $h-row terminal (all shown: $na)"; }
    done
  done
done
unset -f model_path; source "$root/zero-launcher.sh"
[ $bad = 0 ]; ok $? "menu within 80 columns and the terminal's height"
# 16b. a greyed option is not passed: Copy drafting chosen, the 27B started with Drafting off; on with Drafting on
cat_load; OPTC=(); OPTC[copy_draft]=on; GPUS=1; SW_PC=""; PORT=8090; HOST=""; EXTRA=()
SW_WORDS=$(m_words 27b); SW_DRAFTING=0; SW_SKIP=1; SEL_DRAFT=off; server_cmd /m.gguf 1; a="${CMD[*]}"
SW_DRAFTING=1; SEL_DRAFT=both; server_cmd /m.gguf 1; b="${CMD[*]}"; SW_SKIP=0; SEL_DRAFT=""; OPTC=()
[[ "$a" != *--copy-draft* ]] && [[ "$a" != *--mtp* ]] && [[ "$b" == *--copy-draft* ]] && [[ "$b" == *--mtp* ]]
ok $? "greyed option left out ($a | $b)"
# 16b2. #81: after a toggle the focus stays on the same option when the folded list changes (Drafting off: the
#       options that need it leave the list, and the focus by index landed on another option)
cat_load; OPTC=(); GPUS=1; SW_PC=""; TUI_SHOWNA=0; TUI_NA_SET=1
TUI_KEYS=(); for e in "${CATALOG[@]}"; do TUI_KEYS+=("${e%%|*}"); done
for ((s = 0; s < ${#TUI_KEYS[@]}; s++)); do [ "${TUI_KEYS[$s]}" = 27b ] && TUI_SEL=$s; done
k=27b; unset 'TUI_DRAFT[27b]'; tui_state $k; tui_items $k; before=("${TUI_ITEMS[@]}"); last=${TUI_ITEMS[$((${#TUI_ITEMS[@]} - 2))]}
TUI_FOC=$((${#TUI_ITEMS[@]} - 2)); TUI_DRAFT[$k]=off; tui_keep_focus "$last"; after=("${TUI_ITEMS[@]}")
[ "${TUI_ITEMS[$TUI_FOC]}" = "$last" ] && [ ${#after[@]} -le ${#before[@]} ]
ok $? "the focus follows the option after a toggle (${#before[@]} -> ${#after[@]} items, on $last)"
unset 'TUI_DRAFT[27b]'; TUI_NA_SET=0
# 16c. #39: arguments after -- that give --ctx / --think-budget / --session-cache-gb replace the launcher's own;
#      without them the launcher's defaults stay
cat_load; OPTC=(); GPUS=1; SW_PC=""; PORT=8090; HOST=""; SEL_DRAFT=off
EXTRA=(--ctx 50000 --think-budget 2048); server_cmd /m.gguf 0; a="${CMD[*]}"
EXTRA=(-c 4096 --session-cache-gb 0); server_cmd /m.gguf 0; b="${CMD[*]}"
EXTRA=(); server_cmd /m.gguf 0; c="${CMD[*]}"; SEL_DRAFT=""
[[ "$a" != *"--ctx auto"* ]] && [[ "$a" == *"--ctx 50000"* ]] && [[ "$a" != *"--think-budget 8192"* ]] &&
  [[ "$a" == *"--session-cache-gb 8"* ]] && [[ "$b" != *"--ctx auto"* ]] && [[ "$b" != *"--session-cache-gb 8"* ]] &&
  [[ "$b" == *"--think-budget 8192"* ]] && [[ "$c" == *"--ctx auto"* ]] && [[ "$c" == *"--think-budget 8192"* ]]
ok $? "the user's --ctx / --think-budget / --session-cache-gb after -- replace the launcher's ($a | $b | $c)"
# 16d. #39: the launcher's own --ctx N takes the place of --ctx auto (an --ctx after -- still wins); a bad value is
#      refused; --help points to serverz's own options
SEL_DRAFT=off; CTX=32768; EXTRA=(); server_cmd /m.gguf 0; a="${CMD[*]}"
EXTRA=(--ctx 4096); server_cmd /m.gguf 0; b="${CMD[*]}"; CTX=""; EXTRA=(); SEL_DRAFT=""
OUT=$(bash "$root/zero-launcher.sh" --ctx lots --check 2>&1); rc=$?
H=$(bash "$root/zero-launcher.sh" --help 2>&1)
[[ "$a" == *"--ctx 32768"* ]] && [[ "$a" != *"--ctx auto"* ]] && [[ "$b" == *"--ctx 4096"* ]] && [[ "$b" != *"32768"* ]] &&
  [ $rc -eq 2 ] && [[ "$OUT" == *"--ctx: a number of tokens"* ]] && [[ "$H" == *"--ctx N"* ]] && [[ "$H" == *"serverz --help all"* ]]
ok $? "--ctx N of the launcher ($a | $b), a bad --ctx refused, --help names serverz --help all"
# 17. --opt with --model: an option the model cannot take is refused with the reason, before options.json is written
T=$(mktemp -d); OUT=$(ARC_DATA_DIR="$T" bash "$root/zero-launcher.sh" --opt copy_draft --model flash 2>&1); rc=$?
[ $rc -eq 2 ] && [[ "$OUT" == *"--opt copy_draft: for models with MTP only"* ]] && [ ! -f "$T/options.json" ]; ok $? "--opt refused for the model"
rm -rf "$T"
# 18. best-setups.tsv (the automatic preset): two B-series cards and the 27B get --gpus 2; two unlike cards none; a
#     35B on a Panther Lake B390 the larger chunks; --no-auto none; a --gpus given wins; two of one family with
#     other device IDs (no multi-same: a B60 + B70) none
sw_discrete() { printf '2'; }
cat_load; OPTC=(); NO_AUTO=0; SEL_DRAFT=""
SW_PC="b-series multi-gpu multi-same"; GPUS=0; best_auto 27b >/dev/null; a=$GPUS
SW_PC="b-series multi-gpu"; GPUS=0; best_auto 27b >/dev/null; b=$GPUS
SW_PC="b-series multi-gpu multi-same"; GPUS=0; NO_AUTO=1; best_auto 27b >/dev/null; c=$GPUS; NO_AUTO=0
SW_PC="b-series multi-gpu multi-same"; GPUS=1; best_auto 27b >/dev/null; d=$GPUS
SW_PC="igpu-xe2 igpu-ptl"; GPUS=0; best_auto 35b >/dev/null; e=${OPTC[moe_big_chunk]:-}
[ "$a" = 2 ] && [ "$b" = 0 ] && [ "$c" = 0 ] && [ "$d" = 1 ] && [ "$e" = on ]
ok $? "best-setups rows (two B-series: $a, unlike: $b, --no-auto: $c, --gpus 1 given: $d, B390 35B chunks: $e)"
OPTC=(); GPUS=0; SW_PC=""; unset -f sw_discrete; source "$root/zero-launcher.sh"

echo "test_zero_launcher: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
