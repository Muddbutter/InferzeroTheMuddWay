#!/usr/bin/env bash
# The Linux launcher's model list (zero-launcher.sh CATALOG) against the Windows one (scripts/model-catalog.ps1): every
# key the launcher offers must have the same repository, files, drafter and extra file there. Run by CI.
#   bash tests/check_catalog_sync.sh
set -u
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../zero-launcher.sh
source "$root/zero-launcher.sh"
# (each entry of the .ps1 on one line: the text from one `[pscustomobject]@{` to the next)
ps=$(tr '\n' ' ' < "$root/scripts/model-catalog.ps1" | sed 's/\[pscustomobject\]@{/\n&/g' | grep '^\[pscustomobject\]@{')
field() {  # ENTRY-LINE NAME: the first `NAME = "..."` value
  printf '%s' "$1" | sed -n "s/.*[{; ]$2 *= *\"\([^\"]*\)\".*/\1/p" | head -1
}
bad=0 n=0
for e in "${CATALOG[@]}"; do
  IFS='|' read -r -a f <<< "$e"
  key=${f[0]} n=$((n + 1))
  line=$(printf '%s\n' "$ps" | grep "key *= *\"$key\"" | head -1)
  if [ -z "$line" ]; then echo "$key: not in model-catalog.ps1"; bad=$((bad + 1)); continue; fi
  # (the entry's own repo: the one right after its key and name; a drafter's or extra file's repo comes later)
  repo=$(printf '%s' "$line" | sed -n "s/.*key *= *\"$key\"; *name *= *\"[^\"]*\"; *repo *= *\"\([^\"]*\)\".*/\1/p")
  files=$(printf '%s' "$line" | sed -n 's/.*files *= *@(\([^)]*\)).*/\1/p' | grep -o '"[^"]*"' | tr -d '"' | tr '\n' ' ')
  files=${files% }
  dr=""; df=""; xr=""; xf=""; xi=""
  if [[ "$line" == *"drafter = @{"* ]]; then
    d=${line#*drafter = @\{}; d=${d%%\}*}
    dr=$(field " $d" repo); df=$(field " $d" file)
  fi
  if [[ "$line" == *"extra = @{"* ]]; then
    x=${line#*extra = @\{}; x=${x%%\}*}
    xr=$(field " $x" repo); xf=$(field " $x" file); xi=$(field " $x" into)
  fi
  chk() {
    if [ "$2" != "$3" ]; then echo "$key: $1 differs: launcher '$2', model-catalog.ps1 '$3'"; bad=$((bad + 1)); fi
  }
  chk repo "${f[2]:-}" "$repo"
  chk files "${f[7]:-}" "$files"
  chk "drafter repo" "${f[8]:-}" "$dr"
  chk "drafter file" "${f[9]:-}" "$df"
  chk "extra repo" "${f[10]:-}" "$xr"
  chk "extra file" "${f[11]:-}" "$xf"
  chk "extra into" "${f[12]:-}" "$xi"
done
echo "catalog sync: $n models checked, $bad difference(s)"
[ "$bad" -eq 0 ]
