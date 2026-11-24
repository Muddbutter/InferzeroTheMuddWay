# The launchers' option catalog, scripts/options.tsv, is consistent (QUEUE row 15 step 1, 2026-11-22):
#   - every option has a tip: 1-3 plain sentences, at most 450 characters, each ending in a full stop;
#   - relates: "needs:" / "has:" / "or:" / "not:" entries, separated by ";", each naming an option id of the catalog,
#     or mtp (the --mtp flag) or gpus2 (--gpus 2 or more). "has:B" on A needs "needs:A" on B and the other way round;
#     "not:" and "or:" go both ways;
#   - applies: words of the model (dense, moe, moe_std (a standard Qwen MoE, not Flash Next), qsa, flashnext, mtp, h8: a dense Qwen3.5 / 3.8) or of the PC (scripts/pc-kinds.ps1's words:
#     b-series, alchemist, igpu-140t, igpu-xe2, igpu-xe-lpg, multi-gpu), separated by ";"; empty: any model, any PC;
#   - variant: "suffix:value" entries, separated by ";": a suffix of lowercase letters, digits and dashes starting with
#     a dash, unique over the catalog, and a value (the option's value then, "on" for a flag without one).
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\test_launcher_options.ps1
# Prints "test_launcher_options: all passed" or each problem, and exits 1 then. Windows PowerShell 5.1 and 7.
param([string]$Catalog = "")  # (another catalog file: the test's own check of itself)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$tsv = if ($Catalog) { $Catalog } else { Join-Path $root "scripts\options.tsv" }
$lines = [IO.File]::ReadAllText($tsv, (New-Object Text.UTF8Encoding $false)) -split "\r?\n" | Where-Object { $_ -ne "" }
$cols = $lines[0] -split "`t"
$rows = @(foreach ($l in $lines[1..($lines.Count - 1)]) {
  $v = $l -split "`t"; $d = @{}
  for ($i = 0; $i -lt $cols.Count; $i++) { $d[$cols[$i]] = $(if ($i -lt $v.Count) { $v[$i].Trim() } else { "" }) }
  if ($v.Count -ne $cols.Count) { $d["_cols"] = $v.Count }
  $d
})
$bad = New-Object System.Collections.Generic.List[string]
foreach ($c in "id", "tip", "relates", "applies", "variant") { if ($cols -notcontains $c) { $bad.Add("no column '$c'") } }
$ids = @($rows | ForEach-Object { $_.id })
$modelWords = @("dense", "moe", "moe_std", "qsa", "flashnext", "mtp", "h8")
$pcWords = @("b-series", "alchemist", "igpu-140t", "igpu-xe2", "igpu-xe-lpg", "multi-gpu")
$rel = @{}  # "kind|from|to" -> $true
$suffixes = @{}
function Split-List([string]$s) { @($s -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
foreach ($r in $rows) {
  $id = $r.id
  if ($r.ContainsKey("_cols")) { $bad.Add("${id}: $($r._cols) fields, the header has $($cols.Count)") }
  if (@($ids | Where-Object { $_ -eq $id }).Count -gt 1) { $bad.Add("${id}: the id appears twice") }
  # tip
  $t = $r.tip
  if (-not $t) { $bad.Add("${id}: no tip") }
  else {
    $n = ([regex]::Matches($t, '[.!?](\s|$)')).Count
    if ($n -lt 1 -or $n -gt 3) { $bad.Add("${id}: the tip has $n sentences (1-3)") }
    if ($t.Length -gt 450) { $bad.Add("${id}: the tip is $($t.Length) characters (at most 450)") }
    if ($t -notmatch '[.!?]$') { $bad.Add("${id}: the tip does not end a sentence") }
  }
  # relates
  foreach ($e in (Split-List $r.relates)) {
    if ($e -notmatch '^(needs|has|or|not):([a-z0-9_]+)$') { $bad.Add("${id}: relates entry '$e' (needs: / has: / or: / not: and an id)"); continue }
    $k = $Matches[1]; $to = $Matches[2]
    if ($ids -notcontains $to -and -not ($k -eq "needs" -and ($to -eq "mtp" -or $to -eq "gpus2"))) { $bad.Add("${id}: relates '$e' names no option") ; continue }
    if ($to -eq $id) { $bad.Add("${id}: relates '$e' names itself") }
    $rel["$k|$id|$to"] = $true
  }
  # applies
  foreach ($w in (Split-List $r.applies)) {
    if ($modelWords -notcontains $w -and $pcWords -notcontains $w) { $bad.Add("${id}: applies word '$w' is not known ($(($modelWords + $pcWords) -join ', '))") }
  }
  # variant
  foreach ($e in (Split-List $r.variant)) {
    if ($e -notmatch '^(-[a-z0-9][a-z0-9-]*):(\S+)$') { $bad.Add("${id}: variant '$e' (-suffix:value)"); continue }
    $s = $Matches[1]
    if ($suffixes.ContainsKey($s)) { $bad.Add("${id}: variant suffix $s is also $($suffixes[$s])'s") } else { $suffixes[$s] = $id }
  }
}
# the relations both ways
foreach ($key in @($rel.Keys)) {
  $p = $key -split '\|'; $k = $p[0]; $a = $p[1]; $b = $p[2]
  if ($k -eq "has" -and -not $rel["needs|$b|$a"]) { $bad.Add("${a}: has:$b, but $b has no needs:$a") }
  if ($k -eq "needs" -and $ids -contains $b -and -not $rel["has|$b|$a"]) { $bad.Add("${a}: needs:$b, but $b has no has:$a") }
  if (($k -eq "not" -or $k -eq "or") -and -not $rel["$k|$b|$a"]) { $bad.Add("${a}: ${k}:$b, but $b has no ${k}:$a") }
}
if ($bad.Count) {
  $bad | ForEach-Object { Write-Output "test_launcher_options: $_" }
  exit 1
}
Write-Output "test_launcher_options: all passed ($($rows.Count) options, $($suffixes.Count) variants)"
