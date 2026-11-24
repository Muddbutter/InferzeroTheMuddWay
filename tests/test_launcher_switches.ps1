# The launchers' switches (QUEUE row 15 step 3): scripts\launcher-switches.ps1, the rules zero-launcher.ps1 draws with,
# against tests\launcher_switch_cases.tsv (a model's words, the PC's words, the GPUs, Drafting, the choices changed:
# the line an option shows, "-" when it applies). Also: every option the launchers draw as a switch has a tip and the
# line under it, and a choose-one option gets its two choices. tests/test_zero_launcher.sh checks zero-launcher.sh
# against the same cases.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\test_launcher_switches.ps1
# Prints "test_launcher_switches: all passed" or each problem, and exits 1 then. Windows PowerShell 5.1 and 7.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
. (Join-Path $root "scripts\launcher-switches.ps1")
$cat = @(Read-OptionCatalog (Join-Path $root "scripts\options.tsv"))
$bad = New-Object System.Collections.Generic.List[string]
function Split-Dash([string]$s) { if ($s -eq "-" -or -not $s) { @() } else { @($s -split '[;,]' | Where-Object { $_ }) } }
$lines = [IO.File]::ReadAllText((Join-Path $PSScriptRoot "launcher_switch_cases.tsv"), (New-Object Text.UTF8Encoding $false)) -split "\r?\n" | Where-Object { $_ -ne "" }
$n = 0
foreach ($l in $lines[1..($lines.Count - 1)]) {
  $f = $l -split "`t"
  if ($f.Count -ne 7) { $bad.Add("case '$l': $($f.Count) fields, not 7"); continue }
  $ch = @{}
  foreach ($c in (Split-Dash $f[4])) { $p = $c -split '=', 2; $ch[$p[0]] = $p[1] }
  $st = @{ words = @(Split-Dash $f[0]); pc = @(Split-Dash $f[1]); gpus = [int]$f[2]; drafting = $f[3] -eq "1"; choice = $ch }
  $o = Get-SwRow $cat $f[5]
  if (-not $o) { $bad.Add("case '$l': no option $($f[5])"); continue }
  $got = Get-SwReason $cat $o $st
  $want = $(if ($f[6] -eq "-") { "" } else { $f[6] })
  if ($got -ne $want) { $bad.Add("$($f[5]) for [$($f[0])] on [$($f[1])], gpus $($f[2]), drafting $($f[3]), $($f[4]): '$got', expected '$want'") }
  $n++
}
# every switch drawn: a tip and the line under it
$drawn = 0
foreach ($o in $cat) {
  if (Test-SwValue $o) { continue }
  $drawn++
  if (-not $o.tip) { $bad.Add("$($o.id): drawn as a switch, but has no tip") }
  if (-not $o.title) { $bad.Add("$($o.id): drawn as a switch, but has no title") }
  $t = Get-SwTipLine $cat $o
  if ($t -notmatch '(Same output|Output checked against full precision)\.') { $bad.Add("$($o.id): the line under its tip says nothing of the output ('$t')") }
}
$q = @(Get-SwChoices (Get-SwRow $cat "qsa"))
if (($q | ForEach-Object { "$($_.label)=$($_.value)" }) -join "," -ne "Sparse=on,Full=off") { $bad.Add("qsa's choices: '$(($q | ForEach-Object { "$($_.label)=$($_.value)" }) -join ",")', expected Sparse=on,Full=off") }
if (@(Get-SwChoices (Get-SwRow $cat "mtp_vocab")).Count) { $bad.Add("mtp_vocab came out as a choose-one option") }
foreach ($p in @(@("copy_draft", "off"), @("xcpu", "auto"), @("tp_p2p", "auto"), @("h8_exact", "off"), @("batch", "auto"))) {
  $d = Get-OptDefault (Get-SwRow $cat $p[0])
  if ($d -ne $p[1]) { $bad.Add("$($p[0])'s default read as '$d', expected '$($p[1])'") }
}
# the automatic preset's rows (scripts\best-setups.tsv, the launcher spec B6): which rows a PC and a model get
$best = @(Read-BestSetups (Join-Path $root "scripts\best-setups.tsv"))
if ($best.Count -lt 5) { $bad.Add("best-setups.tsv: $($best.Count) rows read") }
foreach ($c in @(
    @("B390 + 35B", @{ pc = @("igpu-xe2", "igpu-ptl"); os = "windows"; short = "35b"; discrete = 1; notfit = $false }, "ptl-35b"),
    @("Lunar Lake + 35B", @{ pc = @("igpu-xe2"); os = "windows"; short = "35b"; discrete = 1; notfit = $false }, ""),
    @("125H + 27B", @{ pc = @("igpu-xe-lpg"); os = "windows"; short = "27b"; discrete = 1; notfit = $false }, "mtl-npu"),
    @("125H + 27B on Linux", @{ pc = @("igpu-xe-lpg"); os = "linux"; short = "27b"; discrete = 1; notfit = $false }, ""),
    @("2 x B60 + 27B", @{ pc = @("b-series", "multi-gpu", "multi-same"); os = "windows"; short = "27b"; discrete = 2; notfit = $false }, "bseries-tp"),
    @("B60 + B70 + 27B", @{ pc = @("b-series", "multi-gpu"); os = "windows"; short = "27b"; discrete = 2; notfit = $false }, ""),
    @("B60 + A770 + 27B", @{ pc = @("alchemist", "b-series", "multi-gpu"); os = "windows"; short = "27b"; discrete = 2; notfit = $false }, ""),
    @("B580 + 27B not fitting", @{ pc = @("b-series"); os = "windows"; short = "27b"; discrete = 1; notfit = $true }, "notfit-nomtp"),
    @("2 x B60 + Gemma 12B", @{ pc = @("b-series", "multi-gpu", "multi-same"); os = "windows"; short = "g12"; discrete = 2; notfit = $false }, ""))) {
  $got = (@(Get-BestSetups $best $c[1]) | ForEach-Object { $_.id }) -join ","
  if ($got -ne $c[2]) { $bad.Add("best-setups for $($c[0]): '$got', expected '$($c[2])'") }
}
$n += 9
if ($bad.Count) {
  $bad | ForEach-Object { Write-Output "test_launcher_switches: $_" }
  exit 1
}
Write-Output "test_launcher_switches: all passed ($n cases, $drawn switches with tips)"
