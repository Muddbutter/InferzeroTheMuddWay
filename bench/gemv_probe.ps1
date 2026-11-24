# Is the single-token IQ3_S GEMV limited by reading the weights or by decoding them? (issue #12)
#
#   .\bench\gemv_probe.ps1 -Model C:\hf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
#
# Runs `runz <model> --bench-pp 128 --n 32 --prof-gen` three times and prints GPU ms per launch of the FFN's gate /
# up / down GEMVs, and the bandwidth that gives:
#   normal   the shipped gemv (local-window path; ARC_GV_IQ3S is not set)
#   reads    ARC_GV_PROBE=1: same reads, no decode (the weights' bytes used as they are)
#   decode   ARC_GV_PROBE=2: full decode, but of the same two superblocks over and over (no streaming)
# The outputs of the probes are wrong on purpose (garbage tokens): only the times count.
# Reading: reads close to ~540 GB/s and normal at ~400 -> the decode is the limit (QUEUE steps 1 / 2 of issue #12);
# reads also at ~400 -> the access pattern is (step 3). decode ms per launch near the normal one -> ALU-bound.
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [double]$PeakGBs = 0   # the card's measured GB/s (runz prints it); 0 = only ms
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found" }
if (Get-Process serverz, runz -ErrorAction SilentlyContinue) { Write-Warning "a serverz / runz is running: stop it first (device-lost risk)" }
$inv = [Globalization.CultureInfo]::InvariantCulture

function Run([string]$probe) {
  Remove-Item Env:ARC_GV_PROBE -ErrorAction SilentlyContinue
  Remove-Item Env:ARC_GV_IQ3S -ErrorAction SilentlyContinue
  if ($probe) { $env:ARC_GV_PROBE = $probe }
  try { $o = & $runz $Model --bench-pp 128 --n 32 --prof-gen 2>&1 | Out-String }
  finally { Remove-Item Env:ARC_GV_PROBE -ErrorAction SilentlyContinue }
  $r = @{ step = $null; gate_up = $null; gate = $null; up = $null; down = $null }
  if ($o -match 'tg steps: min [\d.]+, median ([\d.]+)') { $r.step = [double]::Parse($Matches[1], $inv) }
  # (gate_up: gate and up in one launch, where both have the same type; gate / up: the separate launches)
  foreach ($k in "gate_up", "gate", "up", "down") {
    if ($o -match "gemm_$k\s+([\d.]+) ms\s+[\d.]+%\s+\((\d+) launches\)") {
      $r[$k] = [double]::Parse($Matches[1], $inv) / [double]$Matches[2] * 1000.0   # us per launch
    }
  }
  return $r
}

$rows = [ordered]@{ normal = ""; reads = "1"; decode = "2" }
Write-Host ("{0,-8} {1,10} {2,10} {3,10} {4,10} {5,10}" -f "variant", "step ms", "gate_up us", "gate us", "up us", "down us")
foreach ($name in $rows.Keys) {
  $r = Run $rows[$name]
  Write-Host ("{0,-8} {1,10:F2} {2,10:F1} {3,10:F1} {4,10:F1} {5,10:F1}" -f $name, $r.step, $r.gate_up, $r.gate, $r.up, $r.down)
}
Write-Host ""
Write-Host "gate / up read ~38 MB per launch on the 27B IQ3_S (17,408 x 5,120 x 3.44 bit): GB/s = 38.3e6 / (us * 1e3);"
Write-Host "gate_up reads both: GB/s = 76.6e6 / (us * 1e3)."
Write-Host "Paste this table in the issue."
