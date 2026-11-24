# The options that move the logits, each against the f32 CPU reference (tools/ref_forward), on your GPU, in one table:
# the KL mean and max and the same top token over every position, so a contributor can check the new options' accuracy
# on their own card (the user, 2026-11-21: "make the user test KL against f32 reference for the new options").
# The rows that apply to the model, found from its first run's start-up lines:
#   - the default against its reference (the row every other one is read against);
#   - the two QSA models (the QSA 27B, Flash Next): the other attention mode (ARC_QSA=1 / ARC_QSA=0) against its own
#     reference and against the default's, the selection cut to 64 keys on both sides (ARC_QSA_TOPK / REF_TOPK) so
#     that a few hundred tokens are sparse (it starts at 2,048 keys otherwise);
#   - where H8 runs: ARC_H8_Q2=1 (More precise prompt reading);
#   - Flash Next: its fast mode (ARC_FN_PRECISE=0; precise mode is the default since 2026-11-22);
#   - with an MTP block: ARC_MTP_VOCAB=32768 (Faster drafting) must give the same text as plain --mtp (a text check).
# The reference reads the model's weights row by row from the file, so it needs a few GB of RAM besides the file's
# pages, and CPU time: about 30 minutes for 480 tokens of the 27B on a laptop's 4 + 8 cores, less on a desktop. Each
# reference is made once and kept (-Fresh makes them again). Windows PowerShell 5.1 or 7; no Python.
#   powershell -ExecutionPolicy Bypass -File bench\kl_options.ps1 -Model <model.gguf> [-Tokens 480]
# Then paste the table it prints into the GitHub issue.
param([Parameter(Mandatory = $true)][string]$Model, [int]$Tokens = 480, [switch]$Fresh)
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$runz = Join-Path $root "bin\runz.exe"
$ref = Join-Path $root "bin\ref_forward.exe"
foreach ($e in $runz, $ref) { if (-not (Test-Path $e)) { throw "$e not found: run this from an inferzero package (or build it first)" } }
if (-not (Test-Path $Model)) { throw "no model at $Model" }
if (Get-Process serverz, runz -ErrorAction SilentlyContinue) { throw "a serverz / runz is running: stop it first (two programs on one GPU can fail)" }
# (the file's name and its size: a folder per model file, as kl_options.sh, so no model reads another's references)
$work = Join-Path $env:USERPROFILE (".inferzero\kl-options\" + ((Split-Path $Model -Leaf) -replace '[^A-Za-z0-9._-]', '_') + "-" + (Get-Item -LiteralPath $Model).Length)
if ($Fresh) { Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $work | Out-Null
function Stamp([string]$x) { Write-Host ("{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $x) }
function Invoke-Exe([string]$exe, [string[]]$a, [hashtable]$envs) {
  foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
  $o = & $exe @a 2>&1 | Out-String
  foreach ($k in $envs.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
  return $o
}
function Get-TextHash([string]$t) {
  $sha = [Security.Cryptography.SHA256]::Create()
  return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($t))) -replace '-', '').Substring(0, 16)
}
# --- the tokens: the first $Tokens of USAGE.md after its first 2,000 characters (prose, the same on every machine)
$ids = Join-Path $work "tokens$Tokens.ids"
if (-not (Test-Path $ids)) {
  $txt = [IO.File]::ReadAllText((Join-Path $root "USAGE.md"), [Text.Encoding]::UTF8)
  $slice = Join-Path $work "usage.txt"
  [IO.File]::WriteAllText($slice, $txt.Substring(2000, [Math]::Min(40000, $txt.Length - 2000)), (New-Object Text.UTF8Encoding $false))
  $all = Join-Path $work "usage.ids"
  Invoke-Exe $runz @($Model, "--text-file", $slice, "--dump-ids", $all) @{} | Out-Null
  $list = @(([IO.File]::ReadAllText($all, [Text.Encoding]::UTF8) -split '[,\s]+') | Where-Object { $_ -ne "" })
  if ($list.Count -lt $Tokens) { throw "only $($list.Count) tokens from USAGE.md" }
  [IO.File]::WriteAllText($ids, ($list[0..($Tokens - 1)] -join ","), (New-Object Text.UTF8Encoding $false))
}
# --- what the model has: from a first run's start-up lines
$probe = Invoke-Exe $runz @($Model, "--tokens-file", $ids, "--n", "1") @{}
$h8 = $probe -match "H8 on for the prompt GEMMs"
$fn = $probe -match "Flash Next with"
$qsa27 = $probe -match "QSA indexer present"
$qsa = $fn -or $qsa27
Stamp ("model: " + (Split-Path $Model -Leaf) + " | H8 " + $(if ($h8) { "on" } else { "off" }) + " | " + $(if ($fn) { "Flash Next (sparse attention by default)" } elseif ($qsa27) { "QSA 27B (full attention by default)" } else { "no QSA indexer" }))
# --- the references: full attention (REF_DENSE=1, also the plain one for models without QSA) and, for the QSA models,
#     sparse attention at 64 keys
function Get-Ref([string]$name, [hashtable]$envs) {
  $f = Join-Path $work "$name.bin"
  if (-not (Test-Path $f)) {
    Stamp "f32 reference '$name' (CPU, minutes to an hour)..."
    $t0 = Get-Date
    $envs["REF_ALL"] = "1"
    Invoke-Exe $ref @($Model, "@$ids", $f) $envs | Out-File (Join-Path $work "$name.log") -Encoding utf8
    if (-not (Test-Path $f)) { throw "the reference '$name' failed: see $work\$name.log" }
    Stamp ("  done in {0:N0} min" -f ((Get-Date) - $t0).TotalMinutes)
  }
  return $f
}
$refFull = Get-Ref "ref_full" @{ REF_DENSE = "1" }
$refSparse = if ($qsa) { Get-Ref "ref_sparse64" @{ REF_TOPK = "64" } } else { $null }
# --- the rows
$rows = New-Object System.Collections.ArrayList
function Add-Row([string]$opt, [string]$against, [string]$refFile, [hashtable]$envs) {
  $o = Invoke-Exe $runz @($Model, "--tokens-file", $ids, "--kl-ref", $refFile, "--n", "1") $envs
  $m = [regex]::Match($o, 'KL\(ref \|\| inferzero\) = ([0-9.]+) \(mean over (\d+) positions, max ([0-9.]+)\), top token agrees (\d+) / (\d+)')
  # (the share without the reference's near-ties, the rule of 2026-11-24: runz prints it after the strict one)
  $nt = [regex]::Match($o, 'without near-ties \(ref top-2 margin < [0-9.]+\): (\d+) / (\d+)')
  $tie = if ($nt.Success) { "$($nt.Groups[1].Value) / $($nt.Groups[2].Value)" } else { "-" }
  if ($m.Success) {
    [void]$rows.Add([pscustomobject]@{ option = $opt; against = $against; mean = $m.Groups[1].Value; max = $m.Groups[3].Value; top = "$($m.Groups[4].Value) / $($m.Groups[5].Value)"; tie = $tie })
  } else {
    [void]$rows.Add([pscustomobject]@{ option = $opt; against = $against; mean = "failed"; max = "-"; top = "-"; tie = "-" })
  }
  Stamp "$opt against $against done"
}
if ($fn) {
  Add-Row "default (sparse attention)" "sparse reference" $refSparse @{ ARC_QSA_TOPK = "64" }
  Add-Row "ARC_QSA=0 (full attention)" "full reference" $refFull @{ ARC_QSA = "0" }
  Add-Row "ARC_QSA=0 (full attention)" "sparse reference (how far from the default)" $refSparse @{ ARC_QSA = "0" }
  Add-Row "ARC_FN_PRECISE=0 (fast mode)" "sparse reference" $refSparse @{ ARC_FN_PRECISE = "0"; ARC_QSA_TOPK = "64" }
  Add-Row "ARC_FN_PRECISE=0, ARC_QSA=0 (fast mode)" "full reference" $refFull @{ ARC_FN_PRECISE = "0"; ARC_QSA = "0" }
} elseif ($qsa27) {
  Add-Row "default (full attention)" "full reference" $refFull @{}
  Add-Row "ARC_QSA=1 (sparse attention)" "sparse reference" $refSparse @{ ARC_QSA = "1"; ARC_QSA_TOPK = "64" }
  Add-Row "ARC_QSA=1 (sparse attention)" "full reference (how far from the default)" $refFull @{ ARC_QSA = "1"; ARC_QSA_TOPK = "64" }
} else {
  Add-Row "default" "reference" $refFull @{}
}
if ($h8) { Add-Row "ARC_H8_Q2=1 (more precise prompt reading)" "the default's reference" $(if ($fn) { $refSparse } else { $refFull }) $(if ($fn) { @{ ARC_H8_Q2 = "1"; ARC_QSA_TOPK = "64" } } else { @{ ARC_H8_Q2 = "1" } }) }
# --- Faster drafting: the same text as plain drafting (a code question, 128 tokens)
$draft = "not run (no MTP block)"
$q = Join-Path $work "question.txt"
[IO.File]::WriteAllText($q, "<|im_start|>user`nWrite a Python function that merges two sorted lists into one sorted list, with a docstring.<|im_end|>`n<|im_start|>assistant`n<think>`n`n</think>`n`n", (New-Object Text.UTF8Encoding $false))
$a = Invoke-Exe $runz @($Model, "--text-file", $q, "--n", "128", "--mtp", "--show") @{}
if ($a -match "MTP:" -and $a -notmatch "no MTP block") {
  $b = Invoke-Exe $runz @($Model, "--text-file", $q, "--n", "128", "--mtp", "--show") @{ ARC_MTP_VOCAB = "32768" }
  $ta = if ($a -match '(?s)----- text -----\r?\n(.*?)\r?\n----------------') { $Matches[1] } else { "" }
  $tb = if ($b -match '(?s)----- text -----\r?\n(.*?)\r?\n----------------') { $Matches[1] } else { "" }
  $draft = if ($ta -ne "" -and $ta -eq $tb) { "the same text as --mtp ($(Get-TextHash $ta))" } else { "TEXT DIFFERS: --mtp $(Get-TextHash $ta), ARC_MTP_VOCAB=32768 $(Get-TextHash $tb)" }
}
# --- the table
$gpu = ($probe -split "`n" | Select-String "GPU 0 \(" | Select-Object -First 1)
Write-Output ""
Write-Output ("kl_options: {0}, {1} tokens of USAGE.md, {2}" -f (Split-Path $Model -Leaf), $Tokens, $(if ($gpu) { $gpu.Line.Trim() } else { "GPU 0" }))
$rel = (Get-Content (Join-Path $root "bin\PREBUILT.txt") -Encoding UTF8 -ErrorAction SilentlyContinue | Select-Object -First 1) -join ""
if (-not $rel) { $rel = "source tree " + ((& git -C $root log -1 --format=%h 2>$null) -join "") }
Write-Output ("release: " + $rel.Trim([char]0xFEFF, ' '))
Write-Output ""
Write-Output "| option | against | KL mean | KL max | same top token | without near-ties |"
Write-Output "|---|---|---|---|---|---|"
foreach ($r in $rows) { Write-Output ("| {0} | {1} | {2} | {3} | {4} | {5} |" -f $r.option, $r.against, $r.mean, $r.max, $r.top, $r.tie) }
Write-Output "(without near-ties: positions whose reference top two are within 0.08 nats left out, ARC_KL_TIE; the KL counts every position)"
Write-Output ""
Write-Output "Faster drafting (ARC_MTP_VOCAB=32768): $draft"
if ($qsa) { Write-Output "(the sparse rows with the selection cut to 64 keys on both sides, ARC_QSA_TOPK / REF_TOPK, so that these tokens are sparse)" }
Write-Output "paste everything from 'kl_options:' into the GitHub issue"
