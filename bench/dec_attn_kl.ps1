# dec_attn_kl.ps1: the long-context decode attention's switches against the default path, token by token (2026-11-23,
# the A770's 92K generation, #45: ARC_QWEN_DSPLIT=1 and ARC_ATTN_DEC_SMAX=512 were faster there, but they move the
# attention's sums, so they need a KL check before they can be defaults).
#
# An f32 CPU reference of 16K+ tokens of the 27B would take hours, so the default GPU path is the reference: its own
# log-probs at every decoded position (runz ARC_KL_SAVE), then each switch decoded the same way (runz --kl-decode) and
# its KL against them. The first -Ctx tokens go in as a prompt (ARC_KL_FROM), the next -Steps are decoded one by one:
# the decode attention at the long context, not the prompt's.
#
#   powershell -ExecutionPolicy Bypass -File bench\dec_attn_kl.ps1 -Model <27B.gguf>
#   ... -Ctx 32768 -Steps 256          (a longer context; -Ctx 65536 / 92000 where the card holds it)
#
# The text: the repository's docs, repeated to the length. Prints, per switch: the mean and max KL, and the share of
# positions with the same top token. The limit for these switches (AGENTS.md): KL <= 0.015 at every position (the
# max), the same top token >= 98 %. The reference is the default GPU path, not f32: the switch's own change, to which the
# default's KL against f32 adds (the laptop's 27B: 0.00058 mean, 0.0062 max at 480 positions).
# Works in Windows PowerShell 5.1 and PowerShell 7.
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [int]$Ctx = 16384,
  [int]$Steps = 256,
  [string]$Runz = ""
)
$ErrorActionPreference = "Continue"  # (runz writes its start lines to stderr: Windows PowerShell 5.1 would stop on them; failures are checked)
$root = Split-Path $PSScriptRoot -Parent
if (-not $Runz) { $Runz = Join-Path $root "bin\runz.exe" }
if (-not (Test-Path $Runz)) { throw "runz not found: $Runz" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
$busy = Get-Process serverz, runz -ErrorAction SilentlyContinue
if ($busy) { throw "serverz / runz is running (pid $($busy.Id -join ', ')): stop it first, two GPU users at once can fault the card" }
$work = Join-Path ([IO.Path]::GetTempPath()) ("dec_attn_kl_" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $work | Out-Null
$enc = New-Object Text.UTF8Encoding $false
try {
  # the text: the docs, repeated until it holds enough tokens (~4 characters a token; a margin of 2x)
  $docs = @(Get-ChildItem (Join-Path $root "docs") -Filter *.md | Sort-Object Name | ForEach-Object { [IO.File]::ReadAllText($_.FullName, $enc) })
  $sb = New-Object Text.StringBuilder
  while ($sb.Length -lt ($Ctx + $Steps) * 8) { foreach ($d in $docs) { [void]$sb.Append($d).Append("`n`n") } }
  $txt = Join-Path $work "text.txt"
  [IO.File]::WriteAllText($txt, $sb.ToString(), $enc)
  $ids = Join-Path $work "ids.txt"
  & $Runz $Model --text-file $txt --dump-ids $ids | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "runz --dump-ids failed" }
  $all = ([IO.File]::ReadAllText($ids, $enc).Trim()) -split ","
  $n = $Ctx + $Steps
  if ($all.Count -lt $n) { throw "the text gave $($all.Count) tokens, fewer than $n" }
  $cut = Join-Path $work "ids_cut.txt"
  [IO.File]::WriteAllText($cut, ($all[0..($n - 1)] -join ","), $enc)
  $env:ARC_KL_FROM = "$Ctx"
  $ctxArg = [string]([int][Math]::Ceiling(($n + 256) / 256.0) * 256)
  $ref = Join-Path $work "ref.bin"
  Write-Output ("dec_attn_kl: {0}, {1} prompt tokens, {2} decoded one by one (--ctx {3})" -f (Split-Path $Model -Leaf), $Ctx, $Steps, $ctxArg)
  $env:ARC_KL_SAVE = $ref
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & $Runz $Model --tokens-file $cut --ctx $ctxArg --kl-decode 2>&1 | Select-String "ARC_KL_SAVE|error" | ForEach-Object { Write-Output ("  default: " + $_.Line) }
  if ($LASTEXITCODE -ne 0) { throw "the default run failed" }
  Remove-Item Env:ARC_KL_SAVE
  Write-Output ("  (the reference run took {0:F0} s)" -f $sw.Elapsed.TotalSeconds)
  $cases = @(
    @{ name = "ARC_QWEN_DSPLIT=1"; env = @{ ARC_QWEN_DSPLIT = "1" } },
    @{ name = "ARC_ATTN_DEC_SMAX=512"; env = @{ ARC_ATTN_DEC_SMAX = "512" } },
    @{ name = "both"; env = @{ ARC_QWEN_DSPLIT = "1"; ARC_ATTN_DEC_SMAX = "512" } }
  )
  foreach ($c in $cases) {
    foreach ($k in $c.env.Keys) { Set-Item "Env:$k" $c.env[$k] }
    $out = & $Runz $Model --tokens-file $cut --ctx $ctxArg --kl-ref $ref --kl-decode 2>&1 | Select-String "KL decode path|error"
    foreach ($k in $c.env.Keys) { Remove-Item "Env:$k" }
    Write-Output ("  {0,-24} {1}" -f $c.name, (($out | ForEach-Object { $_.Line }) -join " "))
  }
  Remove-Item Env:ARC_KL_FROM
  Write-Output "dec_attn_kl: against the default GPU path; the limit: max KL <= 0.015 and the same top token at >= 98 % of the positions"
} finally {
  Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
