# The IQ3_S GEMV of generation in a repacked layout (a prototype, 2026-11-17, bench\gemv_iq3p.cl) against today's
# gemv (zero\kernels\decode.cl), on IQ3_S matrices of a real model: the outputs compared bit for bit, the weights'
# GB/s of each. The engine does not use the repack yet; this only measures what it would gain.
#   .\bench\gemv_iq3p.ps1 [-Model <gguf>] [-Tensors blk.2.ffn_up.weight,blk.1.ffn_down.weight] [-Grows 4,2]
# Default model: Qwen3.8 27B GSQ-RCO IQ3_S from the Hugging Face cache (its IQ3_S matrices: blk.2.ffn_up,
# blk.1.ffn_down, blk.4.ssm_out, ...; bin\gguf_dump.exe lists the types). Needs bin\gemv_bench.exe (.\build.ps1
# bench\gemv_bench.cpp).
param(
  [string]$Model = "",
  [string[]]$Tensors = @("blk.2.ffn_up.weight", "blk.1.ffn_down.weight"),
  [int[]]$Grows = @(4, 2)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$bench = Join-Path $root "bin\gemv_bench.exe"
if (-not (Test-Path $bench)) { throw "bin\gemv_bench.exe not found: .\build.ps1 bench\gemv_bench.cpp" }
if (-not $Model) {
  $Model = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF\snapshots\*\Qwen3.8-27B-GSQ-RCO-IQ3_S*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
  if (-not $Model) { throw "no -Model given and the 27B GSQ-RCO IQ3_S is not in the Hugging Face cache" }
}
$tmp = Join-Path ([IO.Path]::GetTempPath()) "inferzero-gemv-iq3p"
New-Item -ItemType Directory $tmp -Force | Out-Null
$kfile = Join-Path $tmp "decode_iq3p.cl"
# (the prototype appended to decode.cl: it uses its helpers, the codebook defines and HEAD_OUT)
[IO.File]::WriteAllText($kfile, [IO.File]::ReadAllText((Join-Path $root "zero\kernels\decode.cl")) + "`n" +
                                 [IO.File]::ReadAllText((Join-Path $PSScriptRoot "gemv_iq3p.cl")))
Push-Location $root
try {
  foreach ($t in $Tensors) {
    foreach ($g in $Grows) {
      "== $t, $g rows a sub-group"
      $a = & $bench $Model $t "-DGROWS=$g" gemv 16 20 2>&1 | ForEach-Object { "$_" }
      ($a | Where-Object { $_ -match "^blk|median" }) -replace "^median", "  today:    median"
      # (1: the planes per row; 2: inside each pair of superblocks, which keeps the pairs where they were)
      foreach ($mode in @(@{n = "per row "; r = "1"; d = ""}, @{n = "per pair"; r = "2"; d = " -DIQ3P_TILE"})) {
        $env:GB_REPACK = $mode.r; $env:GB_CMP = "gemv:-DGROWS=$g"; $env:GB_CMP_FILE = "zero\kernels\decode.cl"
        $b = & $bench $Model $t "-DGROWS=$g$($mode.d)" gemv_iq3p 16 20 $kfile 2>&1 | ForEach-Object { "$_" }
        Remove-Item Env:GB_REPACK, Env:GB_CMP, Env:GB_CMP_FILE -ErrorAction SilentlyContinue  # (before the next "today" run)
        ($b | Where-Object { $_ -match "bitwise|median|GB_REPACK|error" }) -replace "^median", "  $($mode.n): median" -replace "^bitwise", "  $($mode.n): bitwise"
      }
    }
  }
} finally {
  # (not left set in the caller's session, whatever stopped the loop: a later gemv_bench run would compare)
  foreach ($v in "GB_REPACK", "GB_CMP", "GB_CMP_FILE") { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  Pop-Location
}
