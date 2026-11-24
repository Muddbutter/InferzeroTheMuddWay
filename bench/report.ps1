# Summarizes bench\run_bench.ps1 results (JSON lines) into markdown tables comparing inferzero with llama.cpp.
#   powershell -File bench\report.ps1 bench\results\results.jsonl [more.jsonl ...]
# (The PowerShell port of the former bench\report.py, 2026-11-15: no Python anywhere a user runs.)
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Paths)
$ErrorActionPreference = "Stop"
$engines = @("inferzero", "vulkan", "sycl")
# (rows recorded before the rename, 2026-11-10, name the engine "arcinferzero": read as "inferzero")
$oldEngine = @{ "arcinferzero" = "inferzero"; "arcinferzero-npu" = "inferzero-npu" }
$inv = [Globalization.CultureInfo]::InvariantCulture
$dash = [string][char]0x2013; $times = [string][char]0xD7  # (as code points: Windows PowerShell reads a script without a BOM as ANSI)

$rows = New-Object System.Collections.ArrayList
foreach ($p in $Paths) {
  foreach ($line in [IO.File]::ReadAllLines((Resolve-Path $p))) {
    $line = $line.Trim([char]0xFEFF, ' ', "`t", "`r")
    if (-not $line) { continue }
    $r = $line | ConvertFrom-Json
    if ($r.engine -and $oldEngine.ContainsKey([string]$r.engine)) { $r.engine = $oldEngine[[string]$r.engine] }
    [void]$rows.Add($r)
  }
}

function Get-Val($r, $key) { $pr = $r.PSObject.Properties[$key]; if ($pr) { $pr.Value } else { $null } }

function Best($model, $engine, $test, $key, $prompt = $null, [bool]$higher = $true) {
  $vals = @($rows | Where-Object {
      $_.model -eq $model -and $_.engine -eq $engine -and $_.test -eq $test -and
      ($null -eq $prompt -or (Get-Val $_ "prompt") -eq $prompt) -and (Get-Val $_ $key) } |
    ForEach-Object { [double](Get-Val $_ $key) })
  if (-not $vals) { return $null }
  if ($higher) { ($vals | Measure-Object -Maximum).Maximum } else { ($vals | Measure-Object -Minimum).Minimum }
}

function Fmt($v, [int]$digits = 1) { if ($null -eq $v) { $dash } else { ([double]$v).ToString("F$digits", $inv) } }
function Ratio($a, $b) { if ($null -eq $a -or $null -eq $b -or $b -eq 0) { $dash } else { ($a / $b).ToString("F2", $inv) + $times } }

# the models in the order they first appear
$models = New-Object System.Collections.ArrayList
foreach ($r in $rows) { if (-not $models.Contains($r.model)) { [void]$models.Add($r.model) } }

$out = New-Object System.Collections.Generic.List[string]
$out.Add("## Throughput (tokens/s, higher is better)`n")
$out.Add("| Model | Test | inferzero | llama.cpp Vulkan | llama.cpp SYCL | vs Vulkan | vs SYCL |")
$out.Add("|---|---|---:|---:|---:|---:|---:|")
foreach ($m in $models) {
  foreach ($t in "pp512", "pp2048", "tg64") {
    $v = @{}; foreach ($e in $engines) { $v[$e] = Best $m $e $t "tps" }
    if (@($engines | Where-Object { $null -ne $v[$_] }).Count -eq 0) { continue }
    $out.Add("| $m | $t | $(Fmt $v.inferzero) | $(Fmt $v.vulkan) | $(Fmt $v.sycl) | " +
      "$(Ratio $v.inferzero $v.vulkan) | $(Ratio $v.inferzero $v.sycl) |")
  }
}
$out.Add("`n## Time to first token on real code prompts (seconds, lower is better)`n")
$out.Add("| Model | Prompt | Tokens | inferzero | llama.cpp Vulkan | llama.cpp SYCL | speed-up vs Vulkan | vs SYCL | " +
  "gen t/s at that context (arc / Vulkan / SYCL) |")
$out.Add("|---|---|---:|---:|---:|---:|---:|---:|---|")
foreach ($m in $models) {
  $prompts = @($rows | Where-Object { $_.model -eq $m -and $_.test -eq "ttft" } | ForEach-Object { Get-Val $_ "prompt" } |
      Sort-Object -Unique -CaseSensitive)
  foreach ($p in $prompts) {
    $v = @{}; $g = @{}
    foreach ($e in $engines) { $v[$e] = Best $m $e "ttft" "ttft_s" $p $false; $g[$e] = Best $m $e "ttft" "tg_tps" $p }
    $first = @($rows | Where-Object { $_.model -eq $m -and (Get-Val $_ "prompt") -eq $p })[0]
    $n = if ($first) { Get-Val $first "n_prompt" } else { "" }
    $out.Add("| $m | $p | $n | $(Fmt $v.inferzero) | $(Fmt $v.vulkan) | $(Fmt $v.sycl) | " +
      "$(Ratio $v.vulkan $v.inferzero) | $(Ratio $v.sycl $v.inferzero) | " +
      "$(Fmt $g.inferzero 2) / $(Fmt $g.vulkan 2) / $(Fmt $g.sycl 2) |")
  }
}
$out.Add("`n## Perplexity (ctx 512, lower is better)`n")
$out.Add("| Model | inferzero | llama.cpp Vulkan | llama.cpp SYCL |")
$out.Add("|---|---:|---:|---:|")
foreach ($m in $models) {
  $v = @{}; foreach ($e in $engines) { $v[$e] = Best $m $e "ppl" "ppl" $null $false }
  if (@($engines | Where-Object { $null -ne $v[$_] }).Count -eq 0) { continue }
  $out.Add("| $m | $(Fmt $v.inferzero 4) | $(Fmt $v.vulkan 4) | $(Fmt $v.sycl 4) |")
}
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$out -join "`n"
