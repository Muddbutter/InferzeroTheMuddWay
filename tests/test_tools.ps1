# Exercises serverz's OpenAI API: streaming + reasoning + tool calls + multi-turn prefix reuse.
#   powershell -NoProfile -File tests\test_tools.ps1 [http://127.0.0.1:8090]
# (The PowerShell port of the former tests\test_tools.py, 2026-11-15: run_tests.ps1 needs no Python for it.)
# Prints "turn 1: finish=tool_calls ..." and "turn 2: ... cache_n=N ..." for run_tests.ps1 to check.
param([string]$Base = "http://127.0.0.1:8090")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http
$tools = @(@{
    type     = "function"
    function = @{
      name        = "read_file"
      description = "Read a text file from the workspace"
      parameters  = @{ type = "object"; required = @("path")
        properties = @{ path = @{ type = "string"; description = "File path" }; max_lines = @{ type = "integer" } } }
    }
  })
$http = New-Object System.Net.Http.HttpClient
$http.Timeout = [TimeSpan]::FromSeconds(1800)

function Get-P($o, $name) { if ($null -eq $o) { return $null }; $p = $o.PSObject.Properties[$name]; if ($p) { $p.Value } else { $null } }

function Stream-Chat($messages) {
  $body = @{ model = "x"; messages = $messages; tools = $tools; stream = $true; stream_options = @{ include_usage = $true }
    max_tokens = 1024; temperature = 0.6 } | ConvertTo-Json -Depth 20 -Compress
  $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, "$Base/v1/chat/completions")
  $req.Content = New-Object System.Net.Http.StringContent($body, [Text.Encoding]::UTF8, "application/json")
  $t0 = [Diagnostics.Stopwatch]::StartNew()
  $resp = $http.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
  $rd = New-Object IO.StreamReader($resp.Content.ReadAsStreamAsync().Result, [Text.Encoding]::UTF8)
  $s = @{ reasoning = ""; content = ""; calls = New-Object System.Collections.ArrayList; finish = $null; usage = $null; timings = $null }
  $toolEvents = 0
  while ($null -ne ($line = $rd.ReadLine())) {
    $line = $line.Trim()
    if (-not $line.StartsWith("data:")) { continue }
    $data = $line.Substring(5).Trim()
    if ($data -eq "[DONE]") { continue }
    $j = $data | ConvertFrom-Json
    if (Get-P $j "usage") { $s.usage = $j.usage }
    if (Get-P $j "timings") { $s.timings = $j.timings }
    foreach ($ch in @(Get-P $j "choices")) {
      if ($null -eq $ch) { continue }
      $d = Get-P $ch "delta"
      $s.reasoning += [string](Get-P $d "reasoning_content")
      $s.content += [string](Get-P $d "content")
      # tool calls arrive in pieces (OpenAI streaming): id and name first, then the arguments JSON
      foreach ($tc in @(Get-P $d "tool_calls")) {
        if ($null -eq $tc) { continue }
        $toolEvents++
        $k = Get-P $tc "index"; if ($null -eq $k) { $k = $s.calls.Count }
        while ($s.calls.Count -le $k) { [void]$s.calls.Add(@{ id = ""; type = "function"; function = @{ name = ""; arguments = "" } }) }
        if (Get-P $tc "id") { $s.calls[$k].id = $tc.id }
        $fn = Get-P $tc "function"
        if (Get-P $fn "name") { $s.calls[$k].function.name = $fn.name }
        $s.calls[$k].function.arguments += [string](Get-P $fn "arguments")
      }
      $f = Get-P $ch "finish_reason"; if ($f) { $s.finish = $f }
    }
  }
  $rd.Dispose(); $resp.Dispose()
  if ($s.calls.Count) { Write-Output "  tool call deltas: $toolEvents" }
  $s.seconds = $t0.Elapsed.TotalSeconds
  $s
}

function Short([string]$t, [int]$n) { if ($t.Length -gt $n) { $t.Substring(0, $n) } else { $t } }
$inv = [Globalization.CultureInfo]::InvariantCulture
$msgs = New-Object System.Collections.ArrayList
[void]$msgs.Add(@{ role = "user"; content = "Tolong baca file README.md lalu ringkas isinya." })
$out = Stream-Chat $msgs; $r1 = $out | Where-Object { $_ -is [hashtable] }; $out | Where-Object { $_ -is [string] }
Write-Output ("turn 1: finish=$($r1.finish) reasoning=$($r1.reasoning.Length) chars content='$(Short $r1.content 80)' " +
  "calls=$(ConvertTo-Json @($r1.calls) -Depth 10 -Compress) usage=$(ConvertTo-Json $r1.usage -Compress) " + $r1.seconds.ToString("F1", $inv) + "s")
if ($r1.finish -ne "tool_calls" -or -not $r1.calls.Count) { Write-Output "FAIL: expected a tool call"; exit 1 }
$args1 = $r1.calls[0].function.arguments | ConvertFrom-Json
Write-Output "  parsed arguments: $(ConvertTo-Json $args1 -Compress)"
[void]$msgs.Add(@{ role = "assistant"; content = $r1.content; reasoning_content = $r1.reasoning; tool_calls = @($r1.calls) })
[void]$msgs.Add(@{ role = "tool"; tool_call_id = $r1.calls[0].id
    content = "# Supreme Search`nMesin pencari dokumen internal berbasis FastAPI dan SQLite FTS5.`nFitur: indeks PDF, pencarian full-text, ranking BM25, API REST." })
$out = Stream-Chat $msgs; $r2 = $out | Where-Object { $_ -is [hashtable] }; $out | Where-Object { $_ -is [string] }
$cn = Get-P $r2.timings "cache_n"
Write-Output ("turn 2: finish=$($r2.finish) content='$(Short $r2.content 160)' usage=$(ConvertTo-Json $r2.usage -Compress) cache_n=$cn " +
  $r2.seconds.ToString("F1", $inv) + "s")
