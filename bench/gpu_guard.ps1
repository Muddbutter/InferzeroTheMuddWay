# The watchdog for GPU test runs (docs/gpu-failsafe.md 5; 2026-11-23, after a hung test kernel froze the development
# laptop's screen twice). Every GPU test on the laptop runs through it:
#   powershell -File bench\gpu_guard.ps1 -TimeoutS 60 -- .\bin\attn_bench.exe --edge
# 1. Refuses to start while the GPU hang marker (%USERPROFILE%\.inferzero\gpu_hang.txt) is from this start of the
#    computer, or while another GPU job runs (runz, serverz, attn_bench, xw_peak, gpu_bench, ref_forward, or a live
#    gpu_guard's lock file; a lock whose process is gone is removed).
# 2. Keeps the computer and the screen awake for the run (a request of this process, no settings changed; released at
#    the end and on Ctrl+C).
# 3. Runs the command with ARC_GPU_WAIT_S set (unless set already) a little below -TimeoutS, so the program's own limit
#    names the kernel first; its output goes to the console and to -Log (default: a file in %TEMP%).
# 4. Past -TimeoutS: kills the process tree; if it is still there 10 s later (stuck in the driver) it writes the marker
#    (kind=hang, killed=no), and no further runs start until a restart.
# 5. The command's exit code 3 (a GPU hang) or 4 (a lost GPU): the marker is there; prints it.
# Exit code: the command's; 124 when it was killed at the timeout; 3 when a refusal or a stuck process.
# Windows PowerShell 5.1 and PowerShell 7.
# (the options read from $args by hand, no param block: with -File, 5.1's binder took "--" for a parameter with an
#  empty name, and the command's own options (/c, -x) must reach it untouched)
$TimeoutS = 60
$WaitS = 0.0  # ARC_GPU_WAIT_S for the command (0: -TimeoutS - 10, at least 5)
$Log = ""
$Command = @()
for ($i = 0; $i -lt $args.Count; $i++) {
  $a = [string]$args[$i]
  if ($a -eq "--") { $Command = @($args | Select-Object -Skip ($i + 1) | ForEach-Object { [string]$_ }); break }
  elseif ($a -eq "-TimeoutS" -and $i + 1 -lt $args.Count) { $TimeoutS = [int]$args[++$i] }
  elseif ($a -eq "-WaitS" -and $i + 1 -lt $args.Count) { $WaitS = [double]$args[++$i] }
  elseif ($a -eq "-Log" -and $i + 1 -lt $args.Count) { $Log = [string]$args[++$i] }
  else { $Command = @($args | Select-Object -Skip $i | ForEach-Object { [string]$_ }); break }
}
$ErrorActionPreference = "Stop"
if (-not $Command -or $Command.Count -eq 0) {
  Write-Host "usage: gpu_guard.ps1 [-TimeoutS 60] [-WaitS S] [-Log file] -- program [arguments]"
  exit 1
}

Add-Type -Namespace GpuGuard -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
[DllImport("kernel32.dll")] public static extern ulong GetTickCount64();
'@

$dataDir = if ($env:ARC_DATA_DIR) { $env:ARC_DATA_DIR } else { Join-Path $env:USERPROFILE ".inferzero" }
New-Item -ItemType Directory -Force $dataDir | Out-Null
$marker = Join-Path $dataDir "gpu_hang.txt"
$lock = Join-Path $dataDir "gpu_guard.lock"
$utf8 = New-Object Text.UTF8Encoding $false

# this boot's id as the engine writes it (src/gpu_fail.h): the boot time in seconds since 1970, the same within 2 min
function Get-BootId { [long]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()) - [long]([GpuGuard.Native]::GetTickCount64() / 1000) }
function Read-Marker {
  if (-not (Test-Path -LiteralPath $marker)) { return $null }
  $kv = @{}
  foreach ($l in (Get-Content -Encoding UTF8 -LiteralPath $marker)) {
    $i = $l.IndexOf("=")
    if ($i -gt 0) { $kv[$l.Substring(0, $i)] = $l.Substring($i + 1) }
  }
  return $kv
}
function Test-MarkerThisBoot {
  $kv = Read-Marker
  if (-not $kv -or -not $kv.ContainsKey("boot")) { return $false }
  $b = 0L
  if (-not [long]::TryParse($kv["boot"], [ref]$b)) { return $false }
  # (older than 12 h: stale, as the engine counts it; Fast Startup keeps the time since boot over a "Shut down")
  $t = [DateTime]::MinValue
  $sty = [Globalization.DateTimeStyles]::AdjustToUniversal -bor [Globalization.DateTimeStyles]::AssumeUniversal
  if ($kv.ContainsKey("time") -and [DateTime]::TryParseExact($kv["time"], "yyyy-MM-ddTHH:mm:ssZ", [Globalization.CultureInfo]::InvariantCulture, $sty, [ref]$t)) {
    if (([DateTime]::UtcNow - $t).TotalHours -gt 12) { return $false }
  }
  return [math]::Abs((Get-BootId) - $b) -le 120
}
function Show-Marker {
  Write-Host "The GPU hang marker ($marker):" -ForegroundColor Red
  Get-Content -Encoding UTF8 -LiteralPath $marker | ForEach-Object { Write-Host "  $_" }
  Write-Host "Restart the computer (Restart, not Shut down) before using the GPU again (or runz --clear-hang)." -ForegroundColor Red
}
function Write-Marker([string]$prog, [string]$extra) {
  $s = "time=" + [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ", [Globalization.CultureInfo]::InvariantCulture) + "`nprogram=$prog`nkernel=(unknown: killed by gpu_guard)`nkind=hang`ngpu=`nboot=" + (Get-BootId) + "`n$extra`n"
  $tmp = "$marker.tmp"
  [IO.File]::WriteAllText($tmp, $s, $utf8)
  Move-Item -Force -LiteralPath $tmp -Destination $marker
}

# 1. the refusals
if (Test-MarkerThisBoot) { Write-Host "gpu_guard: a GPU hang since this computer started: no run." -ForegroundColor Red; Show-Marker; exit 3 }
$gpuProgs = "runz", "serverz", "attn_bench", "xw_peak", "gpu_bench", "ref_forward"
$busy = @(Get-Process -Name $gpuProgs -ErrorAction SilentlyContinue)
if ($busy.Count) {
  Write-Host ("gpu_guard: another GPU job runs: " + (($busy | ForEach-Object { "$($_.ProcessName) ($($_.Id))" }) -join ", ") + ": no run.") -ForegroundColor Red
  exit 3
}
if (Test-Path -LiteralPath $lock) {
  $lp = 0
  [void][int]::TryParse(([IO.File]::ReadAllText($lock)).Trim(), [ref]$lp)
  if ($lp -and (Get-Process -Id $lp -ErrorAction SilentlyContinue)) { Write-Host "gpu_guard: another gpu_guard runs (process $lp): no run." -ForegroundColor Red; exit 3 }
  Remove-Item -Force -LiteralPath $lock
}
[IO.File]::WriteAllText($lock, "$PID", $utf8)

if (-not $Log) { $Log = Join-Path ([IO.Path]::GetTempPath()) ("gpu_guard_" + [DateTime]::Now.ToString("yyyyMMdd_HHmmss") + ".log") }
$errLog = "$Log.err"
$exe = $Command[0]
$cmdArgs = @($Command | Select-Object -Skip 1)
$prog = [IO.Path]::GetFileNameWithoutExtension($exe)
$rc = 0
$hadWait = [bool]$env:ARC_GPU_WAIT_S
try {
  # 2. awake: ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
  [void][GpuGuard.Native]::SetThreadExecutionState([uint32]2147483651)
  # 3. the run
  if (-not $hadWait) {
    $w = if ($WaitS -gt 0) { $WaitS } else { [math]::Max(5, $TimeoutS - 10) }
    $env:ARC_GPU_WAIT_S = "$w"
  }
  Write-Host "gpu_guard: $($Command -join ' ')  (timeout $TimeoutS s, ARC_GPU_WAIT_S $env:ARC_GPU_WAIT_S, log $Log)"
  $spArgs = @{ FilePath = $exe; NoNewWindow = $true; PassThru = $true; RedirectStandardOutput = $Log; RedirectStandardError = $errLog }
  if ($cmdArgs.Count) {  # (one string, each argument with spaces or quotes quoted: 5.1 joins a list unquoted)
    $q = [string][char]34
    $spArgs.ArgumentList = ($cmdArgs | ForEach-Object { if ($_ -match '[\s"]' -or $_ -eq '') { $q + ($_ -replace '"', '\"') + $q } else { $_ } }) -join " "
  }
  $p = Start-Process @spArgs
  $h = $p.Handle  # (kept: ExitCode needs the handle)
  $pos = @{ $Log = 0L; $errLog = 0L }
  $show = {  # the new part of the two logs, to the console
    foreach ($f in @($Log, $errLog)) {
      if (-not (Test-Path -LiteralPath $f)) { continue }
      $fs = [IO.File]::Open($f, "Open", "Read", "ReadWrite")
      try {
        if ($fs.Length -gt $pos[$f]) {
          [void]$fs.Seek($pos[$f], "Begin")
          $b = New-Object byte[] ($fs.Length - $pos[$f])
          $n = $fs.Read($b, 0, $b.Length)
          $pos[$f] += $n
          [Console]::Out.Write($utf8.GetString($b, 0, $n))
        }
      } finally { $fs.Close() }
    }
  }
  $t0 = Get-Date
  $timedOut = $false
  while (-not $p.WaitForExit(500)) {
    & $show
    if (((Get-Date) - $t0).TotalSeconds -gt $TimeoutS) { $timedOut = $true; break }
  }
  & $show
  if ($timedOut) {
    # 4. the kill, then the check that it is gone
    Write-Host "gpu_guard: $prog still running after $TimeoutS s: killing it" -ForegroundColor Red
    # (taskkill's stderr under "Stop" in 5.1 would be a terminating error, skipping the check below)
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & taskkill.exe /T /F /PID $p.Id 2>&1 | Out-Null } catch {}
    $ErrorActionPreference = $eap
    if (-not $p.WaitForExit(10000)) {
      Write-Marker $prog "killed=no"
      Write-Host "gpu_guard: $prog did not end 10 s after the kill (stuck in the driver): marker written, no further GPU runs until a restart." -ForegroundColor Red
      Show-Marker
      $rc = 3
    } else {
      Write-Host "gpu_guard: killed." -ForegroundColor Red
      $rc = 124
    }
  } else {
    $p.WaitForExit()
    $rc = $p.ExitCode
    # 5. the program's own hang or lost device
    if (($rc -eq 3 -or $rc -eq 4) -and (Test-MarkerThisBoot)) { Show-Marker }
  }
} finally {
  [void][GpuGuard.Native]::SetThreadExecutionState([uint32]2147483648)
  if (-not $hadWait) { Remove-Item Env:ARC_GPU_WAIT_S -ErrorAction SilentlyContinue }
  Remove-Item -Force -LiteralPath $lock -ErrorAction SilentlyContinue
}
Write-Host "gpu_guard: exit code $rc"
exit $rc
