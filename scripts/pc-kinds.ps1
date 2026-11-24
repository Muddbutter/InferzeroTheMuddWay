# Which kinds of PC this one is, from Windows' display adapters: the words of scripts\options.tsv's best_for column.
# Dot-sourced by zero-server.ps1 (zero-server options) and zero-launcher.ps1 (the options section), so the two agree.
# Windows PowerShell 5.1 or 7.
#   igpu-xe2      Lunar Lake / Panther Lake iGPUs (Arc 130V / 140V, B370 / B390: Xe2 / Xe3, 16-lane XMX)
#   igpu-ptl      Panther Lake's B370 / B390 too (besides igpu-xe2)
#   igpu-140t     the Arc 140T (Xe-LPG Plus, 8-lane XMX: the 140T preset)
#   igpu-xe-lpg   other Core Ultra iGPUs (Meteor Lake's Arc Graphics)
#   alchemist     discrete Arc A-series (A770, A750, A580, ...)
#   b-series      discrete Arc B-series (B580, B570, Pro B50 / B60 / B65 / B70)
#   multi-gpu     two or more discrete Arc cards
# (the iGPU names first: Panther Lake's B370 / B390 have a B and three digits too, and are no discrete card; the
#  Researcher's review, 2026-11-21)
function Get-PcKinds {
  $k = @(); $discrete = 0
  $gs = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "Intel" } | ForEach-Object { $_.Name })
  foreach ($g in $gs) {
    if ($g -match 'B3[0-9]0|140V|130V|Xe2') {
      $k += "igpu-xe2"
      if ($g -match 'B3[0-9]0') { $k += "igpu-ptl" }  # (Panther Lake's B370 / B390 alone: best-setups.tsv's rows for it, 2026-11-22)
    }
    elseif ($g -match '140T') { $k += "igpu-140t" }
    elseif ($g -match 'Arc\S* A\d{3}') { $k += "alchemist"; $discrete++ }
    elseif ($g -match 'Arc\S* (Pro )?B\d{2,3}\b') { $k += "b-series"; $discrete++ }
    elseif ($g -match 'Arc|Graphics') { $k += "igpu-xe-lpg" }
  }
  if ($discrete -gt 1) { $k += "multi-gpu" }
  # multi-same: two or more discrete Arc cards, all of one PCI device ID (tensor parallel takes only GPU 0's own: a
  # B60 + B70 or an A770 + A750 is refused without --split layer; the Researcher's review, 2026-11-22)
  if ($discrete -gt 1) {
    $ids = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match "Intel" -and $_.Name -notmatch 'B3[0-9]0|140V|130V|Xe2|140T' -and $_.Name -match 'Arc\S* (A\d{3}|(Pro )?B\d{2,3}\b)' } |
      ForEach-Object { if ("$($_.PNPDeviceID)" -match 'DEV_([0-9A-Fa-f]{4})') { $Matches[1].ToUpper() } else { "?" } })
    if ($ids.Count -gt 1 -and @($ids | Select-Object -Unique).Count -eq 1 -and $ids[0] -ne "?") { $k += "multi-same" }
  }
  return @($k | Select-Object -Unique)
}
# the discrete Arc cards (the launcher's Multi-GPU choice splits a model over them); 1 when there are none
function Get-DiscreteGpuCount {
  $n = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object {
      $_.Name -match "Intel" -and $_.Name -notmatch 'B3[0-9]0|140V|130V|Xe2|140T' -and $_.Name -match 'Arc\S* (A\d{3}|(Pro )?B\d{2,3}\b)' }).Count
  return [math]::Max(1, $n)
}
