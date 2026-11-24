# The launchers' switches (QUEUE row 15 step 3, docs/launcher-design.md): for a model and a PC, which options of the
# catalog scripts\options.tsv apply, the one-line reason when one does not, each option's tip with the line under it,
# and its default. Dot-sourced by zero-launcher.ps1 and tests\test_launcher_switches.ps1; zero-launcher.sh has the
# same rules in bash (its sw_* functions), and both are checked against tests\launcher_switch_cases.tsv.
# Windows PowerShell 5.1 and 7. ASCII only (Windows PowerShell 5.1 reads scripts without a BOM as ANSI).
#
# The state a rule looks at ($st, a hashtable):
#   words     the model's words (serverz --list-models "opts": dense, moe, qsa, flashnext, mtp, h8) that are true
#   pc        this PC's words (scripts\pc-kinds.ps1); empty: not known, and no option is greyed for the PC
#   gpus      the GPUs the model is split over (1: one GPU)
#   drafting  $true when Drafting is on (the --mtp of needs:mtp)
#   choice    id -> "on" / "off" / a value: what was changed from the defaults (options.json)

function Read-OptionCatalog([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) { return @() }
  $ls = [IO.File]::ReadAllText($path, (New-Object Text.UTF8Encoding $false)) -split "\r?\n" | Where-Object { $_ -ne "" }
  $cc = $ls[0] -split "`t"
  return @(foreach ($l in $ls[1..($ls.Count - 1)]) {
      $v = $l -split "`t"; $d = [ordered]@{}
      for ($i = 0; $i -lt $cc.Count; $i++) { $d[$cc[$i]] = $(if ($i -lt $v.Count) { $v[$i].Trim() } else { "" }) }
      [pscustomobject]$d
    })
}

# the words in the lines people read
$script:SwModelWords = [ordered]@{ dense = "dense models"; moe = "MoE models"; moe_std = "standard MoE models"; qsa = "QSA models"; flashnext = "Flash Next"
  mtp = "models with MTP"; h8 = "dense Qwen3.5 / 3.8 models" }
$script:SwPcWords = [ordered]@{ "b-series" = "Arc B-series"; "alchemist" = "Arc A-series"; "igpu-140t" = "Arc 140T"
  "igpu-xe2" = "Lunar Lake / Panther Lake"; "igpu-xe-lpg" = "Meteor Lake" }

function Split-SwList([string]$s) { @("$s" -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
function Join-SwOr([string[]]$a) {
  if ($a.Count -le 1) { return ($a -join "") }
  return (($a[0..($a.Count - 2)] -join ", ") + " or " + $a[$a.Count - 1])
}
function Get-SwRow($cat, [string]$id) { @($cat | Where-Object { $_.id -eq $id })[0] }
# an option's title, or the words for mtp (Drafting) and gpus2
function Get-SwTitle($cat, [string]$id) {
  if ($id -eq "mtp") { return "Drafting" }
  if ($id -eq "gpus2") { return "2 or more GPUs" }
  $r = Get-SwRow $cat $id
  if ($r) { return $r.title } else { return $id }
}
# a value option (budget: <GiB>) is a field of its own, not a switch
function Test-SwValue($o) { "$($o.on)".StartsWith("<") }
# the default as on / off, or auto when the engine picks it by the PC or the model (the default column's words)
function Get-OptDefault($o) {
  $d = "$($o.default)".Trim()
  if ($d -match '^off\b') { return "off" }
  if ($d -match '^on\b' -and $d -notmatch '^on\s*:') { return "on" }
  return "auto"
}
# a choose-one option: its two variants are its on and off values (qsa: -sparse:1 / -full:0); the choices then
# are the model's default and the two variants, by their suffixes
function Get-SwChoices($o) {
  $vs = @(Split-SwList $o.variant)
  if ($vs.Count -ne 2 -or -not $o.on -or -not $o.off) { return @() }
  $out = @()
  foreach ($e in $vs) {
    $p = $e -split ':', 2
    $val = $(if ($p[1] -eq $o.on) { "on" } elseif ($p[1] -eq $o.off) { "off" } else { "" })
    if (-not $val) { return @() }
    $w = $p[0].TrimStart('-')
    $out += [pscustomobject]@{ value = $val; label = ($w.Substring(0, 1).ToUpper() + $w.Substring(1)) }
  }
  return $out
}
# whether id is on: Drafting for mtp, the GPUs for gpus2, else the choice or the default; $null when the engine picks
function Test-SwOn($cat, [string]$id, $st) {
  if ($id -eq "mtp") { return [bool]$st.drafting }
  if ($id -eq "gpus2") { return [int]$st.gpus -ge 2 }
  if ($st.choice -and $st.choice.ContainsKey($id)) { return $st.choice[$id] -ne "off" }
  $r = Get-SwRow $cat $id
  if (-not $r) { return $false }
  switch (Get-OptDefault $r) { "on" { return $true } "off" { return $false } default { return $null } }
}

# the one-line reason an option does not apply now, or "" when it does. The most specific one wins: what the model
# or the PC can never take first, then the GPUs, then a chosen option it cannot be combined with, then the option it
# needs that is off. An option the engine picks by itself (auto) counts as neither on nor off: it greys nothing.
function Get-SwReason($cat, $o, $st) {
  $words = @($st.words); $pc = @($st.pc)
  $ap = @(Split-SwList $o.applies)
  $mw = @($ap | Where-Object { $script:SwModelWords.Contains($_) })
  if ($mw.Count -and -not @($mw | Where-Object { $words -contains $_ }).Count) {
    return "for " + (Join-SwOr @($mw | ForEach-Object { $script:SwModelWords[$_] })) + " only"
  }
  $pw = @($ap | Where-Object { $script:SwPcWords.Contains($_) })
  if ($pc.Count -and $pw.Count -and -not @($pw | Where-Object { $pc -contains $_ }).Count) {
    return "needs " + (Join-SwOr @($pw | ForEach-Object { $script:SwPcWords[$_] }))
  }
  $rel = @(Split-SwList $o.relates)
  $multi = ($ap -contains "multi-gpu") -or ($rel -contains "needs:gpus2")
  if ($multi -and [int]$st.gpus -lt 2) {
    if ($pc -contains "multi-gpu") { return "choose a Multi-GPU split first" }
    return "needs 2 or more GPUs"
  }
  foreach ($e in $rel) {
    if ($e -match '^not:(.+)$' -and (Test-SwOn $cat $Matches[1] $st) -eq $true) { return "not with " + (Get-SwTitle $cat $Matches[1]) }
  }
  foreach ($e in $rel) {
    if ($e -match '^needs:(.+)$' -and $Matches[1] -ne "gpus2" -and (Test-SwOn $cat $Matches[1] $st) -eq $false) {
      return "turn " + (Get-SwTitle $cat $Matches[1]) + " on first"
    }
  }
  return ""
}

# the tip (the catalog's) and the line under it: what it needs, goes with or excludes, whether the output stays the
# same, and its cost
function Get-SwTipLine($cat, $o) {
  $p = @()
  $rel = @(Split-SwList $o.relates)
  $n = @($rel | Where-Object { $_ -match '^needs:' } | ForEach-Object { $i = $_ -replace '^needs:', ''; if ($i -eq "mtp") { "Drafting on (MTP)" } else { Get-SwTitle $cat $i } })
  $h = @($rel | Where-Object { $_ -match '^has:' } | ForEach-Object { Get-SwTitle $cat ($_ -replace '^has:', '') })
  $x = @($rel | Where-Object { $_ -match '^not:' } | ForEach-Object { Get-SwTitle $cat ($_ -replace '^not:', '') })
  if ($n.Count) { $p += "Needs: " + ($n -join ", ") + "." }
  if ($h.Count) { $p += "Goes with: " + ($h -join ", ") + "." }
  if ($x.Count) { $p += "Not with: " + ($x -join ", ") + "." }
  if (-not $p.Count) { $p += "Works on its own." }
  $p += $(if ($o.output -eq "same") { "Same output." } else { "Output checked against full precision." })
  # (the cost up to its first bracket or semicolon: the catalog's cost column goes on with measurements)
  $c = ("$($o.cost)" -split '\s*[(;]', 2)[0].Trim()
  if ($c -and $c -ne "none") { $p += "Costs: " + $c + "." }
  return ($p -join "   ")
}
function Get-SwTip($cat, $o) { "$($o.title)`r`n$($o.tip)`r`n$(Get-SwTipLine $cat $o)" }

# what Changed from the defaults lists: each changed option's title, with off when turned off
function Get-SwChanged($cat, $choice) {
  @(foreach ($r in $cat) {
      if (-not $choice.ContainsKey($r.id)) { continue }
      $c = $choice[$r.id]
      $ch = @(Get-SwChoices $r | Where-Object { $_.value -eq $c })
      if ($ch.Count) { "$(($r.title -split ':')[0]): $($ch[0].label)"; continue }  # (a choose-one option: "Attention: Full")
      if ($c -eq "off") { "$($r.title): off" } elseif ($c -eq "on") { $r.title } else { "$($r.title) ($c)" }
    })
}

# ---- the automatic preset, "Best measured for this PC" (the launcher spec B, 2026-11-22): scripts\best-setups.tsv's
# rows, each a setting measured faster than the engine's default on some PC and model (its evidence and confidence
# in the file). A row applies when this PC has all its pc words, the model is one of its models (shortcuts; * any),
# the OS is its os (windows / linux / any) and its needs hold: gpus2same (two or more discrete cards, all of one PCI device ID)
# and notfit (the model does not fit the card: weights in host memory). Its settings are zero-server's names:
# Opt=<options.tsv id>, Npu=on, Gpus=all, NoMtp. Adding a row needs no code change
function Read-BestSetups([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) { return @() }
  $lines = @(Get-Content -Encoding UTF8 -LiteralPath $path | Where-Object { $_ -and -not $_.StartsWith("#") })
  if ($lines.Count -lt 2) { return @() }
  $h = $lines[0] -split "`t"
  @(foreach ($l in $lines[1..($lines.Count - 1)]) {
      $f = $l -split "`t"
      $o = [ordered]@{}
      for ($i = 0; $i -lt $h.Count; $i++) { $o[$h[$i]] = $(if ($i -lt $f.Count) { $f[$i] } else { "" }) }
      [pscustomobject]$o
    })
}
# the rows for a model on this PC: $st = @{ pc = words; os = "windows" / "linux"; short = the model's shortcut
# (without a variant suffix); discrete = the discrete cards; notfit = the model's weights partly in host memory }
function Get-BestSetups($rows, $st) {
  @(foreach ($r in $rows) {
      $pcw = @(Split-SwList $r.pc | Where-Object { $_ -ne "-" })
      if (@($pcw | Where-Object { @($st.pc) -notcontains $_ }).Count) { continue }
      $ms = @(Split-SwList $r.models)
      if ($ms -notcontains "*" -and $ms -notcontains "$($st.short)") { continue }
      if ($r.os -and $r.os -ne "any" -and $r.os -ne $st.os) { continue }
      $ok = $true
      foreach ($n in @(Split-SwList $r.needs | Where-Object { $_ -ne "-" })) {
        switch ($n) {
          "gpus2same" { $ok = $ok -and [int]$st.discrete -ge 2 -and @($st.pc) -contains "multi-same" }  # (pc-kinds.ps1: every card one device ID)
          "notfit" { $ok = $ok -and [bool]$st.notfit }
          default { $ok = $false }  # (a need this launcher does not know: the row is left out)
        }
      }
      if ($ok) { $r }
    })
}
# a row's settings as (key, value) pairs: "Opt=moe_big_chunk" -> (Opt, moe_big_chunk), "NoMtp" -> (NoMtp, $true)
function Get-BestSettings($r) {
  @(foreach ($s in @(Split-SwList $r.settings)) {
      $k, $v = $s -split '=', 2
      [pscustomobject]@{ key = $k.Trim(); value = $(if ($null -eq $v) { $true } else { $v.Trim() }) }
    })
}
