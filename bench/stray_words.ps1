# stray_words: counts the words from other languages in Japanese answers (2026-11-21, for issue #4's stray words: a
# Japanese answer with a Korean, Hebrew or Swedish word in it), so runs compare by number, not by eye.
#   .\bench\stray_words.ps1 -Path answers\*.txt          one answer per file (UTF-8)
#   .\bench\stray_words.ps1 -Path a.txt,b.txt -Show       and each stray piece with the text around it
# Counted per answer and per 1,000 characters:
#   script  characters of scripts no Japanese text uses: Hangul, Hebrew, Cyrillic, Greek, Arabic, Thai, Devanagari;
#   latin   words of 4+ Latin letters inside a Japanese sentence (one with kana), such as "Swedish" in the middle of
#           a Japanese line (code, names and English answers are not Japanese sentences and are not counted).
#   simp    Simplified Chinese characters that Japanese text cannot hold (2026-11-22, issue #4): not in code page 932
#           (Shift-JIS, Windows' Japanese set) and in bench/simplified_han.txt (manoreken2's list of Simplified forms,
#           such as U+6743 / U+8FD9 / U+8D75). Characters Japanese shares with Chinese (U+4E0E, U+5199) are in CP932
#           and never counted.
#   outside CP932  Han characters outside CP932 that are not on that list: rare characters Japanese and Chinese share, such
#           as the surname U+90A2 or U+6C10 (in JIS X 0213, so Japanese fonts show them), or other Simplified forms the
#           list lacks. Shown apart, not called Simplified.
#   romanized  a romanization in parentheses right after Han text, such as a name and then (Shi Le) or (Zhao): legitimate,
#           counted apart and not as latin (2026-11-23, manoreken2 #4).
# When a file has "=== thinking" and "=== answer" lines (a saved server answer), every column counts the answer part,
# and script_think / latin_think / simp_think the thinking part apart: the thinking is often English with Japanese
# quotes and Chinese names, so counted with the answer it made ~10 Latin words per 1,000 characters of every answer
# (2026-11-23, #4: "Zhao", "Need", "emperor" from the thinking). A file without those lines is all answer.
# Not counted: Chinese written with characters Japanese also uses (the two share them; telling them apart needs a
# model, not a character table).
param(
  [Parameter(Mandatory = $true)][string[]]$Path,
  [switch]$Show
)
$ErrorActionPreference = "Stop"
# (CP932 with an empty replacement: a character it cannot hold encodes to 0 bytes. PowerShell 7 (.NET) needs the code
#  pages provider registered first; Windows PowerShell 5.1 (.NET Framework) has CP932 built in and no such type)
try { [Text.Encoding]::RegisterProvider([Text.CodePagesEncodingProvider]::Instance) } catch {}
$cp932 = [Text.Encoding]::GetEncoding(932, (New-Object Text.EncoderReplacementFallback("")), (New-Object Text.DecoderReplacementFallback("")))
$one = New-Object char[] 1
# (the Simplified forms outside CP932: bench/simplified_han.txt, hex code points, '#' lines are notes)
$simpSet = New-Object 'System.Collections.Generic.HashSet[int]'
$listFile = Join-Path $PSScriptRoot "simplified_han.txt"
if (-not (Test-Path $listFile)) { throw "stray_words: $listFile not found (it ships next to this script)" }
foreach ($l in [IO.File]::ReadAllLines($listFile)) {
  if ($l -match '^\s*#') { continue }
  foreach ($h in ($l -split '\s+')) { if ($h) { [void]$simpSet.Add([Convert]::ToInt32($h, 16)) } }
}
function Test-NotJapaneseHan([char]$ch) {
  $c = [int]$ch
  if (-not (($c -ge 0x4E00 -and $c -le 0x9FFF) -or ($c -ge 0x3400 -and $c -le 0x4DBF))) { return $false }
  $one[0] = $ch
  return $cp932.GetByteCount($one) -eq 0
}
# (the Han characters outside CP932 in s: simp the Simplified ones on the list, "outside CP932" the others)
function Measure-Simp([string]$s, [string]$where) {
  $r = @{ simp = 0; rare = 0 }
  for ($i = 0; $i -lt $s.Length; $i++) {
    if (Test-NotJapaneseHan $s[$i]) {
      $k = if ($simpSet.Contains([int]$s[$i])) { "simp" } else { "rare" }  # (shown as "outside CP932")
      $r[$k]++
      if ($Show) {
        $a = [Math]::Max(0, $i - 12)
        $script:pieces += "    [$k $where] " + $s[$i] + " in ..." + $s.Substring($a, [Math]::Min(30, $s.Length - $a)).Replace("`n", " ") + "..."
      }
    }
  }
  return $r
}
$ranges = @(
  @{ n = "Hangul"; r = @(@(0xAC00, 0xD7A3), @(0x1100, 0x11FF), @(0x3130, 0x318F)) },
  @{ n = "Hebrew"; r = @(, @(0x0590, 0x05FF)) },
  @{ n = "Cyrillic"; r = @(, @(0x0400, 0x04FF)) },
  @{ n = "Greek"; r = @(, @(0x0370, 0x03FF)) },
  @{ n = "Arabic"; r = @(, @(0x0600, 0x06FF)) },
  @{ n = "Thai"; r = @(, @(0x0E00, 0x0E7F)) },
  @{ n = "Devanagari"; r = @(, @(0x0900, 0x097F)) }
)
function Script-Of([int]$c) {
  foreach ($s in $ranges) { foreach ($r in $s.r) { if ($c -ge $r[0] -and $c -le $r[1]) { return $s.n } } }
  return $null
}
$kana = [regex]'[\u3040-\u30FF]'
$latin = [regex]'[A-Za-z\u00C0-\u024F]{4,}'
# (a romanization: Latin letters, spaces, hyphens and apostrophes in ASCII or full-width parentheses right after a Han
#  character or the repeat mark; escapes, as Windows PowerShell 5.1 reads this file as ANSI)
$roman = [regex]'(?<h>[\u3400-\u4DBF\u4E00-\u9FFF\u3005])\s*[\uFF08(]\s*[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F'' \-]*\s*[\uFF09)]'
# (characters, other scripts, Latin words in Japanese sentences and romanizations of one part of a file)
function Measure-Stray([string]$t, [string]$where) {
  $r = @{ chars = 0; script = 0; by = @{}; latin = 0; roman = 0 }
  for ($i = 0; $i -lt $t.Length; $i++) {
    if ([char]::IsWhiteSpace($t[$i])) { continue }
    $r.chars++
    $s = Script-Of ([int]$t[$i])
    if ($s) {
      $r.script++; $r.by[$s] = 1 + [int]$r.by[$s]
      if ($Show -and ($i -eq 0 -or -not (Script-Of ([int]$t[$i - 1])))) {
        $a = [Math]::Max(0, $i - 12); $script:pieces += "    [$s $where] ..." + $t.Substring($a, [Math]::Min(36, $t.Length - $a)).Replace("`n", " ") + "..."
      }
    }
  }
  foreach ($sent in ($t -split '[\u3002\uFF01\uFF1F!?\n]')) {
    if (-not $kana.IsMatch($sent)) { continue }
    $rm = $roman.Matches($sent)
    $r.roman += $rm.Count
    if ($rm.Count) { $sent = $roman.Replace($sent, '${h}') }  # (the romanizations out before the Latin words are counted)
    foreach ($m in $latin.Matches($sent)) {
      $r.latin++
      if ($Show) { $script:pieces += "    [latin $where] " + $m.Value + " in: " + $sent.Substring(0, [Math]::Min(60, $sent.Length)).Trim() }
    }
  }
  return $r
}
$files = @(foreach ($p in $Path) { Get-ChildItem -Path $p -File -ErrorAction Stop })
if (-not $files.Count) { throw "no files: $Path" }
$inv = [Globalization.CultureInfo]::InvariantCulture
$rows = @(); $tc = 0; $ts = 0; $tl = 0; $tro = 0; $tsimp = 0; $trare = 0; $nsimp = 0; $tsT = 0; $tlT = 0
foreach ($f in $files) {
  $t = [IO.File]::ReadAllText($f.FullName, [Text.Encoding]::UTF8)
  $pieces = @()
  # (the answer part, and the thinking part apart, when the file marks them; else the whole file is the answer)
  $mt = [regex]::Match($t, '(?im)^\s*===\s*thinking[^\n]*\n')
  $ma = [regex]::Match($t, '(?im)^\s*===\s*answer[^\n]*\n')
  $think = $null
  if ($ma.Success) {
    $ans = $t.Substring($ma.Index + $ma.Length)
    if ($mt.Success -and $mt.Index -lt $ma.Index) { $think = $t.Substring($mt.Index + $mt.Length, $ma.Index - $mt.Index - $mt.Length) }
  } else {
    $ans = $t
  }
  $a = Measure-Stray $ans "answer"
  $sr = Measure-Simp $ans "answer"
  $sT = ""; $lT = ""; $simpT = ""
  if ($null -ne $think) {
    $th = Measure-Stray $think "thinking"
    $sT = [string]$th.script; $lT = [string]$th.latin; $simpT = [string](Measure-Simp $think "thinking").simp
    $tsT += $th.script; $tlT += $th.latin
  }
  $per = if ($a.chars) { 1000.0 * ($a.script + $a.latin) / $a.chars } else { 0 }
  $rows += [pscustomobject]@{ answer = $f.Name; chars = $a.chars; script = $a.script; latin = $a.latin; romanized = $a.roman
    simp = $sr.simp; "outside CP932" = $sr.rare; script_think = $sT; latin_think = $lT; simp_think = $simpT
    per1000 = [string]::Format($inv, "{0:F2}", $per); which = (($a.by.Keys | Sort-Object | ForEach-Object { "$_ $($a.by[$_])" }) -join ", ") }
  $tc += $a.chars; $ts += $a.script; $tl += $a.latin; $tro += $a.roman; $tsimp += $sr.simp; $trare += $sr.rare; if ($sr.simp) { $nsimp++ }
  if ($Show -and $pieces.Count) { Write-Output ("{0}:" -f $f.Name); $pieces | Select-Object -First 20 | ForEach-Object { Write-Output $_ } }
}
$rows | Format-Table -AutoSize | Out-String -Width 240 | Write-Output
$clean = @($rows | Where-Object { $_.script -eq 0 -and $_.latin -eq 0 }).Count
Write-Output ([string]::Format($inv, "all {0} answers: {1} characters, {2} of other scripts, {3} Latin words in Japanese sentences: {4:F2} per 1,000 characters; {5} of {0} answers clean; {6} romanizations after Han text (not counted)",
  $rows.Count, $tc, $ts, $tl, $(if ($tc) { 1000.0 * ($ts + $tl) / $tc } else { 0 }), $clean, $tro))
if ($tsT -or $tlT -or @($rows | Where-Object { $_.latin_think -ne "" }).Count) {
  Write-Output ("in the thinking parts (not in the numbers above): {0} of other scripts, {1} Latin words in Japanese sentences" -f $tsT, $tlT)
}
Write-Output ([string]::Format($inv, "Simplified Chinese characters (not in CP932, on bench\simplified_han.txt), in the answers: {0} in {1} of {2} answers; outside CP932 (not on the list, so not counted as Simplified: CP932 alone is too narrow to tell): {3}",
  $tsimp, $nsimp, $rows.Count, $trare))
