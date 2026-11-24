# inferzero launcher: a window with every model (and variant) serverz.exe finds, what each supports, how its automatic
# context was counted (GPU memory by part) and whether the options as set fit, every server option, presets, and the
# running server (state, GPU memory, requests and speeds, its log). Start / Stop go through zero-server.ps1, so the server
# keeps running after the window closes. Minimized, it waits in the tray.
# Start it with zero-launcher.cmd (project folder) or:  pwsh -STA -File scripts\zero-launcher.ps1
#   -Theme light|dark   instead of the Windows setting;  -SelfTest <png>: open, load the list, save a picture, close
# (ASCII only: Windows PowerShell 5.1 reads scripts without a BOM as ANSI)
param([int]$Port = 8090, [ValidateSet("system", "light", "dark")][string]$Theme = "system", [switch]$HideConsole, [string]$SelfTest = "")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# ---- Win32: DPI awareness, Windows 11 window styling (DWM), dark scroll bars, the console window. Compiled by Add-Type
# with the C# compiler PowerShell carries (no SDK needed)
if (-not ("ArcWin" -as [type])) {
  Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ArcWin {
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("shell32.dll", CharSet = CharSet.Unicode)] public static extern int SetCurrentProcessExplicitAppUserModelID(string id);
  [DllImport("dwmapi.dll")] static extern int DwmSetWindowAttribute(IntPtr h, int attr, ref int val, int size);
  [DllImport("uxtheme.dll", CharSet = CharSet.Unicode)] public static extern int SetWindowTheme(IntPtr h, string app, string id);
  [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  public static int Dwm(IntPtr h, int attr, int v) { try { return DwmSetWindowAttribute(h, attr, ref v, 4); } catch { return -1; } }
}
"@
}
# (the window started by zero-launcher.cmd: its console is not needed; a terminal the user opened is left alone)
if ($HideConsole) { try { $cw = [ArcWin]::GetConsoleWindow(); if ($cw -ne [IntPtr]::Zero) { [void][ArcWin]::ShowWindow($cw, 0) } } catch {} }
# (its own taskbar button with the inferzero icon: without an app id of its own the window was grouped under
#  PowerShell, whose button showed no icon of the launcher; the user, 2026-11-22)
try { [void][ArcWin]::SetCurrentProcessExplicitAppUserModelID("inferzero.launcher") } catch {}
try { [void][ArcWin]::SetProcessDPIAware() } catch {}  # (sharp text at 125-200 % scaling; sizes below scale with S)
[System.Windows.Forms.Application]::EnableVisualStyles()
# an error inside a WinForms event goes to the message bar, not to .NET's "Unhandled exception" box (before any window)
try { [System.Windows.Forms.Application]::SetUnhandledExceptionMode("CatchException") } catch {}
[System.Windows.Forms.Application]::add_ThreadException({
    param($s, $e)
    try { Show-Bar "Launcher error: $($e.Exception.Message)" "error" -Details "$($e.Exception)" } catch {}
  })
$g0 = [System.Drawing.Graphics]::FromHwnd([IntPtr]::Zero); $script:dpiK = $g0.DpiX / 96.0; $g0.Dispose()
function S([double]$px) { [int][math]::Round($px * $script:dpiK) }

$root = Split-Path $PSScriptRoot -Parent
$arcServer = Join-Path $PSScriptRoot "zero-server.ps1"
. (Join-Path $PSScriptRoot "app-dir.ps1")
$stateDir = Get-AppDir  # (%USERPROFILE%\.inferzero; renamed from arcinfer once)
New-Item -ItemType Directory -Force $stateDir | Out-Null
# the engine: inferzero (bin\serverz.exe; MoE experts streamed from the SSD); the model list and its contexts come
# from it
function Exe { Join-Path $root "bin\serverz.exe" }
$exe = Exe
$stateFile = Join-Path $stateDir "server.json"
# (logs\server.log since 2026-11-23; zero-server -LogDir's folder, kept in log-dir.txt, or ARC_LOG_DIR: zero-server.ps1)
$serverLogDir = if ($env:ARC_LOG_DIR) { [IO.Path]::GetFullPath($env:ARC_LOG_DIR) }
                elseif (Test-Path -LiteralPath (Join-Path $stateDir "log-dir.txt")) { "$(Get-Content -Encoding UTF8 -LiteralPath (Join-Path $stateDir "log-dir.txt") -TotalCount 1)".Trim() }
                else { Join-Path $stateDir "logs" }
if (-not $serverLogDir) { $serverLogDir = Join-Path $stateDir "logs" }  # (an empty log-dir.txt)
$serverLog = Join-Path $serverLogDir "server.log"
$presetDir = Join-Path $stateDir "presets"
$updateFile = Join-Path $stateDir "update-check.json"
$shell = (Get-Process -Id $PID).Path  # the PowerShell running this window (pwsh or powershell)
function Base { "http://127.0.0.1:$([int]$portBox.Value)" }
function GiB($b) { if ($null -eq $b) { return "-" }; "{0:N2} GiB" -f ([double]$b / 1GB) }

# ---- colours: Windows 11's light / dark palettes, the accent colour from the Windows setting
$dark = $false
if ($Theme -eq "dark") { $dark = $true }
elseif ($Theme -eq "system") {
  try { $dark = (Get-ItemPropertyValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize" AppsUseLightTheme -ErrorAction Stop) -eq 0 } catch {}
}
function RGB([int]$r, [int]$g, [int]$b) { [System.Drawing.Color]::FromArgb($r, $g, $b) }
$accent = RGB 0 95 184
try {
  $ac = [uint32](Get-ItemPropertyValue "HKCU:\Software\Microsoft\Windows\DWM" AccentColor -ErrorAction Stop)  # (0xAABBGGRR)
  $accent = RGB ($ac -band 0xFF) (($ac -shr 8) -band 0xFF) (($ac -shr 16) -band 0xFF)
} catch {}
if ($dark) {
  # (a dark accent on a dark window: lighter, as Windows 11 does for text and buttons)
  $lift = { param($v) [int]($v + (255 - $v) * 0.35) }
  $accent = RGB (& $lift $accent.R) (& $lift $accent.G) (& $lift $accent.B)
  $Pal = @{ bg = (RGB 32 32 32); card = (RGB 43 43 43); input = (RGB 45 45 45); text = (RGB 255 255 255); sub = (RGB 170 170 170)
    line = (RGB 61 61 61); over = (RGB 74 64 34); run = (RGB 30 58 35); off = (RGB 38 38 38); ok = (RGB 108 203 95)
    warn = (RGB 252 225 0); bad = (RGB 255 153 164); sel = (RGB 60 72 90); onAccent = (RGB 0 0 0)
    barErr = (RGB 68 39 38); barWarn = (RGB 67 53 25); barInfo = (RGB 39 52 72) }
} else {
  $Pal = @{ bg = (RGB 243 243 243); card = (RGB 255 255 255); input = (RGB 255 255 255); text = (RGB 26 26 26); sub = (RGB 96 96 96)
    line = (RGB 229 229 229); over = (RGB 255 244 206); run = (RGB 223 246 221); off = (RGB 240 240 240); ok = (RGB 15 123 15)
    warn = (RGB 157 93 0); bad = (RGB 196 43 28); sel = (RGB 204 228 247); onAccent = (RGB 255 255 255)
    barErr = (RGB 253 231 233); barWarn = (RGB 255 244 206); barInfo = (RGB 230 240 250) }
}
$Pal.accent = $accent
# (the mark in site/icon.svg's colours, the website's and the web UI's icon: lit #0078d4 / dim #c9d3de, on dark
#  #59a7e3 / #2b3440; not the Windows accent colour, so it is one icon everywhere, the user asked 2026-11-17)
$markLit = if ($dark) { RGB 89 167 227 } else { RGB 0 120 212 }
$markDim = if ($dark) { RGB 43 52 64 } else { RGB 201 211 222 }
# (the tray's and the taskbar's: their own light / dark setting, which can differ from the apps')
$taskbarDark = $dark
try { $taskbarDark = (Get-ItemPropertyValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize" SystemUsesLightTheme -ErrorAction Stop) -eq 0 } catch {}

# ---- the inferzero mark: 4 x 4 rounded tiles, the lit ones drawing a Z (site/icon.svg, the web UI's /icon.svg).
# Small sizes snap the tiles to whole pixels, so a 16 / 32-pixel icon stays sharp.
function Draw-ArcMark($g, [single]$x, [single]$y, [single]$size, $lit, $dim) {
  if ($size -le 48) {
    $step = [math]::Floor($size / 4); $cell = $step - [math]::Max(1, [math]::Round($step * 3 / 16))
    $off = [math]::Floor(($size - (3 * $step + $cell)) / 2)
  } else { $step = $size / 4; $cell = $step * 13 / 16; $off = ($size - (3 * $step + $cell)) / 2 }
  $d = [single]($cell * 6 / 13)   # (the corner's diameter: a radius of 3 in 13)
  $zt = @(0, 1, 2, 3, 6, 9, 12, 13, 14, 15)   # (row * 4 + column: the top row, the diagonal, the bottom row)
  for ($i = 0; $i -lt 16; $i++) {
    $cx = [single]($x + $off + ($i % 4) * $step); $cy = [single]($y + $off + [math]::Floor($i / 4) * $step); $c = [single]$cell
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    if ($d -lt 2) { $p.AddRectangle((New-Object System.Drawing.RectangleF($cx, $cy, $c, $c))) }
    else {
      $p.AddArc($cx, $cy, $d, $d, 180, 90); $p.AddArc($cx + $c - $d, $cy, $d, $d, 270, 90)
      $p.AddArc($cx + $c - $d, $cy + $c - $d, $d, $d, 0, 90); $p.AddArc($cx, $cy + $c - $d, $d, $d, 90, 90); $p.CloseFigure()
    }
    $b = New-Object System.Drawing.SolidBrush($(if ($zt -contains $i) { $lit } else { $dim }))
    $g.FillPath($b, $p); $b.Dispose(); $p.Dispose()
  }
}

# ---- splash: shown as soon as the colours are known, while the fonts, the window and its parts are made (a few
# seconds with nothing on the screen before); closed when the window shows. No message loop runs yet, so each step
# paints it with Refresh(); a borderless window with Windows 11's rounded corners, in the theme's colours.
$script:splash = $null; $script:splashMsg = $null
# (the step and how far the loading is, in per cent: the blue line under the text is the progress bar. No message
#  loop runs yet, so it moves at each step, not in between)
$script:splashBar = $null
function Set-Splash([string]$t, [int]$pct = -1) {
  if (-not $script:splash) { return }
  try {
    $script:splashMsg.Text = $t
    if ($pct -ge 0 -and $script:splashBar) { $script:splashBar.Width = [int]($script:splashBar.Parent.Width * [math]::Min(100, $pct) / 100) }
    $script:splash.Refresh()
  } catch {}
}
try {
  $sp = New-Object System.Windows.Forms.Form
  $sp.FormBorderStyle = "None"; $sp.StartPosition = "CenterScreen"; $sp.ShowInTaskbar = $false; $sp.TopMost = $true
  $sp.AutoScaleMode = "None"; $sp.Size = New-Object System.Drawing.Size((S 440), (S 190))
  $sp.BackColor = $Pal.line; $sp.Padding = New-Object System.Windows.Forms.Padding(1)  # (a 1-pixel frame in the line colour)
  $sp.Text = "inferzero"
  $sp.Add_HandleCreated({ param($s) [void][ArcWin]::Dwm($s.Handle, 33, 2); [void][ArcWin]::Dwm($s.Handle, 20, [int]$dark) })
  $in = New-Object System.Windows.Forms.Panel; $in.Dock = "Fill"; $in.BackColor = $Pal.bg
  $sp.Controls.Add($in)
  # the logo: the inferzero mark in site/icon.svg's colours, drawn into a picture (no Paint event needed)
  $n = S 72
  $bmp = New-Object System.Drawing.Bitmap($n, $n)
  $g = [System.Drawing.Graphics]::FromImage($bmp); $g.SmoothingMode = "AntiAlias"
  Draw-ArcMark $g 0 0 $n $markLit $markDim
  $g.Dispose()
  $pic = New-Object System.Windows.Forms.PictureBox; $pic.Image = $bmp; $pic.Size = New-Object System.Drawing.Size($n, $n)
  $pic.Location = New-Object System.Drawing.Point((S 30), (S 40))
  $t1 = New-Object System.Windows.Forms.Label; $t1.AutoSize = $true; $t1.Text = "inferzero"; $t1.ForeColor = $Pal.text
  $t1.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 20); $t1.Location = New-Object System.Drawing.Point((S 120), (S 38))
  $t2 = New-Object System.Windows.Forms.Label; $t2.AutoSize = $true; $t2.Text = "Local AI on the Intel Arc GPU"; $t2.ForeColor = $Pal.sub
  $t2.Font = New-Object System.Drawing.Font("Segoe UI", 10); $t2.Location = New-Object System.Drawing.Point((S 123), (S 82))
  $script:splashMsg = New-Object System.Windows.Forms.Label; $script:splashMsg.AutoSize = $true; $script:splashMsg.ForeColor = $Pal.sub
  $script:splashMsg.Font = New-Object System.Drawing.Font("Segoe UI", 9); $script:splashMsg.Location = New-Object System.Drawing.Point((S 30), (S 146))
  $script:splashMsg.Text = "Starting..."
  # (a thin accent line under the text: the window's one touch of colour besides the logo)
  # (the progress bar: a track in the line colour, the part done in the accent colour; 2026-11-22 the user asked for
  #  the line to show the loading)
  $ln = New-Object System.Windows.Forms.Panel; $ln.BackColor = $Pal.line; $ln.Size = New-Object System.Drawing.Size((S 378), (S 3))
  $ln.Location = New-Object System.Drawing.Point((S 30), (S 132))
  $script:splashBar = New-Object System.Windows.Forms.Panel; $script:splashBar.BackColor = $accent
  $script:splashBar.Location = New-Object System.Drawing.Point(0, 0); $script:splashBar.Size = New-Object System.Drawing.Size([int]($ln.Width * 0.05), $ln.Height)
  $ln.Controls.Add($script:splashBar)
  $in.Controls.AddRange(@($pic, $t1, $t2, $ln, $script:splashMsg))
  $sp.Show()
  $script:splash = $sp
  $sp.Refresh()
  if ($SelfTest) {  # (-SelfTest x.png: the splash as x-splash.png too)
    $sb = New-Object System.Drawing.Bitmap($sp.Width, $sp.Height)
    $sp.DrawToBitmap($sb, (New-Object System.Drawing.Rectangle(0, 0, $sp.Width, $sp.Height)))
    $sb.Save(($SelfTest -replace '\.png$', '') + "-splash.png", [System.Drawing.Imaging.ImageFormat]::Png); $sb.Dispose()
  }
} catch { $script:splash = $null }
function Close-Splash { if ($script:splash) { try { $script:splash.Close(); $script:splash.Dispose() } catch {}; $script:splash = $null } }
Set-Splash "Loading fonts..." 10
# Segoe UI Variable (Windows 11), else Segoe UI; Cascadia Mono, else Consolas
$fams = @((New-Object System.Drawing.Text.InstalledFontCollection).Families | ForEach-Object { $_.Name })
$uiFont = if ($fams -contains "Segoe UI Variable Text") { "Segoe UI Variable Text" } else { "Segoe UI" }
$uiHead = if ($fams -contains "Segoe UI Variable Display") { "Segoe UI Variable Display" } else { "Segoe UI" }
$monoFont = if ($fams -contains "Cascadia Mono") { "Cascadia Mono" } else { "Consolas" }
function Font([double]$pt, [string]$style = "Regular", [string]$fam = $uiFont) { New-Object System.Drawing.Font($fam, $pt, [System.Drawing.FontStyle]$style) }
$fontBold = if ($fams -contains "Segoe UI Variable Text Semibold") { New-Object System.Drawing.Font("Segoe UI Variable Text Semibold", 9) } else { Font 9 "Bold" }

# background work as processes writing files, collected by a timer: the model list (serverz.exe --list-models; up to
# a minute per model the first time, while the automatic context is computed), zero-server.ps1 start / stop, and
# small jobs (doctor, the update check)
$script:listProc = $null
$script:listOut = Join-Path $env:TEMP "zero-launcher-models.json"
$script:actProc = $null
$script:actOut = Join-Path $env:TEMP "zero-launcher-action.log"
$script:actErr = Join-Path $env:TEMP "zero-launcher-action.err"
$script:models = @()
$script:running = $null
$script:props = $null
$script:jobs = New-Object System.Collections.ArrayList
$script:benching = $false  # (the GPU benchmark running: Start-GpuBench)

Set-Splash "Building the window..." 25
$form = New-Object System.Windows.Forms.Form
$form.Text = "inferzero"
$form.AutoScaleMode = "None"  # (sizes are scaled by S here)
# (clamped to the screen's work area, the taskbar left out: at 150 % 1240 x 820 is 1860 x 1230, taller than a 1080p or
#  1200p laptop's, and Start / Stop sat below its edge; launcher review 2026-11-19, docs/launcher-research.md section 7)
$wa = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$form.Size = New-Object System.Drawing.Size([Math]::Min((S 1240), $wa.Width), [Math]::Min((S 820), $wa.Height))
$form.MinimumSize = New-Object System.Drawing.Size([Math]::Min((S 960), $wa.Width), [Math]::Min((S 620), $wa.Height))
$form.StartPosition = "CenterScreen"
$form.Font = Font 9
$form.BackColor = $Pal.bg; $form.ForeColor = $Pal.text

function New-Label($t, $bold = $false) {
  $l = New-Object System.Windows.Forms.Label; $l.Text = $t; $l.AutoSize = $true
  $l.Margin = New-Object System.Windows.Forms.Padding((S 3), (S 7), (S 3), 0)
  if ($bold) { $l.Font = $fontBold }
  $l
}
function New-Button($t, $w = 110, [switch]$Primary) {
  $b = New-Object System.Windows.Forms.Button; $b.Text = $t; $b.Width = S $w; $b.Height = S 32
  $b.Margin = New-Object System.Windows.Forms.Padding((S 4), (S 3), (S 4), (S 3))
  $b.FlatStyle = "Flat"; $b.FlatAppearance.BorderSize = 1; $b.UseVisualStyleBackColor = $false
  if ($Primary) {
    $b.BackColor = $Pal.accent; $b.ForeColor = $Pal.onAccent; $b.FlatAppearance.BorderColor = $Pal.accent; $b.Font = $fontBold
  } else {
    $b.BackColor = $Pal.card; $b.ForeColor = $Pal.text; $b.FlatAppearance.BorderColor = $Pal.line
  }
  $b.Tag = $(if ($Primary) { "primary" } else { "" })
  $b
}
function New-Num($min, $max, $val, $dec = 0, $w = 80) {
  $n = New-Object System.Windows.Forms.NumericUpDown; $n.Minimum = $min; $n.Maximum = $max; $n.DecimalPlaces = $dec
  $n.Value = $val; $n.Width = S $w; $n.Margin = New-Object System.Windows.Forms.Padding((S 3)); $n.BorderStyle = "FixedSingle"
  $n.BackColor = $Pal.input; $n.ForeColor = $Pal.text; $n
}
function New-Text($w = 80) {
  $t = New-Object System.Windows.Forms.TextBox; $t.Width = S $w; $t.Margin = New-Object System.Windows.Forms.Padding((S 3))
  $t.BorderStyle = "FixedSingle"; $t.BackColor = $Pal.input; $t.ForeColor = $Pal.text; $t
}
function New-Combo($items, $w = 90) {
  $c = New-Object System.Windows.Forms.ComboBox; $c.DropDownStyle = "DropDownList"; $c.Width = S $w
  $c.Margin = New-Object System.Windows.Forms.Padding((S 3)); $c.FlatStyle = "Flat"; $c.BackColor = $Pal.input; $c.ForeColor = $Pal.text
  [void]$c.Items.AddRange($items); if ($c.Items.Count) { $c.SelectedIndex = 0 }; $c
}
# a combo that also takes typed text (a folder or file next to auto / off)
function New-EditCombo($items, $w = 90) { $c = New-Combo $items $w; $c.DropDownStyle = "DropDown"; $c }
function New-Check($t) { $c = New-Object System.Windows.Forms.CheckBox; $c.Text = $t; $c.AutoSize = $true; $c.Margin = New-Object System.Windows.Forms.Padding((S 3), (S 5), (S 3), 0); $c }
function New-Info {
  $t = New-Object System.Windows.Forms.TextBox; $t.Multiline = $true; $t.ReadOnly = $true; $t.Dock = "Fill"
  $t.ScrollBars = "Vertical"; $t.BackColor = $Pal.card; $t.ForeColor = $Pal.text; $t.BorderStyle = "None"
  $t.Font = Font 9 "Regular" $monoFont; $t
}
# a white (dark: grey) card with a title, the way Windows 11 settings pages group things
function New-Card($title, $ctl) {
  $p = New-Object System.Windows.Forms.Panel; $p.Dock = "Fill"; $p.BackColor = $Pal.card
  $p.Padding = New-Object System.Windows.Forms.Padding((S 12), (S 8), (S 8), (S 8))
  $p.Margin = New-Object System.Windows.Forms.Padding((S 6))
  $ctl.Dock = "Fill"
  [void]$p.Controls.Add($ctl)
  if ($title) {
    $h = New-Object System.Windows.Forms.Label; $h.Text = $title; $h.Dock = "Top"; $h.Height = S 26; $h.Font = $fontBold
    [void]$p.Controls.Add($h)
  }
  $p.Add_Paint({ param($s, $e) $pen = New-Object System.Drawing.Pen($Pal.line); $e.Graphics.DrawRectangle($pen, 0, 0, $s.Width - 1, $s.Height - 1); $pen.Dispose() })
  $p
}

# ---- status line and the header buttons
$status = New-Object System.Windows.Forms.Label
$status.Dock = "Fill"; $status.AutoEllipsis = $true; $status.TextAlign = "MiddleLeft"
$status.Font = Font 13 "Regular" $uiHead
$status.Text = "Checking the server..."
$webBtn = New-Button "Open web UI" 120
$moreBtn = New-Button ("More " + [char]0x25BE) 90
$header = New-Object System.Windows.Forms.TableLayoutPanel
$header.Dock = "Top"; $header.Height = S 56; $header.ColumnCount = 3; $header.RowCount = 1
$header.Padding = New-Object System.Windows.Forms.Padding((S 14), (S 8), (S 10), (S 4))
[void]$header.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", 100)))
[void]$header.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("AutoSize")))
[void]$header.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("AutoSize")))
$header.Controls.Add($status, 0, 0); $header.Controls.Add($webBtn, 1, 0); $header.Controls.Add($moreBtn, 2, 0)

# ---- the message bar: errors, warnings and news inside the window instead of boxes that stop everything (a box
# under a ticking timer repeated itself, GitHub #5)
$bar = New-Object System.Windows.Forms.TableLayoutPanel
$bar.Dock = "Top"; $bar.Height = S 40; $bar.ColumnCount = 4; $bar.RowCount = 1; $bar.Visible = $false
$bar.Padding = New-Object System.Windows.Forms.Padding((S 14), (S 4), (S 10), (S 4))
[void]$bar.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", 100)))
foreach ($i in 1..3) { [void]$bar.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("AutoSize"))) }
$barText = New-Object System.Windows.Forms.Label; $barText.Dock = "Fill"; $barText.TextAlign = "MiddleLeft"; $barText.AutoEllipsis = $true
$barAct = New-Button "" 120; $barCopy = New-Button "Copy details" 110; $barClose = New-Button ([string][char]0x00D7) 36
$bar.Controls.Add($barText, 0, 0); $bar.Controls.Add($barAct, 1, 0); $bar.Controls.Add($barCopy, 2, 0); $bar.Controls.Add($barClose, 3, 0)
$script:barDetails = ""; $script:barAction = $null
# Show-Bar "text" error|warn|info [-Details "..."] [-Action "Button text" { ... }]
function Show-Bar([string]$text, [string]$kind = "info", [string]$Details = "", [string]$ActionText = "", [scriptblock]$Action = $null) {
  $bar.BackColor = switch ($kind) { "error" { $Pal.barErr } "warn" { $Pal.barWarn } default { $Pal.barInfo } }
  $barText.ForeColor = $Pal.text
  $barText.Text = ($text -replace "\s*`r?`n\s*", "  ")
  $script:barDetails = $(if ($Details) { "$text`r`n`r`n$Details" } else { $text })
  $barCopy.Visible = [bool]$Details
  $script:barAction = $Action
  $barAct.Visible = [bool]$Action
  if ($Action) { $barAct.Text = $ActionText; $barAct.Width = [math]::Max((S 90), [System.Windows.Forms.TextRenderer]::MeasureText($ActionText, $barAct.Font).Width + (S 24)) }
  $bar.Visible = $true
  if ($SelfTest -and $kind -eq "error") { $script:selfErrors += $text }
}
$barClose.Add_Click({ $bar.Visible = $false })
$barCopy.Add_Click({ try { [System.Windows.Forms.Clipboard]::SetText($script:barDetails) } catch {}; $msg.Text = "Copied the message and its details." })
$barAct.Add_Click({ if ($script:barAction) { & $script:barAction } })
$script:selfErrors = @()

# ---- model table
$grid = New-Object System.Windows.Forms.DataGridView
$grid.Dock = "Fill"; $grid.ReadOnly = $true; $grid.AllowUserToAddRows = $false; $grid.AllowUserToDeleteRows = $false
$grid.AllowUserToResizeRows = $false; $grid.SelectionMode = "FullRowSelect"; $grid.MultiSelect = $false
$grid.RowHeadersVisible = $false; $grid.AutoSizeColumnsMode = "Fill"; $grid.BorderStyle = "None"
$grid.BackgroundColor = $Pal.card; $grid.GridColor = $Pal.line; $grid.CellBorderStyle = "SingleHorizontal"
$grid.EnableHeadersVisualStyles = $false; $grid.ColumnHeadersBorderStyle = "None"
$grid.ColumnHeadersHeightSizeMode = "DisableResizing"; $grid.ColumnHeadersHeight = S 32
$grid.ColumnHeadersDefaultCellStyle.BackColor = $Pal.card; $grid.ColumnHeadersDefaultCellStyle.ForeColor = $Pal.sub
$grid.ColumnHeadersDefaultCellStyle.SelectionBackColor = $Pal.card; $grid.ColumnHeadersDefaultCellStyle.Font = $fontBold
$grid.DefaultCellStyle.BackColor = $Pal.card; $grid.DefaultCellStyle.ForeColor = $Pal.text
$grid.DefaultCellStyle.SelectionBackColor = $Pal.sel; $grid.DefaultCellStyle.SelectionForeColor = $Pal.text
$grid.DefaultCellStyle.Padding = New-Object System.Windows.Forms.Padding((S 4), 0, (S 4), 0)
$grid.RowTemplate.Height = S 30
foreach ($c in @(
    @("model", "Model", 44), @("size", "File", 9), @("fit", "Fits", 9), @("ctx", "Auto context", 11),
    @("id", "Model id  [shortcut]", 27))) {
  $col = New-Object System.Windows.Forms.DataGridViewTextBoxColumn
  $col.Name = $c[0]; $col.HeaderText = $c[1]; $col.FillWeight = $c[2]
  [void]$grid.Columns.Add($col)
}

# ---- options (every server option zero-server.ps1 passes on). Each field shows what the server picks for the
# selected model; a field you change turns yellow and is passed on (the others are left to the server)
$portBox = New-Num 1024 65535 $Port 0 70
$ctxBox = New-Text 80
$chunkBox = New-Combo @("auto", "512", "1024", "2048", "4096", "8192", "16384")  # (auto: the server's choice, the launcher spec A2)
$draftBox = New-Combo @("1", "2", "3", "4", "5", "6", "7") 60
$thinkBox = New-Num -1 262144 8192 0 80  # (-1: no limit, serverz's own default)
$imgMinBox = New-Num 0 16384 0 0 70
$imgMaxBox = New-Num 0 16384 0 0 70
$budgetBox = New-Num 0 64 0 1 70
$sessBox = New-Num 0 512 0 0 70
$sessDirBox = New-Text 170
$prefixBox = New-Num 0 512 8 0 70
$noPrefixChk = New-Check "no prefix cache"
$noSwapChk = New-Check "keep the loaded model (requests do not switch models)"
$netChk = New-Check "reachable from other computers (0.0.0.0)"
$f32Chk = New-Check "f32 attention (tests)"
$noMtpChk = New-Check "no speculative decoding"
$visionBox = New-Combo @("on", "off") 60
$slotsBox = New-Num 1 8 1 0 60
$gpusBox = New-Num 1 8 1 0 60
$npuBox = New-Combo @("off", "on") 60
$noStreamChk = New-Check "every expert on the GPU (no SSD reads, smaller context)"
$preciseChk = New-Check "precise mode (Flash Next, on by default; off: the fast mode, prompts ~1.7x as fast, less exact; AutoRound: always)"
$keepThinkChk = New-Check "earlier turns keep their reasoning"
$whisperBox = New-EditCombo @("off", "auto") 170
$asrBox = New-EditCombo @("auto", "off") 170
$whDevBox = New-Combo @("gpu", "npu") 60
$whMaxBox = New-Num 1 600 180 0 70  # (serverz clamps to 1..600)
# (serverz options the launcher had no field for, the launcher spec A1, 2026-11-22; each passed only when changed)
$hostBox = New-Text 120
$xramBox = New-Num -1 1024 -1 0 70
$xcacheBox = New-Num 0 512 0 1 70
$drvBox = New-Num 0 8 0.5 1 60
$asrSlotsBox = New-Num 1 4 1 0 60
$mtpModelBox = New-Text 170
$mmprojBox = New-Text 170
$aliasBox = New-Text 120
$logFileBox = New-Text 170
$logVBox = New-Combo @("info", "warn", "error", "debug") 70
$noWebChk = New-Check "no web tools in the web UI"
$noTermChk = New-Check "no terminal in the web UI"

$opt = New-Object System.Windows.Forms.TableLayoutPanel
$opt.Dock = "Fill"; $opt.ColumnCount = 2; $opt.AutoScroll = $true
[void]$opt.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("AutoSize")))
[void]$opt.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", 100)))
$script:wrapRows = New-Object System.Collections.ArrayList  # (@(control, spans both columns))
function Fit-Rows {
  $cw = $opt.ClientSize.Width - [System.Windows.Forms.SystemInformation]::VerticalScrollBarWidth - (S 8)
  $c0 = 0; try { $c0 = @($opt.GetColumnWidths())[0] } catch {}
  foreach ($r in $script:wrapRows) {
    $w = [math]::Max((S 120), $(if ($r[1]) { $cw } else { $cw - $c0 }))
    $r[0].MaximumSize = New-Object System.Drawing.Size($w, 0)
    if (-not $r[0].AutoSize) { $r[0].Width = $w }  # (the tip box: a fixed height, the column's width)
  }
}
function Add-Opt($label, $ctl, $hint = "", [switch]$Advanced) {
  $p = New-Object System.Windows.Forms.FlowLayoutPanel; $p.AutoSize = $true; $p.WrapContents = $true
  [void]$script:wrapRows.Add(@($p, $false))
  $p.Margin = New-Object System.Windows.Forms.Padding(0)
  [void]$p.Controls.Add($ctl)
  if ($hint) { $h = New-Label $hint; $h.ForeColor = $Pal.sub; [void]$p.Controls.Add($h) }
  $l = New-Label $label
  $opt.Controls.Add($l); $opt.Controls.Add($p)
  if ($Advanced) { $script:advRows += , @($l, $p) }
  return $p
}
# a control over both columns (a heading, a switch's row, a line of text)
function Add-Span($ctl, [switch]$Advanced) {
  $opt.Controls.Add($ctl); $opt.SetColumnSpan($ctl, 2)
  [void]$script:wrapRows.Add(@($ctl, $true))
  if ($Advanced) { $script:advRows += , @($ctl, $ctl) }
}
function New-Head([string]$t) { $h = New-Label $t $true; $h.Margin = New-Object System.Windows.Forms.Padding((S 3), (S 12), (S 3), (S 2)); $h }
$script:advRows = @()
# (the command-line names in tooltips, for zero-server / serverz users; the labels say what each does)
$tips = New-Object System.Windows.Forms.ToolTip; $tips.AutoPopDelay = 30000

# ---- the selected model's switches (QUEUE row 15 step 3, docs/launcher-design.md): one row a model file, its
# choices here. The switches, their tips and the reason one is greyed all come from scripts\options.tsv through
# scripts\launcher-switches.ps1 (zero-launcher.sh has the same rules); the drafting and images choices from the
# model list's rows of the file (serverz --list-models). A switch left alone stays "default": the engine's.
Set-Splash "Reading the options..." 45
. (Join-Path $PSScriptRoot "launcher-switches.ps1")
$script:catalog = @(Read-OptionCatalog (Join-Path $PSScriptRoot "options.tsv"))
$optJson = Join-Path $env:USERPROFILE ".inferzero\options.json"
function Read-OptChoice {
  $c = @{}
  if (Test-Path -LiteralPath $optJson) { try { (Get-Content -Encoding UTF8 -LiteralPath $optJson -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $c[$_.Name] = "$($_.Value)" } } catch {} }
  return $c
}
# (which kinds of PC this is: scripts\pc-kinds.ps1, shared with zero-server options)
. (Join-Path $PSScriptRoot "pc-kinds.ps1")
$script:pcKinds = @(); try { $script:pcKinds = @(Get-PcKinds) } catch {}
$script:gpuCount = 1; try { $script:gpuCount = Get-DiscreteGpuCount } catch {}
# (the automatic preset's rows, scripts\best-setups.tsv: Apply-Auto)
$script:bestRows = @(); try { $script:bestRows = @(Read-BestSetups (Join-Path $PSScriptRoot "best-setups.tsv")) } catch {}
$script:autoOn = $true; $script:autoSet = @(); $script:autoWhy = @{}
$script:optChoice = Read-OptChoice  # (the switches' choices: options.json's, changed here; Start passes the difference)
$script:swFilling = $false

$modelHead = New-Label "Select a model" $true
$modelHead.Font = Font 11 "Regular" $uiHead; $modelHead.Margin = New-Object System.Windows.Forms.Padding((S 3), (S 4), (S 3), (S 6))
Add-Span $modelHead
# the tip of the switch under the mouse or with the keyboard focus, under the switches (the tooltip has it too)
# (a fixed three lines, cut with an ellipsis: a long tip pushed the rest of the panel down; the tooltip has it all)
$tipLbl = New-Label ""; $tipLbl.ForeColor = $Pal.sub; $tipLbl.AutoSize = $false; $tipLbl.AutoEllipsis = $true
$tipLbl.Height = [int]($tipLbl.Font.GetHeight() * 3.4 * $script:dpiK) + (S 4); $tipLbl.Width = S 420
function Show-Tip([string]$t) { $tipLbl.Text = $t }
# (the info mark: Windows' own Info glyph, a circled i (U+E946 in Segoe Fluent Icons on Windows 11, Segoe MDL2 Assets
#  on Windows 10), as Windows Settings draws it; "(i)" where neither font is there)
$infoFont = if ($fams -contains "Segoe Fluent Icons") { "Segoe Fluent Icons" } elseif ($fams -contains "Segoe MDL2 Assets") { "Segoe MDL2 Assets" } else { "" }
function New-InfoMark([string]$t) {
  $i = New-Label $(if ($infoFont) { [string][char]0xE946 } else { "(i)" })
  if ($infoFont) { $i.Font = New-Object System.Drawing.Font($infoFont, 10); $i.Margin = New-Object System.Windows.Forms.Padding((S 3), (S 8), (S 3), 0) }
  $i.ForeColor = $Pal.accent; $tips.SetToolTip($i, $t); $i.Tag = $t; $i.AccessibleName = "About this option"; $i.AccessibleDescription = $t
  $i.Cursor = [System.Windows.Forms.Cursors]::Help
  $i.Add_MouseEnter({ param($s) Show-Tip $s.Tag }); $i
}
function Hook-Tip($ctl, [string]$t) {
  $tips.SetToolTip($ctl, $t); $ctl.AccessibleDescription = $t
  $ctl.Add_MouseEnter({ param($s) Show-Tip $s.AccessibleDescription })
  $ctl.Add_Enter({ param($s) Show-Tip $s.AccessibleDescription })
}
# a choose-one group: toggle buttons in one panel (radio buttons: one on, the arrow keys move between them)
function New-Choice([string]$name) {
  $p = New-Object System.Windows.Forms.FlowLayoutPanel; $p.AutoSize = $true; $p.WrapContents = $false
  $p.Margin = New-Object System.Windows.Forms.Padding(0); $p.Tag = $name; $p.AccessibleName = $name; $p
}
function Paint-Choice($rb) {
  if ($rb.Checked) { $rb.BackColor = $Pal.accent; $rb.ForeColor = $Pal.onAccent } else { $rb.BackColor = $Pal.card; $rb.ForeColor = $(if ($rb.Enabled) { $Pal.text } else { $Pal.sub }) }
}
# $items: @{ value; label } each; $sel: the value on
function Set-Choices($panel, $items, [string]$sel, [bool]$enabled, [string]$tip) {
  $script:swFilling = $true
  try {
    $panel.SuspendLayout()
    foreach ($c in @($panel.Controls)) { $c.Dispose() }
    $panel.Controls.Clear()
    foreach ($it in $items) {
      $rb = New-Object System.Windows.Forms.RadioButton
      $rb.Appearance = "Button"; $rb.AutoSize = $true; $rb.FlatStyle = "Flat"; $rb.TextAlign = "MiddleCenter"
      $rb.Padding = New-Object System.Windows.Forms.Padding((S 6), 0, (S 6), 0)
      $rb.Margin = New-Object System.Windows.Forms.Padding(0, (S 3), 0, 0)
      $rb.FlatAppearance.BorderColor = $Pal.line; $rb.FlatAppearance.CheckedBackColor = $Pal.accent
      $rb.Text = $it.label; $rb.Tag = $it.value; $rb.Enabled = $enabled; $rb.Checked = ($it.value -eq $sel)
      $rb.AccessibleName = "$($panel.Tag): $($it.label)"
      if ($tip) { Hook-Tip $rb $tip }
      $rb.Add_CheckedChanged({ param($s) Paint-Choice $s; if ($s.Checked -and -not $script:swFilling) { Set-GroupChoice $s.Parent.Tag $s.Tag } })
      Paint-Choice $rb
      [void]$panel.Controls.Add($rb)
    }
    $panel.ResumeLayout()
  } finally { $script:swFilling = $false }
}
function New-Reason { $r = New-Label ""; $r.ForeColor = $Pal.sub; $r }
$draftPanel = New-Choice "Drafting"; $draftWhy = New-Reason
$draftRow = Add-Opt "Drafting" $draftPanel; [void]$draftRow.Controls.Add($draftWhy)
[void]$draftRow.Controls.Add((New-InfoMark "Drafting: small, fast guesses of the next tokens, each checked by the model itself, so the text stays the same and comes faster. MTP: the model's own draft block (a Gemma drafter on Gemma). DFlash2: a separate drafter, found next to the model. MTP + DFlash2: both, picked per step."))
$imgChk = New-Check "on"; $imgWhy = New-Reason
$imgRow = Add-Opt "Images" $imgChk; [void]$imgRow.Controls.Add($imgWhy)
[void]$imgRow.Controls.Add((New-InfoMark "Images (and audio on Gemma E4B): the model's image encoder is loaded, which takes GPU memory from the context. Off: a longer context."))
$attnPanel = New-Choice "Attention"; $attnWhy = New-Reason
$attnRow = Add-Opt "Attention" $attnPanel; [void]$attnRow.Controls.Add($attnWhy)
$mgPanel = New-Choice "Multi-GPU"; $mgWhy = New-Reason
$mgRow = Add-Opt "Multi-GPU" $mgPanel; [void]$mgRow.Controls.Add($mgWhy)
[void]$mgRow.Controls.Add((New-InfoMark "Multi-GPU: one model over several cards. Split by tensors: each GPU holds a share of every layer (the default). Split by layers: each GPU holds a range of the layers, for cards of different kinds or slow links."))
Add-Span (New-Head "Speed and accuracy")
# one row a switch: its check box (three states when the engine picks the default: the square is "automatic"),
# the reason when it does not apply, and (i)
$script:swRows = [ordered]@{}
foreach ($o in $script:catalog) {
  if ((Test-SwValue $o) -or @(Get-SwChoices $o).Count -or $o.id -eq "split_layer") { continue }  # (fields and groups of their own)
  $row = New-Object System.Windows.Forms.FlowLayoutPanel; $row.AutoSize = $true; $row.WrapContents = $true
  $row.Margin = New-Object System.Windows.Forms.Padding(0)
  $cb = New-Check $o.title; $cb.Tag = $o.id
  $auto = (Get-OptDefault $o) -eq "auto"
  $cb.ThreeState = $auto
  $tip = (Get-SwTip $script:catalog $o) + "`r`nDefault: $($o.default).   zero-server -Opt $($o.id) / -NoOpt $($o.id)"
  Hook-Tip $cb $tip
  $why = New-Reason
  [void]$row.Controls.Add($cb); [void]$row.Controls.Add($why); [void]$row.Controls.Add((New-InfoMark $tip))
  $cb.Add_CheckStateChanged({ param($s) if (-not $script:swFilling) { Set-Switch $s.Tag $s.CheckState } })
  Add-Span $row
  $script:swRows[$o.id] = @{ box = $cb; why = $why; row = $o; auto = $auto; panel = $row }
}
$naChk = New-Check ""; $naChk.ForeColor = $Pal.sub
$naChk.Add_CheckedChanged({ Update-Switches })
Add-Span $naChk
Add-Span $tipLbl
$opt.Add_Resize({ Fit-Rows })
[void](Add-Opt "Context (tokens)" $ctxBox "auto: the largest that fits")
# the fit bar: weights, context and buffers against this GPU's memory (amber above 90 %, red when it will not fit),
# with the parts or the reason under it
$fitBar = New-Object System.Windows.Forms.Panel
$fitBar.Size = New-Object System.Drawing.Size((S 360), (S 14)); $fitBar.Margin = New-Object System.Windows.Forms.Padding((S 3), (S 8), (S 3), 0)
$fitBar.AccessibleName = "GPU memory"; $script:fitParts = $null
$fitBar.Add_Paint({
    param($s, $e)
    $g = $e.Graphics; $w = $s.Width - 1; $h = $s.Height - 1
    $bg = New-Object System.Drawing.SolidBrush($Pal.off); $g.FillRectangle($bg, 0, 0, $w, $h); $bg.Dispose()
    $fp = $script:fitParts
    if ($fp -and $fp.free -gt 0) {
      $x = 0.0
      $col = if ($fp.auto) { $Pal.accent } elseif ($fp.need -gt $fp.free) { $Pal.bad } elseif ($fp.need -gt 0.9 * $fp.free) { $Pal.warn } else { $Pal.accent }  # (auto: full by design)
      foreach ($part in @(@($fp.weights, 255), @($fp.context, 170), @($fp.buffers, 110))) {
        $pw = [math]::Min($w - $x, $w * [double]$part[0] / [math]::Max($fp.free, $fp.need))
        if ($pw -le 0) { continue }
        $b = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb([int]$part[1], $col))
        $g.FillRectangle($b, [single]$x, 0, [single]$pw, $h); $b.Dispose(); $x += $pw
      }
    }
    $pen = New-Object System.Drawing.Pen($Pal.line); $g.DrawRectangle($pen, 0, 0, $w, $h); $pen.Dispose()
  })
$fitLbl = New-Label ""
$fitBox = New-Object System.Windows.Forms.FlowLayoutPanel; $fitBox.AutoSize = $true; $fitBox.FlowDirection = "TopDown"; $fitBox.WrapContents = $false
$fitBox.Margin = New-Object System.Windows.Forms.Padding(0)
[void]$fitBox.Controls.Add($fitBar); [void]$fitBox.Controls.Add($fitLbl)
[void](Add-Opt "GPU memory" $fitBox)
# what was changed from the defaults, with Reset
$chgLbl = New-Label "nothing"; $chgLbl.ForeColor = $Pal.sub; $chgLbl.MaximumSize = New-Object System.Drawing.Size((S 440), 0)
$resetBtn = New-Button "Reset" 70
$resetBtn.AccessibleName = "Reset the switches to the defaults"
$chgRow = Add-Opt "Changed from the defaults" $chgLbl; [void]$chgRow.Controls.Add($resetBtn)
$advChk = New-Check "Advanced (port, network, slots, presets, speech, ...)"
$advChk.Margin = New-Object System.Windows.Forms.Padding((S 3), (S 12), (S 3), (S 4))
Add-Span $advChk

# ---- Advanced: every other server option zero-server.ps1 passes on. Each field shows what the server picks for the
# selected model; a field you change turns yellow and is passed on (the others are left to the server)
# presets: named sets of options (and the model), the same files zero-server start -Preset <name> reads
$presetBox = New-Combo @() 170
$presetSave = New-Button "Save as..." 90
$presetDel = New-Button "Delete" 70
$presetPanel = New-Object System.Windows.Forms.FlowLayoutPanel
$presetPanel.AutoSize = $true; $presetPanel.WrapContents = $false; $presetPanel.Margin = New-Object System.Windows.Forms.Padding(0)
[void]$presetPanel.Controls.Add($presetBox); [void]$presetPanel.Controls.Add($presetSave); [void]$presetPanel.Controls.Add($presetDel)
[void](Add-Opt "Preset" $presetPanel -Advanced)
[void](Add-Opt "Port" $portBox -Advanced)
$netPanel = Add-Opt "Network" $netChk -Advanced
# the addresses other apps connect to (OpenAI / Anthropic / Ollama formats), with a Copy button each
$apiPanel = New-Object System.Windows.Forms.FlowLayoutPanel
$apiPanel.AutoSize = $true; $apiPanel.WrapContents = $false; $apiPanel.Margin = New-Object System.Windows.Forms.Padding(0)
$apiBox = New-Text 230; $apiBox.ReadOnly = $true
$apiFmt = New-Combo @("OpenAI (/v1)", "Anthropic (/v1/messages)", "Ollama") 170
$apiFmt.SelectedIndex = 0
$apiCopy = New-Button "Copy" 60
function Update-Api {
  $b = Base
  $apiBox.Text = switch ($apiFmt.SelectedIndex) { 1 { "$b/v1/messages" } 2 { $b } default { "$b/v1" } }
}
$apiFmt.Add_SelectedIndexChanged({ Update-Api })
$apiCopy.Add_Click({ Update-Api; [System.Windows.Forms.Clipboard]::SetText($apiBox.Text) })
[void]$apiPanel.Controls.Add($apiFmt); [void]$apiPanel.Controls.Add($apiBox); [void]$apiPanel.Controls.Add($apiCopy)
Update-Api
[void](Add-Opt "API address" $apiPanel "for other apps; API key: anything; model: the id in the list" -Advanced)
[void](Add-Opt "Conversations at once" $slotsBox "-Slots: each its own KV cache; the context is shared out (auto)" -Advanced)
[void](Add-Opt "GPUs (one model split)" $gpusBox "-Gpus: the cards a split model uses (Multi-GPU above sets it)" -Advanced)
[void](Add-Opt "Decoded weights kept (GB)" $budgetBox "0 = chosen with the context" -Advanced)
[void](Add-Opt "MTP drafts per step" $draftBox -Advanced)
[void](Add-Opt "Thinking budget (tokens)" $thinkBox "-1 = no limit" -Advanced)
[void](Add-Opt "Session cache (GB on disk)" $sessBox "whole conversations, kept when switching sessions; all models together (default 8); 0 = off" -Advanced)
[void](Add-Opt "NPU" $npuBox "part of each prompt chunk on the NPU (27B, Gemma 12B); kept on a restart" -Advanced)
[void](Add-Opt "Prompt chunk" $chunkBox "smaller = more context, slower prompts" -Advanced)
[void](Add-Opt "Image tokens min / max" $imgMinBox -Advanced)
[void](Add-Opt "" $imgMaxBox -Advanced)
[void](Add-Opt "Session cache folder" $sessDirBox "empty = %USERPROFILE%\.inferzero\zero\sessions" -Advanced)
[void](Add-Opt "Prefix cache (GB on disk)" $prefixBox "system prompts per project" -Advanced)
[void](Add-Opt "" $noPrefixChk -Advanced)
[void](Add-Opt "" $noSwapChk -Advanced)
[void](Add-Opt "" $noStreamChk -Advanced)
[void](Add-Opt "" $keepThinkChk -Advanced)
[void](Add-Opt "Listen address" $hostBox "any address of this PC (Network above: all of them)" -Advanced)
[void](Add-Opt "Experts in RAM (GB)" $xramBox "MoE: -1 = automatic, 0 = none" -Advanced)
[void](Add-Opt "Experts on the GPU (GB)" $xcacheBox "MoE: 0 = what fits" -Advanced)
[void](Add-Opt "GPU memory left to the driver (GB)" $drvBox -Advanced)
[void](Add-Opt "MTP head file" $mtpModelBox "a .gguf of its own (Flash Next's mtp-*.gguf)" -Advanced)
[void](Add-Opt "Image projector file" $mmprojBox "when it is not the one next to the model" -Advanced)
[void](Add-Opt "Model id in the API" $aliasBox "empty = the list's" -Advanced)
[void](Add-Opt "Log file" $logFileBox -Advanced)
[void](Add-Opt "Log detail" $logVBox -Advanced)
[void](Add-Opt "" $noWebChk -Advanced)
[void](Add-Opt "" $noTermChk -Advanced)
# ---- speech to text (-Whisper): the Transcripts / Meetings tabs, /v1/audio/transcriptions
Add-Span (New-Head "Speech to text") -Advanced
[void](Add-Opt "Whisper" $whisperBox "auto | off | a faster-whisper folder; ~8 GB off the context; kept on a restart" -Advanced)
[void](Add-Opt "Qwen3-ASR" $asrBox "auto | off (Whisper alone) | a .gguf file" -Advanced)
[void](Add-Opt "Whisper encoder on" $whDevBox -Advanced)
[void](Add-Opt "Longest clip (minutes)" $whMaxBox "uploads and chat clips; the library takes up to 6 h" -Advanced)
[void](Add-Opt "Speech requests at once" $asrSlotsBox -Advanced)
# (f32 attention: a test switch, out of the window since 2026-11-19 (launcher review); a preset that sets it still does.
#  No MTP, Images and Precise mode are the Drafting, Images and Precise mode switches above)
foreach ($tp in @(@($noSwapChk, "zero-server -NoSwap (serverz --no-swap)"), @($noStreamChk, "zero-server -NoStream (serverz --no-stream)"),
    @($keepThinkChk, "zero-server -PreserveThinking"), @($netChk, "zero-server -ListenHost 0.0.0.0 (serverz --host 0.0.0.0)"),
    @($whisperBox, "zero-server -Whisper"), @($asrBox, "zero-server -Asr"), @($draftBox, "zero-server -MtpDrafts (serverz --mtp-drafts)"),
    @($budgetBox, "zero-server -BudgetGb (serverz --budget): decoded W8 weights kept on the GPU"))) {
  $tips.SetToolTip($tp[0], $tp[1])
}
# (ticked: the addresses other computers use, and that anyone on the network can use it; the API box and the page keep
#  127.0.0.1 for this computer. As zero-launcher.sh says on Linux; launcher review 2026-11-19, finding 5)
$netWarn = New-Label ""
$netWarn.ForeColor = $Pal.warn; $netWarn.Visible = $false
[void]$netPanel.Controls.Add($netWarn)
function Update-NetWarn {
  if (-not $netChk.Checked) { $netWarn.Visible = $false; return }
  $ips = @()
  try {
    $ips = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop | Where-Object {
        $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
      ForEach-Object { "http://$($_.IPAddress):$([int]$portBox.Value)" })
  } catch {}
  $where = if ($ips.Count) { "other computers: " + ($ips -join ", ") } else { "other computers: this computer's network address" }
  $netWarn.Text = "$where. No password: anyone on this network can use it."
  $netWarn.Visible = $true
}
$netChk.Add_CheckedChanged({ Update-NetWarn })
$portBox.Add_ValueChanged({ Update-NetWarn })
function Show-Advanced([bool]$on) { $opt.SuspendLayout(); foreach ($r in $script:advRows) { $r[0].Visible = $on; $r[1].Visible = $on }; $opt.ResumeLayout() }
$advChk.Add_CheckedChanged({ Show-Advanced $advChk.Checked })
Show-Advanced $false

# overrides: field name -> value the user set (passed on by Start, saved by Set as default / a preset)
$script:over = @{}
# a preset's options the window has no field for (-DFlash, -Drafter): passed on by Start
$script:extraArgs = [ordered]@{}
$script:filling = $false
$fields = [ordered]@{
  port = $portBox; ctx = $ctxBox; chunk = $chunkBox; drafts = $draftBox; think = $thinkBox; img_min = $imgMinBox
  img_max = $imgMaxBox; budget = $budgetBox; prompt_cache = $sessBox; prompt_cache_dir = $sessDirBox
  prefix_cache = $prefixBox; no_prefix = $noPrefixChk; no_swap = $noSwapChk; network = $netChk; attn_f32 = $f32Chk
  no_mtp = $noMtpChk; vision = $visionBox; slots = $slotsBox; gpus = $gpusBox; npu = $npuBox; no_stream = $noStreamChk; precise = $preciseChk
  preserve_think = $keepThinkChk; whisper = $whisperBox; asr = $asrBox; whisper_device = $whDevBox; whisper_max = $whMaxBox
  host = $hostBox; expert_ram = $xramBox; expert_cache = $xcacheBox; driver_reserve = $drvBox; asr_slots = $asrSlotsBox
  mtp_model = $mtpModelBox; mmproj = $mmprojBox; alias = $aliasBox; log_file = $logFileBox; log_v = $logVBox
  no_web = $noWebChk; no_term = $noTermChk
}
function Get-Field($k) {
  $c = $fields[$k]
  switch ($c.GetType().Name) {
    "TextBox" { return $c.Text.Trim() }
    "NumericUpDown" { return [double]$c.Value }
    "ComboBox" { if ($c.DropDownStyle -eq "DropDown") { return $c.Text.Trim() }; return [string]$c.SelectedItem }
    default { return [bool]$c.Checked }
  }
}
function Put-Field($k, $v) {
  $c = $fields[$k]
  $script:filling = $true
  try {
    switch ($c.GetType().Name) {
      "TextBox" { $c.Text = "$v" }
      "NumericUpDown" { $c.Value = [decimal][math]::Max([double]$c.Minimum, [math]::Min([double]$c.Maximum, [double]$v)) }
      "ComboBox" {
        if ($c.DropDownStyle -eq "DropDown") { $c.Text = "$v" }
        else { $i = $c.Items.IndexOf([string]$v); if ($i -ge 0) { $c.SelectedIndex = $i } }
      }
      default { $c.Checked = [bool]$v }
    }
  } finally { $script:filling = $false }
}
function Show-Mark($k) {
  $c = $fields[$k]
  $on = $script:over.ContainsKey($k)
  if ($c -is [System.Windows.Forms.CheckBox]) { $c.ForeColor = if ($on) { $Pal.warn } else { $Pal.text } }
  else { $c.BackColor = if ($on) { $Pal.over } else { $Pal.input } }
}
$onEdit = {
  param($sender, $e)
  if ($script:filling) { return }
  $k = $sender.Tag
  $script:over[$k] = Get-Field $k
  Show-Mark $k
  Update-Fit
}
foreach ($k in $fields.Keys) {
  $c = $fields[$k]
  $c.Tag = $k
  switch ($c.GetType().Name) {
    "TextBox" { $c.Add_TextChanged($onEdit) }
    "NumericUpDown" { $c.Add_ValueChanged($onEdit) }
    "ComboBox" { if ($c.DropDownStyle -eq "DropDown") { $c.Add_TextChanged($onEdit) } else { $c.Add_SelectedIndexChanged($onEdit) } }
    default { $c.Add_CheckedChanged($onEdit) }
  }
}
# the ASR / Whisper fields only count with speech on
function Update-Speech { $asrBox.Enabled = $whDevBox.Enabled = $whMaxBox.Enabled = (Get-Field "whisper") -notin "", "off" }
$whisperBox.Add_TextChanged({ Update-Speech })
# ---- one model a file: the model list's rows of a file (with and without the MTP block, the images, DFlash2) are
# its Drafting and Images choices. The grid row's Tag is the list row these choices pick (what Start starts); the
# file's group (its rows, the choices) is in the row's first cell's Tag
function Get-DraftMode($e) {
  if (-not $e.mtp) { return "off" }
  if ("$($e.draft)" -eq "mtp" -or -not $e.dflash) { return "mtp" }
  if ("$($e.draft)" -eq "dflash") { return "dflash" }
  return "both"
}
function Get-DraftLabel([string]$d, $e) {
  switch ($d) { "off" { "Off" } "mtp" { $(if ("$($e.arch)" -eq "gemma4") { "Drafter" } else { "MTP" }) } "both" { "MTP + DFlash2" } default { "DFlash2" } }
}
# the row for the file's Drafting and Images; Images off with no row of its own: a row with images, and Start
# passes -Vision off (zero-server then takes the file's row without them, and passes the drafter itself)
function Resolve-Entry($g) {
  $e = @($g.entries | Where-Object { (Get-DraftMode $_) -eq $g.draft -and [bool]$_.vision -eq $g.img })[0]
  if (-not $e) { $e = @($g.entries | Where-Object { (Get-DraftMode $_) -eq $g.draft })[0] }
  if (-not $e) { $e = $g.entries[0] }
  return $e
}
function Sel-Group { if ($grid.SelectedRows.Count) { $grid.SelectedRows[0].Cells["model"].Tag } else { $null } }
function Sel-Entry { if ($grid.SelectedRows.Count) { $grid.SelectedRows[0].Tag } else { $null } }
# Flash Next's precise mode: on unless switched off (the fast mode; its default since 2026-11-22)
function Test-Precise { return (-not $script:over.ContainsKey("precise")) -or [bool]$script:over["precise"] }
# the auto context of the row, or of its variant when a switch with one is on (serverz --list-models "variants")
function Get-AutoCtx($m) {
  if (-not $m) { return 0 }
  foreach ($v in @($m.variants)) {
    if (-not $v) { continue }
    if ($v.option -eq "fn_precise") {  # (its variant -fast: the fast mode, when precise mode is switched off)
      if (-not (Test-Precise) -and "$($v.value)" -eq "0" -and $v.ctx) { return [int]$v.ctx }
      continue
    }
    $o = Get-SwRow $script:catalog $v.option
    if (-not $o) { continue }
    $c = "$($script:optChoice[$v.option])"
    $val = $(if ($c -eq "on") { $(if ($o.on) { $o.on } else { "on" }) } elseif ($c -eq "off") { $o.off } else { $c })
    if ($c -and "$val" -eq "$($v.value)" -and $v.ctx) { return [int]$v.ctx }
  }
  return [int]$m.ctx
}
# the switches' state for the rules (scripts\launcher-switches.ps1): the words of all the file's rows
function Get-SwState {
  $g = Sel-Group; $m = Sel-Entry
  $words = @()
  if ($g) { foreach ($e in $g.entries) { if ($e.opts) { $words += @($e.opts.PSObject.Properties | Where-Object { $_.Value -eq $true } | ForEach-Object { $_.Name }) } } }
  $ch = @{}; foreach ($k in $script:optChoice.Keys) { if ($k -ne "fn_precise") { $ch[$k] = $script:optChoice[$k] } }
  $ch["fn_precise"] = $(if (($m -and $m.precise_only) -or (Test-Precise)) { "on" } else { "off" })
  $gp = $(if ($script:over.ContainsKey("gpus")) { [int]$script:over["gpus"] } else { 1 })
  return @{ words = @($words | Select-Object -Unique); pc = $script:pcKinds; gpus = $gp; drafting = [bool]($g -and $g.draft -ne "off"); choice = $ch }
}
$draftTip = "Drafting: small, fast guesses of the next tokens, each checked by the model itself, so the text stays the same and comes faster."
# a choose-one group's buttons: built again when its choices change, else only their state (the keyboard focus stays)
$script:choiceKeys = @{}
function Update-Choices($panel, $items, [string]$sel, $enabledOf, [string]$tip) {
  $key = (@($items | ForEach-Object { "$($_.value)=$($_.label)" }) -join "|")
  if ($script:choiceKeys[$panel.Tag] -ne $key) { Set-Choices $panel $items $sel $true $tip; $script:choiceKeys[$panel.Tag] = $key }
  $script:swFilling = $true
  try {
    foreach ($rb in $panel.Controls) {
      $rb.Enabled = [bool](& $enabledOf $rb.Tag)
      $rb.Checked = ("$($rb.Tag)" -eq $sel)
      Paint-Choice $rb
    }
  } finally { $script:swFilling = $false }
}
# the selected model's head line, Drafting and Images, then the switches
function Refresh-Model {
  $g = Sel-Group
  if (-not $g) { $modelHead.Text = "Select a model"; Update-Switches; return }
  # (Images with this Drafting: a file whose list has no row with both, such as the 35B with MTP and images, gets
  #  Images off, greyed with the reason below; Start would have loaded none while the box said on. The Researcher's
  #  review of df7379c)
  $hasImg = @($g.entries | Where-Object { $_.vision }).Count -gt 0
  $imgHere = @($g.entries | Where-Object { $_.vision -and (Get-DraftMode $_) -eq $g.draft }).Count -gt 0
  if ($null -eq $g.imgWant) { $g.imgWant = $g.img }  # (what was asked; g.img: what this Drafting allows)
  $g.img = [bool]$g.imgWant -and $imgHere
  $grid.SelectedRows[0].Tag = Resolve-Entry $g
  Fit-Rows
  $e0 = $g.entries[0]
  $modelHead.Text = "{0}   {1:N1} GB file" -f $g.title, ($e0.bytes / 1e9)
  $modes = @(foreach ($d in "off", "mtp", "both", "dflash") { if (@($g.entries | Where-Object { (Get-DraftMode $_) -eq $d }).Count) { $d } })
  $items = @(foreach ($d in $modes) { [pscustomobject]@{ value = $d; label = (Get-DraftLabel $d $e0) } })
  $many = $modes.Count -gt 1
  Update-Choices $draftPanel $items $g.draft { param($v) $many } $draftTip
  $draftWhy.Text = $(if ($many) { "" } else { "this model has no MTP block or drafter" })
  $script:swFilling = $true
  try { $imgChk.Checked = $imgHere -and $g.img; $imgChk.Enabled = $imgHere } finally { $script:swFilling = $false }
  $imgWhy.Text = $(if (-not $hasImg) { "this model reads no images" } elseif (-not $imgHere) { "not with " + (Get-DraftLabel $g.draft $e0) + " drafting for this model" } else { "" })
  $imgChk.AccessibleName = "Images, " + $(if ($imgChk.Checked) { "on" } else { "off" }) + $(if ($imgWhy.Text) { ", " + $imgWhy.Text } else { "" })
  Update-Switches
}
$imgChk.Add_CheckedChanged({
    if ($script:swFilling) { return }
    $g = Sel-Group
    if ($g) { $g.img = $imgChk.Checked; $g.imgWant = $g.img; Refresh-Model; Show-Detail; Fill-Defaults }
  })
function Update-Switches {
  $st = Get-SwState; $m = Sel-Entry; $na = 0
  $script:swFilling = $true
  try {
    foreach ($id in $script:swRows.Keys) {
      $r = $script:swRows[$id]; $o = $r.row; $cb = $r.box
      $why = Get-SwReason $script:catalog $o $st
      if ($id -eq "fn_precise" -and $m -and $m.precise_only) { $why = "always on for this model" }
      if ($id -eq "fn_precise") { $cb.CheckState = $(if ($st.choice["fn_precise"] -eq "on") { "Checked" } else { "Unchecked" }) }
      elseif ($script:optChoice.ContainsKey($id)) { $cb.CheckState = $(if ($script:optChoice[$id] -eq "off") { "Unchecked" } else { "Checked" }) }
      elseif ($r.auto) { $cb.CheckState = "Indeterminate" }
      else { $cb.CheckState = $(if ((Get-OptDefault $o) -eq "on") { "Checked" } else { "Unchecked" }) }
      $cb.Enabled = -not $why
      $r.panel.Visible = (-not $why) -or $naChk.Checked
      if ($why) { $na++ }
      $r.why.Text = $(if ($why) { $why } elseif ($script:autoWhy[$id]) { "best measured: " + $script:autoWhy[$id] } elseif ($cb.CheckState -eq "Indeterminate") { "automatic" } else { "" })
      $state = switch ("$($cb.CheckState)") { "Checked" { "on" } "Unchecked" { "off" } default { "automatic" } }
      $cb.AccessibleName = "$($o.title), $state" + $(if ($why) { ", $why" } else { "" })
      $changed = $(if ($id -eq "fn_precise") { -not (Test-Precise) } else { $script:optChoice.ContainsKey($id) })
      $cb.ForeColor = $(if ($changed -and -not $why) { $Pal.warn } else { $Pal.text })
    }
    $naChk.Text = $(if ($naChk.Checked) { "Hide the $na that don't apply to this model or PC" } else { "Show the $na that don't apply to this model or PC" })
    $naChk.Visible = $na -gt 0
  } finally { $script:swFilling = $false }
  # Attention: the model's default or the two forms (the qsa option's variants)
  $qo = Get-SwRow $script:catalog "qsa"
  if ($qo) {
    $qwhy = Get-SwReason $script:catalog $qo $st
    $items = @([pscustomobject]@{ value = ""; label = "Model's default" }) + @(Get-SwChoices $qo)
    Update-Choices $attnPanel $items "$($script:optChoice['qsa'])" { param($v) -not $qwhy } ((Get-SwTip $script:catalog $qo) + "`r`nDefault: $($qo.default).")
    $attnWhy.Text = $qwhy
  }
  # Multi-GPU: one GPU, or the model split over the cards by tensors or by layers (split_layer)
  $lo = Get-SwRow $script:catalog "split_layer"
  $mgSel = $(if ($st.gpus -lt 2) { "one" } elseif ($script:optChoice["split_layer"] -eq "on") { "layer" } else { "tensor" })
  $st2 = Get-SwState; $st2.gpus = [math]::Max(2, $st.gpus)
  $lwhy = $(if ($lo) { Get-SwReason $script:catalog $lo $st2 } else { "not in the catalog" })
  $mgItems = @([pscustomobject]@{ value = "one"; label = "One GPU" }, [pscustomobject]@{ value = "tensor"; label = "Split by tensors" }, [pscustomobject]@{ value = "layer"; label = "Split by layers" })
  $multi = $script:gpuCount -ge 2
  Update-Choices $mgPanel $mgItems $mgSel { param($v) $multi -and ($v -ne "layer" -or -not $lwhy) } $(if ($lo) { (Get-SwTip $script:catalog $lo) } else { "" })
  $mgWhy.Text = $(if (-not $multi) { "needs 2 or more GPUs" } elseif ($lwhy) { "Split by layers: $lwhy" } else { "" })
  Update-Changed
}
# a switch clicked: on / off, or back to the default when that is what it shows now (three states: the square is
# the engine's own choice)
function Set-Switch([string]$id, $state) {
  if ($id -eq "fn_precise") {
    $script:over["precise"] = ("$state" -eq "Checked")  # (on too: -Precise, else a running server keeps its -Fast)
  } else {
    $r = $script:swRows[$id]
    $v = switch ("$state") { "Checked" { "on" } "Unchecked" { "off" } default { "" } }
    if (-not $r.auto -and $v -eq (Get-OptDefault $r.row)) { $v = "" }
    if ($v) { $script:optChoice[$id] = $v } else { $script:optChoice.Remove($id) }
  }
  Update-Switches; Update-Fit
  $script:swRows[$id].box.Focus() | Out-Null
}
# a choose-one group's button pressed
function Set-GroupChoice([string]$name, $value) {
  $g = Sel-Group
  switch ($name) {
    "Drafting" { if ($g) { $g.draft = $value; Refresh-Model; Show-Detail; Fill-Defaults } ; return }
    "Attention" { if ($value) { $script:optChoice["qsa"] = $value } else { $script:optChoice.Remove("qsa") } }
    "Multi-GPU" {
      if ($value -eq "one") { $script:over["gpus"] = [double]1 } else { $script:over["gpus"] = [double][math]::Max(2, $script:gpuCount) }
      if ($value -eq "layer") { $script:optChoice["split_layer"] = "on" } else { $script:optChoice.Remove("split_layer") }
      Put-Field "gpus" $(if ($script:over.ContainsKey("gpus")) { $script:over["gpus"] } else { 1 }); Show-Mark "gpus"
    }
  }
  Update-Switches; Update-Fit
}
# ---- "Automatic (best measured)", the default preset (the launcher spec B, 2026-11-22): best-setups.tsv's rows for
# this PC and the selected model, on every setting not changed by hand; what it set is undone when the model changes,
# on Reset (and then set again) and when another preset is chosen. Each setting it made shows its reason
function Undo-Auto {
  $g = Sel-Group
  foreach ($a in $script:autoSet) {
    switch ($a.kind) {
      "opt" { if ("$($script:optChoice[$a.key])" -eq "$($a.value)") { $script:optChoice.Remove($a.key) } }
      "field" { if ($script:over.ContainsKey($a.key) -and "$($script:over[$a.key])" -eq "$($a.value)") { $script:over.Remove($a.key) } }
      "draft" { if ($a.group -and $a.group.draft -eq "off") { $a.group.draft = $a.group.draft0 } }
    }
  }
  $script:autoSet = @(); $script:autoWhy = @{}
}
function Apply-Auto {
  Undo-Auto
  if (-not $script:autoOn) { return }
  $g = Sel-Group; $m = Sel-Entry
  if (-not $g -or -not $m) { return }
  $short = "$($g.entries[0].short)" -replace '-(mtp|vision|std|df|nodf)$', ''
  $st = @{ pc = $script:pcKinds; os = "windows"; short = $short; discrete = $script:gpuCount
    notfit = [bool]($m.memory -and [double]$m.memory.host_weights -gt 0) }
  foreach ($r in @(Get-BestSetups $script:bestRows $st)) {
    foreach ($s in @(Get-BestSettings $r)) {
      switch ($s.key) {
        "Opt" {
          if ($script:optChoice.ContainsKey($s.value) -or -not (Get-SwRow $script:catalog $s.value)) { continue }
          $script:optChoice[$s.value] = "on"; $script:autoSet += @{ kind = "opt"; key = $s.value; value = "on" }
          $script:autoWhy[$s.value] = $r.why
        }
        "Npu" {
          if ($script:over.ContainsKey("npu")) { continue }
          $script:over["npu"] = "$($s.value)"; $script:autoSet += @{ kind = "field"; key = "npu"; value = "$($s.value)" }
          $script:autoWhy["npu"] = $r.why
        }
        "Gpus" {
          if ($script:over.ContainsKey("gpus") -or $script:gpuCount -lt 2) { continue }
          $n = $(if ($s.value -eq "all") { [double]$script:gpuCount } else { [double]$s.value })
          $script:over["gpus"] = $n; $script:autoSet += @{ kind = "field"; key = "gpus"; value = $n }
          $script:autoWhy["gpus"] = $r.why
        }
        "NoMtp" {
          if ($g.draft -ne $g.draft0 -or $g.draft -eq "off") { continue }
          $g.draft = "off"; $script:autoSet += @{ kind = "draft"; group = $g }
          $script:autoWhy["draft"] = $r.why
        }
      }
    }
  }
}
# what was changed from the defaults: the file's Drafting and Images, Precise mode, the GPUs and the switches
function Update-Changed {
  $g = Sel-Group; $x = @()
  if ($g) {
    if ($g.draft -ne $g.draft0) { $x += "Drafting: " + (Get-DraftLabel $g.draft $g.entries[0]) }
    if ($g.img -ne $g.img0) { $x += "Images: " + $(if ($g.img) { "on" } else { "off" }) }
  }
  if (-not (Test-Precise)) { $x += "Fast mode" }
  if ($script:over.ContainsKey("gpus") -and [int]$script:over["gpus"] -gt 1) { $x += "$([int]$script:over['gpus']) GPUs" }
  $x += @(Get-SwChanged @($script:catalog | Where-Object { $_.id -ne "fn_precise" -and -not (Test-SwValue $_) }) $script:optChoice)
  if ($script:autoWhy.Count) { $x += "(Automatic, best measured: " + (@($script:autoWhy.Values | Select-Object -Unique) -join "; ") + ")" }
  $chgLbl.Text = $(if ($x.Count) { $x -join "; " } else { "nothing" })
  $resetBtn.Enabled = $x.Count -gt 0
}
# Reset: the switches, Drafting, Images, Precise mode and the GPUs back to the defaults (Start then passes
# -DefaultOpt for the switches options.json had). Value options (a budget) stay: they have fields of their own
function Reset-Switches {
  foreach ($o in $script:catalog) { if (-not (Test-SwValue $o)) { $script:optChoice.Remove($o.id) } }
  $script:over["precise"] = $true; $script:over["gpus"] = [double]1  # (precise and one GPU: also for a running server)
  $g = Sel-Group
  if ($g) { $g.draft = $g.draft0; $g.img = $g.img0; $g.imgWant = $g.img0 }
  # (then the automatic preset again, when it is the one chosen: its GPUs first, else one GPU as above)
  Undo-Auto
  if ($script:autoOn) {
    $script:over.Remove("gpus")
    Apply-Auto
    if (-not $script:over.ContainsKey("gpus")) { $script:over["gpus"] = [double]1 }
  }
  Refresh-Model; Show-Detail; Fill-Defaults
}
$resetBtn.Add_Click({ Reset-Switches; $msg.Text = "The switches are back to the defaults (Start keeps it that way)." })

# the selected model's best values (what the server picks itself), into every field not overridden
function Fill-Defaults {
  $m = Sel-Entry
  $d = $null
  if ($m) { $d = $m.defaults }
  $actx = Get-AutoCtx $m
  $v = [ordered]@{
    port = $Port; ctx = $(if ($m -and $actx) { "$actx" } else { "auto" })
    chunk = "auto"; drafts = $(if ($d -and $d.mtp_drafts) { $d.mtp_drafts } else { 2 })
    think = 8192; img_min = $(if ($d) { $d.image_min } else { 0 }); img_max = $(if ($d) { $d.image_max } else { 0 })
    budget = 0; prompt_cache = 8; prompt_cache_dir = ""; prefix_cache = 8; no_prefix = $false; no_swap = $false
    network = $false; attn_f32 = $false; no_mtp = $false; vision = "on"; slots = 1; gpus = 1; npu = "off"; no_stream = $false; precise = $true
    preserve_think = $false; whisper = "off"; asr = "auto"; whisper_device = "gpu"; whisper_max = 180
    host = ""; expert_ram = -1; expert_cache = 0; driver_reserve = 0.5; asr_slots = 1; mtp_model = ""; mmproj = ""; alias = ""
    log_file = ""; log_v = "info"; no_web = $false; no_term = $false
  }
  foreach ($k in $fields.Keys) {
    Put-Field $k $(if ($script:over.ContainsKey($k)) { $script:over[$k] } else { $v[$k] })
    Show-Mark $k
  }
  $draftBox.Enabled = [bool]($m -and $m.mtp)
  $imgMinBox.Enabled = $imgMaxBox.Enabled = [bool]($m -and $m.vision)
  $noStreamChk.Enabled = [bool]($m -and $m.moe)
  Update-Speech
  Update-Fit
}
# whether the options fit: the model's fixed parts (weights, MTP block / drafter, buffers at the default chunk,
# snapshots, encoders) + the KV cache of the context asked (per slot) + speech to text, against the GPU memory less
# the driver's share. The same estimate as zero-server start -DryRun; with auto the server makes it fit itself.
# Drawn as the fit bar (weights | context | buffers) with the parts under it.
function Update-Fit {
  $m = Sel-Entry
  $mm = if ($m) { $m.memory } else { $null }
  $script:fitParts = $null; $fitBar.Invalidate()
  if (-not $mm -or -not $mm.device) { $fitLbl.Text = $(if ($m) { "no estimate yet (Refresh list)" } else { "" }); $fitLbl.ForeColor = $Pal.sub; return }
  $free = [double]$mm.device - $mm.keep_free
  $gp = [math]::Max(1, [int](Get-Field "gpus"))  # (split: about 1 / N of the weights, snapshots and context per GPU)
  $w = [double]$mm.weights / $gp
  $g = Sel-Group; $enc = $(if ($g -and -not $g.img) { 0 } else { [double]$mm.encoders })  # (Images off: no encoder)
  $buf = [double]$mm.buffers + $mm.snapshots / $gp + $(if ($gp -gt 1) { 0 } else { [double]$mm.mtp + $enc })  # (a split runs without MTP and images)
  $wh = Get-Field "whisper"; $as = Get-Field "asr"
  $speech = if ($wh -notin "", "off") { 8GB } elseif ($script:over.ContainsKey("asr") -and $as -notin "", "off") { 3.7GB } else { 0 }
  $ctx = Get-Field "ctx"
  $slots = [math]::Max(1, [int](Get-Field "slots"))
  # (weights kept in RAM, serverz --list-models memory.host_weights: a model larger than a discrete card's memory, or a
  #  tight iGPU share; generation reads them over PCIe on a card, so it is slower)
  $hostTxt = if ($mm.host_weights -and [double]$mm.host_weights -gt 0) { "; {0:N1} GiB of the weights in RAM (slower generation)" -f ([double]$mm.host_weights / 1GB) } else { "" }
  $auto = $ctx -eq "auto" -or -not $script:over.ContainsKey("ctx")
  if (-not $auto -and $ctx -notmatch '^\d+$') { $fitLbl.Text = "context: a number of tokens, or auto"; $fitLbl.ForeColor = $Pal.bad; return }
  if ($auto) {
    $actx = Get-AutoCtx $m
    $kv = [double]$mm.context / $gp * $(if ($m.ctx -and $actx) { [double]$actx / $m.ctx } else { 1 })
    $kv = [math]::Max([double]0, [math]::Min([double]$kv, [double]($free - $w - $buf)))  # (the server sizes the auto context to what is left)
  } else { $kv = [double]$ctx / 1000 * $mm.context_per_1k * $slots / $gp }
  $need = $w + $kv + $buf + $(if ($auto) { 0 } else { $speech })
  $script:fitParts = @{ weights = $w; context = $kv; buffers = $buf + $(if ($auto) { 0 } else { $speech }); free = $free; need = $need; auto = $auto }
  $fitBar.Invalidate()
  $parts = "weights {0:N1} | context {1:N1} | buffers {2:N1}" -f ($w / 1GB), ($kv / 1GB), ($script:fitParts.buffers / 1GB)
  $fitBar.AccessibleName = "GPU memory {0:N1} of {1:N1} GiB" -f ($need / 1GB), ($free / 1GB)
  if ($auto) {
    $fitLbl.Text = "{0:N1} / {1:N1} GiB at the auto context ({2:N0} tokens{3}): {4}{5}" -f ($need / 1GB), ($free / 1GB), (Get-AutoCtx $m), $(if ($speech) { "; smaller with speech on" } else { "" }), $parts, $hostTxt
    $fitLbl.ForeColor = $Pal.sub
  } elseif ($need -le $free) {
    $fitLbl.Text = "fits: {0:N1} / {1:N1} GiB: {2}{3}{4}" -f ($need / 1GB), ($free / 1GB), $parts, $(if ($speech) { ", speech {0:N1}" -f ($speech / 1GB) } else { "" }), $hostTxt
    $fitLbl.ForeColor = $(if ($need -gt 0.9 * $free) { $Pal.warn } else { $Pal.ok })
  } else {
    $fitLbl.Text = "won't fit: {0:N1} / {1:N1} GiB ({2}). A smaller context, Images off, or auto" -f ($need / 1GB), ($free / 1GB), $parts
    $fitLbl.ForeColor = $Pal.bad
  }
}
$portBox.Add_ValueChanged({ Update-Api; $script:sync.base = Base; $script:sync.poke = $true })

$detail = New-Info
$monitor = New-Info
# the running server's GPU memory as a bar above its text
$memBar = New-Object System.Windows.Forms.ProgressBar
$memBar.Dock = "Top"; $memBar.Height = S 8; $memBar.Minimum = 0; $memBar.Maximum = 100; $memBar.Visible = $false
$memLbl = New-Object System.Windows.Forms.Label; $memLbl.Dock = "Top"; $memLbl.Height = S 22; $memLbl.Visible = $false
$runPanel = New-Object System.Windows.Forms.Panel; $runPanel.Dock = "Fill"
$runPanel.Controls.Add($monitor); $runPanel.Controls.Add($memBar); $runPanel.Controls.Add($memLbl)

# the server's log, live (the last 500 lines)
$logBox = New-Info; $logBox.WordWrap = $false; $logBox.ScrollBars = "Both"
$logTools = New-Object System.Windows.Forms.FlowLayoutPanel; $logTools.Dock = "Top"; $logTools.Height = S 40; $logTools.WrapContents = $false
$logFollow = New-Check "Follow new lines"; $logFollow.Checked = $true
$logCopy = New-Button "Copy" 70; $logIssue = New-Button "Copy for an issue" 130; $logOpen = New-Button "Open file" 90; $logFolder = New-Button "Open folder" 100
$logTools.Controls.AddRange(@($logCopy, $logIssue, $logOpen, $logFolder, $logFollow))
$logPanel = New-Object System.Windows.Forms.Panel; $logPanel.Dock = "Fill"
$logPanel.Controls.Add($logBox); $logPanel.Controls.Add($logTools)
$script:logStamp = $null

# a strip of three tabs (drawn as flat buttons: the WinForms tab control cannot be themed)
$tabStrip = New-Object System.Windows.Forms.FlowLayoutPanel; $tabStrip.Dock = "Top"; $tabStrip.Height = S 38; $tabStrip.WrapContents = $false
$tabBody = New-Object System.Windows.Forms.Panel; $tabBody.Dock = "Fill"
$tabBody.Padding = New-Object System.Windows.Forms.Padding((S 8), (S 4), (S 4), (S 4))
$script:tabs = [ordered]@{ "Running server" = $runPanel; "Selected model" = $detail; "Log" = $logPanel }
$script:tabBtns = @{}
$script:tab = ""
function Select-Tab([string]$name) {
  $script:tab = $name
  foreach ($n in $script:tabs.Keys) {
    $on = $n -eq $name
    $script:tabs[$n].Visible = $on
    $b = $script:tabBtns[$n]
    $b.ForeColor = if ($on) { $Pal.accent } else { $Pal.sub }
    $b.Font = if ($on) { $fontBold } else { $form.Font }
    $b.FlatAppearance.BorderColor = if ($on) { $Pal.accent } else { $Pal.card }
  }
  if ($name -eq "Log") { Update-Log -Force }
}
foreach ($n in $script:tabs.Keys) {
  $b = New-Button $n 130; $b.BackColor = $Pal.card; $b.FlatAppearance.BorderColor = $Pal.card; $b.Tag = "tab"
  $b.Add_Click({ param($s) Select-Tab $s.Text })
  $script:tabBtns[$n] = $b
  [void]$tabStrip.Controls.Add($b)
  $script:tabs[$n].Dock = "Fill"
  [void]$tabBody.Controls.Add($script:tabs[$n])
}
$rightInner = New-Object System.Windows.Forms.Panel; $rightInner.Dock = "Fill"
$rightInner.Controls.Add($tabBody); $rightInner.Controls.Add($tabStrip)

Set-Splash "Laying out the window..." 65
# (docs/launcher-design.md: the models on the left, one row a file, over the running server, its log and the model's
#  details; the selected model's switches on the right)
$gridCard = New-Card "" $grid
$split = New-Object System.Windows.Forms.SplitContainer
$split.Dock = "Fill"; $split.Orientation = "Horizontal"; $split.SplitterWidth = S 6
[void]$split.Panel1.Controls.Add($gridCard)
[void]$split.Panel2.Controls.Add((New-Card "" $rightInner))
$bottom = New-Object System.Windows.Forms.TableLayoutPanel
$bottom.Dock = "Fill"; $bottom.ColumnCount = 2; $bottom.RowCount = 1
$bottom.Padding = New-Object System.Windows.Forms.Padding((S 8), 0, (S 8), 0)
foreach ($w in 54, 46) { [void]$bottom.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", $w))) }
$bottom.Controls.Add($split, 0, 0)
$bottom.Controls.Add((New-Card "The selected model (Start uses these)" $opt), 1, 0)

# ---- buttons and message line: the main actions; the rest in More
$buttons = New-Object System.Windows.Forms.FlowLayoutPanel
$buttons.Dock = "Bottom"; $buttons.Height = S 50; $buttons.Padding = New-Object System.Windows.Forms.Padding((S 12), (S 6), (S 12), (S 2))
$startBtn = New-Button "Start" 140 -Primary
$stopBtn = New-Button "Stop server" 120
$buttons.Controls.AddRange(@($startBtn, $stopBtn))
$msg = New-Object System.Windows.Forms.Label
$msg.Dock = "Bottom"; $msg.Height = S 26; $msg.Padding = New-Object System.Windows.Forms.Padding((S 16), (S 4), (S 10), 0)
$msg.ForeColor = $Pal.sub; $msg.AutoEllipsis = $true
$msg.Text = "Reading the model list..."

# loading progress: the server's "progress: P % <step>" lines in its log (src/load_progress.h) while a start runs
$loadBar = New-Object System.Windows.Forms.ProgressBar
$loadBar.Dock = "Top"; $loadBar.Height = S 6; $loadBar.Minimum = 0; $loadBar.Maximum = 100; $loadBar.Visible = $false
$loadBar.MarqueeAnimationSpeed = 30

$form.Controls.Add($bottom)
$form.Controls.Add($buttons)
$form.Controls.Add($msg)
$form.Controls.Add($bar)      # (docked under the loading bar: the later Add docks first)
$form.Controls.Add($loadBar)
$form.Controls.Add($header)
$form.Add_Load({ $split.SplitterDistance = [int]($split.Height * 0.5); Select-Tab "Running server" })

# ---- the More menu
$more = New-Object System.Windows.Forms.ContextMenuStrip
$more.Font = $form.Font; $more.BackColor = $Pal.card; $more.ForeColor = $Pal.text; $more.ShowImageMargin = $false
function Add-Item($menu, [string]$text, [scriptblock]$onClick) {
  $it = New-Object System.Windows.Forms.ToolStripMenuItem($text)
  $it.BackColor = $Pal.card; $it.ForeColor = $Pal.text
  if ($onClick) { $it.Add_Click($onClick) }
  [void]$menu.Items.Add($it); $it
}
function Add-Sep($menu) { [void]$menu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator)) }
$moreBtn.Add_Click({ $more.Show($moreBtn, 0, $moreBtn.Height) })

# ---- the selected model: what it supports and how --ctx auto counted the GPU memory
function Show-Detail {
  if ($grid.SelectedRows.Count -eq 0) { $detail.Text = ""; return }
  $m = $grid.SelectedRows[0].Tag
  $L = New-Object System.Collections.Generic.List[string]
  $L.Add("$($m.name)")
  $L.Add(("file    $($m.file)  ({0:N1} GB)" -f ($m.bytes / 1e9)))
  $L.Add("arch    $($m.arch)  -  " + $(if ($m.supported) { "supported by inferzero" } else { "NOT supported by inferzero" }))
  if ($m.note) { $L.Add("note    $($m.note)") }
  if ($m.supported -and $m.moe -and -not (Is-Package $m)) { $L.Add("package: More > Pack makes an inferzero package of it (then its original GGUF can be deleted)") }
  $L.Add("id      $($m.alias)" + $(if ($m.short) { "   (shortcut: zero-pi $($m.short))" } else { "" }))
  $L.Add("variant $(if ($m.tags) { $m.tags } else { '-' })")
  $L.Add("MTP / drafter: $(if ($m.mtp) { 'yes (drafts checked by the main model: same output, faster)' } else { 'no' })")
  $L.Add("images: $(if ($m.vision) { 'yes (encoder from the mmproj file)' } else { 'no' })")
  $L.Add(("max context length (model): {0:N0} tokens" -f $m.ctx_train))
  $mm = $m.memory
  if ($m.supported -and $mm) {
    $budget = $mm.device - $mm.keep_free - $mm.snapshots - $mm.encoders
    $L.Add("")
    $L.Add("Auto context (--ctx auto):")
    $L.Add(("  GPU memory (Intel setting)    {0,12}" -f (GiB $mm.device)))
    $L.Add(("  - left for the driver         {0,12}" -f (GiB $mm.keep_free)))
    $L.Add(("  - conversation snapshots      {0,12}" -f (GiB $mm.snapshots)))
    $L.Add(("  - image / audio encoders      {0,12}" -f (GiB $mm.encoders)))
    $L.Add(("  = for the model               {0,12}" -f (GiB $budget)))
    $L.Add(("  - weights{0}{1,12}" -f $(if ($mm.resident) { " (+ W8 copies)     " } else { "                   " }), (GiB $mm.weights)))
    $L.Add(("  - MTP block / drafter         {0,12}" -f (GiB $mm.mtp)))
    $L.Add(("  - working buffers             {0,12}" -f (GiB $mm.buffers)))
    $L.Add(("  = left for the context        {0,12}" -f (GiB ($budget - $mm.weights - $mm.mtp - $mm.buffers))))
    $L.Add(("  context per 1,000 tokens      {0,12}" -f ("{0:N1} MiB" -f ($mm.context_per_1k / 1MB))))
    $L.Add(("  -> {0:N0} tokens (KV {1}; multiples of 4,096, at most the max context length)" -f $m.ctx, (GiB $mm.context)))
  } elseif ($m.supported) {
    $L.Add(""); $L.Add("(no memory details yet: Refresh list)")
  }
  $detail.Text = $L -join "`r`n"
}

# ---- the running server: /v1/models and /props, asked by a background runspace every 1.5 s (the window never waits
# on the network: a busy or hung server froze it for up to 2 s at a time)
$script:sync = [hashtable]::Synchronized(@{ base = "http://127.0.0.1:$Port"; quit = $false; poke = $false; id = $null; props = $null; n = 0 })
$pollRs = [runspacefactory]::CreateRunspace(); $pollRs.Open()
$pollPs = [powershell]::Create(); $pollPs.Runspace = $pollRs
[void]$pollPs.AddScript({
    param($sync)
    while (-not $sync.quit) {
      $b = $sync.base; $id = $null; $p = $null
      try {
        $d = @((Invoke-RestMethod "$b/v1/models" -TimeoutSec 2).data)
        $l = @($d | Where-Object { $_.loaded -eq $true })
        $id = if ($l.Count) { $l[0].id } elseif ($d.Count) { $d[0].id } else { $null }
        try { $p = Invoke-RestMethod "$b/props" -TimeoutSec 2 } catch {}
      } catch {}
      if ($b -eq $sync.base) { $sync.id = $id; $sync.props = $p; $sync.n++ }
      for ($i = 0; $i -lt 15 -and -not $sync.quit -and -not $sync.poke; $i++) { Start-Sleep -Milliseconds 100 }
      $sync.poke = $false
    }
  }).AddArgument($script:sync)
$null = $pollPs.BeginInvoke()

function Update-Status {
  $id = $script:sync.id; $p = $script:sync.props; $busyAct = $script:actProc -and -not $script:actProc.HasExited
  $script:running = $id; $script:props = $p
  $st = $null
  if (Test-Path $stateFile) { try { $st = Get-Content -Encoding UTF8 $stateFile -Raw | ConvertFrom-Json } catch {} }
  if ($id) {
    $up = ""
    if ($st -and $st.started) { try { $up = "  |  up {0:hh\:mm\:ss}" -f ((Get-Date) - [datetime]$st.started) } catch {} }
    $busy = if ($p -and $p.stats.busy) { "  |  answering" } else { "" }
    $status.Text = ("On  -  {0}  |  context {1:N0}  |  {2}{3}{4}" -f $id, $(if ($p) { $p.n_ctx } else { 0 }), (Base), $up, $busy)
    $status.ForeColor = $Pal.ok
    # (an Intel graphics driver older than the oldest measured right: serverz's /props driver_warning, 2026-11-20)
    if ($p -and $p.driver_warning) {
      $status.Text += "  |  OLD GRAPHICS DRIVER: answers may be wrong (details below)"
      $status.ForeColor = $Pal.warn
      # (once per server start: a bar with a button to Intel's driver page, W2's last part, 2026-11-21)
      if ($script:drvBarFor -ne $id) {
        $script:drvBarFor = $id
        Show-Bar "Windows Update may have installed an old Intel graphics driver: answers may be wrong. Install Intel's current driver, then restart." "warn" -Details $p.driver_warning -ActionText "Intel's driver page" -Action { Start-Process "https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html" }
      }
    }
  } elseif ($busyAct) {
    $status.Text = "Loading / stopping..."
    $status.ForeColor = $Pal.warn
  } elseif ($script:sync.n -eq 0) {
    $status.Text = "Checking the server..."
    $status.ForeColor = $Pal.sub
  } else {
    $status.Text = "Off"
    $status.ForeColor = $Pal.sub
  }
  $stopBtn.Enabled = [bool]$id -and -not $busyAct
  $startBtn.Enabled = -not $busyAct -and $grid.SelectedRows.Count -gt 0
  $script:speechItem.Enabled = -not $busyAct
  Update-Tray $id $busyAct
  # monitoring text
  if ($id -and $p) {
    $L = New-Object System.Collections.Generic.List[string]
    $mem = $p.memory
    if ($p.driver_warning) {
      $L.Add("WARNING  $($p.driver_warning)")
      $L.Add("")
    }
    $L.Add("model    $id")
    $L.Add(("context  {0:N0} tokens, {1:N0} filled   chunk {2}" -f $p.n_ctx, $p.stats.position, $p.chunk))
    $L.Add(("MTP      {0}   images {1}   audio {2}" -f $(if ($p.mtp) { "yes ($($p.mtp_drafts) drafts)" } else { "no" }), $(if ($p.vision) { "yes" } else { "no" }), $(if ($p.audio) { "yes" } else { "no" })))
    if ($mem) {
      $pct = [math]::Min([double]100, [math]::Round(100 * [double]$mem.total / [math]::Max([double]1, [double]$mem.device)))  # (doubles: bytes past 2^31 failed the Int32 overload, issue #31)
      $memBar.Value = [int]$pct; $memBar.Visible = $memLbl.Visible = $true
      $memLbl.Text = "GPU memory: {0} of {1} ({2} %)" -f (GiB $mem.total), (GiB $mem.device), $pct
      $L.Add("")
      $L.Add(("  weights                {0,12}" -f (GiB $mem.weights)))
      $L.Add(("  context (KV)           {0,12}" -f (GiB $mem.context)))
      $L.Add(("  MTP / drafter          {0,12}" -f (GiB $mem.mtp)))
      $L.Add(("  working buffers        {0,12}" -f (GiB $mem.buffers)))
      $L.Add(("  encoders (loaded)      {0,12}" -f (GiB $mem.encoders)))
    }
    $L.Add("")
    $L.Add("requests done  $($p.stats.requests)" + $(if ($p.stats.busy) { "   (one in progress)" } else { "" }))
    $t = $p.stats.last
    if ($t) {
      $L.Add("last request:")
      $L.Add(("  prompt  {0} new tokens ({1} from cache), {2:N1} tok/s" -f $t.prompt_n, $t.cache_n, $t.prompt_per_second))
      $L.Add(("  answer  {0} tokens, {1:N1} tok/s  ({2})" -f $t.predicted_n, $t.predicted_per_second, $t.finish))
      if ($t.draft_n) { $L.Add(("  MTP     {0}/{1} drafts accepted ({2:N0}%)" -f $t.draft_n_accepted, $t.draft_n, (100 * $t.draft_n_accepted / [math]::Max(1, $t.draft_n)))) }
    }
    if ($st) { $L.Add(""); $L.Add("started with: ctx $($st.ctx), chunk $($st.chunk), $(if ($st.extra) { $st.extra } else { 'default options' })") }
    if ($st -and $st.cmd) { $L.Add(""); $L.Add("server command line:"); $L.Add($st.cmd) }
    $monitor.Text = $L -join "`r`n"
  } elseif (-not $id) {
    $monitor.Text = "The server is not running. Select a model and press Start (or double-click it)."
    $memBar.Visible = $memLbl.Visible = $false
  }
  foreach ($r in $grid.Rows) {
    $m = $r.Tag; $g = $r.Cells["model"].Tag
    $on = $id -and $g -and @($g.entries | Where-Object { $_.alias -eq $id }).Count
    $r.DefaultCellStyle.BackColor = if ($on) { $Pal.run } elseif ($m -and -not $m.supported) { $Pal.off } else { $Pal.card }
    $r.DefaultCellStyle.ForeColor = if ($m -and -not $m.supported) { $Pal.sub } else { $Pal.text }
  }
}

# ---- the log tab: the last 500 lines, read again when the file changed
function Update-Log([switch]$Force) {
  if ($script:tab -ne "Log") { return }
  if (-not (Test-Path $serverLog)) { $logBox.Text = "No log yet: the server has not been started."; return }
  $fi = Get-Item $serverLog
  $stamp = "$($fi.Length)|$($fi.LastWriteTimeUtc.Ticks)"
  if (-not $Force -and $stamp -eq $script:logStamp) { return }
  $script:logStamp = $stamp
  $lines = @(Get-Content -Encoding UTF8 -LiteralPath $serverLog -Tail 500 -ErrorAction SilentlyContinue | Where-Object { $_ -notmatch '^progress: ' })
  $logBox.Text = $lines -join "`r`n"
  if ($logFollow.Checked) { $logBox.SelectionStart = $logBox.TextLength; $logBox.ScrollToCaret() }
}
$logCopy.Add_Click({ try { [System.Windows.Forms.Clipboard]::SetText($(if ($logBox.SelectionLength) { $logBox.SelectedText } else { $logBox.Text })) } catch {}; $msg.Text = "Log copied." })
# "Copy for an issue" (docs/logging.md): the whole log since the start (its first lines name the build, system, GPU and
# switches), as a GitHub code block, the user folder and name replaced: paste it into an issue as it is. The last
# 2,000 lines when it is longer (the start is kept: the first 40).
$logIssue.Add_Click({
  if (-not (Test-Path $serverLog)) { $msg.Text = "No log yet: the server has not been started."; return }
  try {
    # (read as UTF-8, the server's encoding: a non-ASCII user name read as ANSI would not be found to replace; the
    #  whole file at once, filtered as an array: line by line froze the window on a long log)
    $all = @((Get-Content -LiteralPath $serverLog -Encoding UTF8 -ReadCount 0 -ErrorAction Stop) -notmatch '^progress: ')
    if ($all.Count -gt 2040) { $all = @($all[0..39]) + @("... ($($all.Count - 2040) lines left out) ...") + @($all[($all.Count - 2000)..($all.Count - 1)]) }
    $t = $all -join "`n"
    $home1 = $env:USERPROFILE
    if ($home1) {  # (any case, \ or /)
      $pat = [regex]::Escape($home1).Replace("\\", "[\\/]")
      $t = [regex]::Replace($t, "(?i)" + $pat, "%USERPROFILE%")
    }
    if ($env:USERNAME -and $env:USERNAME.Length -ge 3) { $t = [regex]::Replace($t, "(?i)[\\/]Users[\\/]" + [regex]::Escape($env:USERNAME) + "\b", "\Users\<user>") }
    # (a fence of four backticks: a log line with three does not end it)
    [System.Windows.Forms.Clipboard]::SetText("Server log (inferzero, from zero-launcher):`n````````text`n$t`n````````")
    $msg.Text = "Log copied for an issue ($($all.Count) lines; your user folder replaced by %USERPROFILE%): paste it on GitHub."
  } catch { $msg.Text = "Could not copy the log: $($_.Exception.Message)" }
})
$logOpen.Add_Click({ if (Test-Path $serverLog) { Start-Process notepad.exe -ArgumentList "`"$serverLog`"" } })
$logFolder.Add_Click({ Start-Process explorer.exe -ArgumentList "`"$stateDir`"" })

# ---- model list
# no bin\serverz.exe (a git clone, which has no exes): the list stays empty, so say why in the panels people look at,
# and offer setup.cmd once per window (issue #5: the line under the list alone was missed)
$script:askedSetup = $false
function Show-NoExe {
  $setup = Join-Path $root "setup.cmd"
  $msg.Text = "bin\serverz.exe not found."
  Show-Bar "bin\serverz.exe is missing, so no model can be listed or started. Take the ready package from the releases page, or build it with setup.cmd." "error" `
    -ActionText "Releases page" -Action { Start-Process "https://github.com/rizagp/inferzero/releases" }
  $detail.Text = (@(
      "bin\serverz.exe is missing, so the model list is empty.",
      "",
      "A copy from git has no exes. Two ways:",
      "  1. Download the ready package (exes built) from",
      "     https://github.com/rizagp/inferzero/releases",
      "     unpack it and run zero-launcher.cmd from there.",
      "  2. Build them here: run setup.cmd in",
      "     $root",
      "     (it needs Visual Studio's C++ tools), then press Refresh list.") -join "`r`n")
  Select-Tab "Selected model"
  if (-not $script:askedSetup -and (Test-Path $setup) -and -not $SelfTest) {
    $script:askedSetup = $true
    $q = "bin\serverz.exe is missing, so no model can be listed or started.`n`n" +
      "Run setup.cmd now to build it? (It needs Visual Studio's C++ tools; the ready package from the GitHub releases page has the exes built.)"
    if ([System.Windows.Forms.MessageBox]::Show($q, "inferzero", "YesNo") -eq "Yes") {
      Start-Process -FilePath $setup -WorkingDirectory $root
      $msg.Text = "setup.cmd is running in its own window: press Refresh list (More) when it has finished."
    }
  }
}
function Start-List {
  if ($script:listProc -and -not $script:listProc.HasExited) { return }
  $script:exe = Exe
  if (-not (Test-Path $script:exe)) { Show-NoExe; return }
  $msg.Text = "Reading the model list (up to a minute per model the first time: the auto context is computed)..."
  Remove-Item $script:listOut -ErrorAction SilentlyContinue
  $script:listProc = Start-Process -FilePath $script:exe -ArgumentList "--list-models" -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $script:listOut -RedirectStandardError (Join-Path $env:TEMP "zero-launcher-models.err")
}
# whether a model fits this GPU: its weights, drafter, buffers and encoders against the GPU memory ("RAM": a MoE
# model whose experts stream from RAM and the SSD)
function Get-FitWord($m) {
  if (-not $m.supported) { return "no ($($m.arch))" }
  $mm = $m.memory
  if (-not $mm -or -not $mm.device) { return "-" }
  $free = [double]$mm.device - $mm.keep_free
  $fixed = [double]$mm.weights + $mm.mtp + $mm.buffers + $mm.snapshots + $mm.encoders
  if ($m.moe -and [double]$m.bytes -gt $free) { return "RAM" }
  if ($fixed -le $free) { return "fits" }
  return "too big"
}
$script:groups = [ordered]@{}
$script:listing = $false
function Show-List {
  $sel = if ($grid.SelectedRows.Count) { $grid.SelectedRows[0].Tag.alias } else { $script:running }
  $selPath = if (-not $sel -and (Test-Path $lastFile)) { (Get-Content -Encoding UTF8 $lastFile -Raw).Trim() } else { $null }
  $old = $script:groups
  $script:groups = [ordered]@{}
  foreach ($m in $script:models) {  # (a file's rows in the list's order: its main row first)
    $k = $(if ($m.path) { $m.path.ToLower() } else { $m.alias })
    if (-not $script:groups.Contains($k)) { $script:groups[$k] = @{ entries = New-Object System.Collections.ArrayList; path = $m.path } }
    [void]$script:groups[$k].entries.Add($m)
  }
  $script:listing = $true
  try {
    $grid.Rows.Clear()
    $selRow = -1
    foreach ($k in $script:groups.Keys) {
      $g = $script:groups[$k]; $e0 = $g.entries[0]
      $g.title = $e0.name; $g.draft0 = Get-DraftMode $e0; $g.img0 = [bool]$e0.vision
      # (the choices made for the file stay over a refresh of the list; the running row's choices otherwise)
      $hit = @($g.entries | Where-Object { $sel -and $_.alias -eq $sel })[0]
      if ($old.Contains($k)) { $g.draft = $old[$k].draft; $g.img = $old[$k].img; $g.imgWant = $old[$k].imgWant }
      elseif ($hit) { $g.draft = Get-DraftMode $hit; $g.img = [bool]$hit.vision; $g.imgWant = $g.img }
      else { $g.draft = $g.draft0; $g.img = $g.img0 }
      $i = $grid.Rows.Add($e0.name + "  -  " + $e0.file, ("{0:N1} GB" -f ($e0.bytes / 1e9)), (Get-FitWord $e0),
        $(if ($e0.supported -and $e0.ctx) { "{0:N0}" -f $e0.ctx } else { "-" }),
        $(if ($e0.short) { "$($e0.alias)  [$($e0.short)]" } else { $e0.alias }))
      $grid.Rows[$i].Cells["model"].Tag = $g
      $grid.Rows[$i].Tag = Resolve-Entry $g
      if ($selRow -lt 0 -and (($hit) -or ($selPath -and -not $sel -and $g.path -and [IO.Path]::GetFullPath($g.path) -eq [IO.Path]::GetFullPath($selPath)))) { $selRow = $i }
    }
    $grid.ClearSelection()
    if ($selRow -ge 0) { $grid.Rows[$selRow].Selected = $true; $grid.FirstDisplayedScrollingRowIndex = $selRow }
  } finally { $script:listing = $false }
  $n = @($script:groups.Values | Where-Object { $_.entries[0].supported }).Count
  $msg.Text = if ($n -eq 0) { "No models found: add your model folder (More > Model folders...), or download one." }
              else { "$n models can run. Select one, choose its switches on the right, and Start (or double-click it). The server keeps running when this window is closed." }
  Refresh-Model
  Show-Detail
  Fill-Defaults
  Update-Status
  Update-TrayMenu
  After-List  # (scripts\launcher-firstrun.ps1: start what was just downloaded, or offer models when there are none)
}

# ---- start / stop through zero-server.ps1 (a separate process: the server outlives this window)
function Start-Action([string[]]$argList) {
  if ($script:actProc -and -not $script:actProc.HasExited) { return }
  Remove-Item $script:actOut, $script:actErr -ErrorAction SilentlyContinue
  $script:actWhat = $argList[0]
  $script:actStart = Get-Date  # (the server's log counts only once rewritten by this start: not the last run's lines)
  $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$arcServer`"") + $argList + @("-Port", [int]$portBox.Value)
  $script:actProc = Start-Process -FilePath $shell -ArgumentList $a -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $script:actOut -RedirectStandardError $script:actErr
  $null = $script:actProc.Handle  # (Windows PowerShell 5.1: without the handle taken now, ExitCode is $null once it exits)
  $bar.Visible = $false
  Update-Status
}
# other zero-server commands in the background (record, doctor, version -Check): $done gets the exit code and the output
function Start-Job2([string[]]$argList, [scriptblock]$done) {
  $id = [guid]::NewGuid().ToString("N").Substring(0, 8)
  $out = Join-Path $env:TEMP "zero-launcher-$id.out"; $err = Join-Path $env:TEMP "zero-launcher-$id.err"
  $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$arcServer`"") + $argList
  $p = Start-Process -FilePath $shell -ArgumentList $a -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  $null = $p.Handle
  [void]$script:jobs.Add(@{ proc = $p; out = $out; err = $err; done = $done })
}
function Check-Jobs {
  foreach ($j in @($script:jobs)) {
    if (-not $j.proc.HasExited) { continue }
    $script:jobs.Remove($j)
    $text = (Get-Content -Encoding UTF8 $j.out -Raw -ErrorAction SilentlyContinue) + (Get-Content -Encoding UTF8 $j.err -Raw -ErrorAction SilentlyContinue)
    Remove-Item $j.out, $j.err -ErrorAction SilentlyContinue
    try { & $j.done $j.proc.ExitCode "$text" } catch { Show-Bar "Launcher error: $($_.Exception.Message)" "error" -Details "$($_.ScriptStackTrace)" }
  }
}
# ---- the GPU benchmark (bin\gpu_bench.exe): the matrix engines (XMX), the vector engines and memory, measured against
# the GPU's figures on paper; a page in the data folder (bench\gpu\gpu-benchmark.html) shows every result there side
# by side, results from other PCs (their .json) dropped in that folder too
function Start-GpuBench {
  $gb = Join-Path $root "bin\gpu_bench.exe"
  if (-not (Test-Path $gb)) { Show-Bar "bin\gpu_bench.exe not found: unpack the latest package (or build it: .\build.ps1 bench\gpu_bench.cpp)." "warn"; return }
  if ($script:benching) { Show-Bar "The GPU benchmark is running already." "info"; return }
  $q = "Measure this GPU's matrix engines (XMX), vector engines and memory against its figures on paper?`n`nAbout half a minute; the GPU is busy meanwhile."
  if ($script:running) { $q += "`n`nThe server is running ($($script:running)): its work would lower the numbers. Stop it first for clean ones." }
  if ([System.Windows.Forms.MessageBox]::Show($q, "inferzero", "YesNo", "Question") -ne "Yes") { return }
  $script:benching = $true
  $id = [guid]::NewGuid().ToString("N").Substring(0, 8)
  $out = Join-Path $env:TEMP "zero-gpubench-$id.out"; $err = Join-Path $env:TEMP "zero-gpubench-$id.err"
  $p = Start-Process -FilePath $gb -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  $null = $p.Handle  # (Windows PowerShell 5.1: without the handle taken now, ExitCode is $null once it exits)
  $msg.Text = "Measuring the GPU (about half a minute)..."
  [void]$script:jobs.Add(@{ proc = $p; out = $out; err = $err; done = {
        param($code, $text)
        $script:benching = $false
        $page = (($text -split "`r?`n" | Where-Object { $_ -match '^page: ' } | Select-Object -First 1) -replace '^page: ', '').Trim()
        if ($code -ne 0 -or -not $page -or -not (Test-Path -LiteralPath $page)) {
          Show-Bar "The GPU benchmark failed (exit code $code)." "error" -Details ($text.Trim())
          return
        }
        $gpu = ($text -split "`r?`n" | Where-Object { $_ -match '^GPU: ' } | Select-Object -First 1) -replace '^GPU: ', ''
        Show-Bar "GPU benchmark done; the page is open in the browser. $gpu" "info" -Details ($text.Trim())
        Start-Process $page
      } })
}
# ---- pack (zero-server pack): a MoE model as an inferzero package, then, if the user agrees, the original GGUF deleted
function Is-Package($m) {
  $pk = [IO.Path]::GetFullPath((Join-Path $stateDir "zero\models")) + "\"
  return $m.path -and [IO.Path]::GetFullPath($m.path).StartsWith($pk, [StringComparison]::OrdinalIgnoreCase)
}
function Start-Pack {
  if ($grid.SelectedRows.Count -eq 0) { $msg.Text = "Select a MoE model first."; return }
  $m = $grid.SelectedRows[0].Tag
  if (-not $m.supported) { Show-Bar "This model cannot be packed: inferzero cannot load it." "warn"; return }
  if (-not $m.moe) { Show-Bar "Packages are for MoE models (35B-A3B, Flash Next, Gemma 4 26B): $($m.file) is a dense model." "warn"; return }
  if (Is-Package $m) { Show-Bar "$($m.file) is a package already." "info"; return }
  if ($script:packing) { Show-Bar "A model is being packed already." "info"; return }
  $q = ("Pack $($m.file) ({0:N1} GB) as an inferzero package?`n`nIt reads the whole model once (a few minutes; Flash Next about 10) and writes a package of about the same size to {1}.`n`n" +
        "The package can be used by inferzero only: llama.cpp, LM Studio and other programs cannot read it. Afterwards you can delete the original GGUF to free its space.") -f ($m.bytes / 1e9), (Join-Path $stateDir "zero\models")
  if ([System.Windows.Forms.MessageBox]::Show($q, "inferzero", "YesNo", "Question") -ne "Yes") { return }
  $script:packing = $true
  $script:packSrc = $m.path
  $script:packFile = $m.file
  $script:packBytes = $m.bytes
  $msg.Text = "Packing $($m.file)... (the model is read once; the window stays usable)"
  # (-Port: zero-server asks that port's server which model it has loaded)
  Start-Job2 @("pack", "-Model", "`"$($m.path)`"", "-Port", [int]$portBox.Value) {
    param($code, $text)
    $script:packing = $false
    if ($code -ne 0) {
      Show-Bar "Packing $($script:packFile) failed." "error" -Details ($text.Trim())
      return
    }
    # (the model list is read again after this question: a list being read is a serverz process, and remove-original
    #  refuses while a serverz it cannot ask runs)
    $q2 = ("The package of $($script:packFile) is ready, and inferzero now runs the model from it.`n`n" +
           "Delete the original GGUF ({0:N1} GB) to free its space?`n`n{1}`n`nWarning: the original GGUF is deleted for good. The package works in inferzero only; to use this model in another program you would download it again.") -f ($script:packBytes / 1e9), $script:packSrc
    if ([System.Windows.Forms.MessageBox]::Show($q2, "inferzero", "YesNo", "Warning") -ne "Yes") {
      # (the original is no longer listed once it has a package: the command, not the menu, deletes it later)
      Show-Bar "Packed $($script:packFile). The original GGUF is kept; to delete it later: zero-server remove-original -Model `"$($script:packSrc)`"" "info"
      Start-List
      return
    }
    Start-Job2 @("remove-original", "-Model", "`"$($script:packSrc)`"", "-Port", [int]$portBox.Value) {
      param($code2, $text2)
      if ($code2 -eq 0) { Show-Bar ("Packed $($script:packFile); the original GGUF was deleted. " + (($text2 -split "`r?`n" | Where-Object { $_ -match "GB freed" } | Select-Object -First 1) -replace '^remove-original: ', '')) "info" }
      else { Show-Bar "The original GGUF could not be deleted." "error" -Details ($text2.Trim()) }
      Start-List
    }
  }
}
# a model's row by id, shortcut or path
# (by id or shortcut: any row of the file, whose Drafting and Images it then takes)
function Find-Row([string]$v) {
  foreach ($r in $grid.Rows) {
    $g = $r.Cells["model"].Tag
    if (-not $g) { continue }
    $hit = @($g.entries | Where-Object { $_.supported -and ($_.alias -eq $v -or ($_.short -and $_.short -eq $v)) })[0]
    if ($hit) { $g.draft = Get-DraftMode $hit; $g.img = [bool]$hit.vision; $g.imgWant = $g.img; $r.Tag = Resolve-Entry $g; return $r }
  }
  foreach ($r in $grid.Rows) {
    $m = $r.Tag
    if ($m -and $m.supported -and $m.path -and $v -and (Test-Path -LiteralPath $v -ErrorAction SilentlyContinue) -and [IO.Path]::GetFullPath($m.path) -eq [IO.Path]::GetFullPath($v)) { return $r }
  }
  return $null
}
# zero-server's arguments for the selected model and the options as set (without -Port; $null when an option is wrong)
function Get-StartArgs([switch]$Speech) {
  if ($Speech) {
    $m = [pscustomobject]@{ alias = "none" }  # (no model's own options apply)
  } else {
    if ($grid.SelectedRows.Count -eq 0) { return $null }
    $m = $grid.SelectedRows[0].Tag
    if (-not $m.supported) { Show-Bar $(if ($m.note) { "$($m.file): $($m.note)" } else { "inferzero does not support this model yet (architecture $($m.arch))." }) "warn"; return $null }
  }
  $a = @("start", "-Model", $m.alias)
  # (the file's Drafting and Images: the row Resolve-Entry picked, and when that row has images or another drafter
  #  than asked, -Vision off and -Drafter; zero-server takes the file's row without images then)
  $g = $(if ($Speech) { $null } else { Sel-Group })
  if ($g) {
    if (-not $g.img -and $m.vision) { $a += @("-Vision", "off") }
    if ($g.draft -ne "off" -and (Get-DraftMode $m) -ne $g.draft) { $a += @("-Drafter", $(switch ($g.draft) { "both" { "auto" } default { $g.draft } })) }
    elseif ($g.draft -ne "off" -and -not $g.img -and $m.vision -and $g.draft -ne "both") { $a += @("-Drafter", $g.draft) }
  }
  foreach ($k in @($script:over.Keys)) {
    $v = $script:over[$k]
    switch ($k) {
      "ctx" {
        if ($v -and $v -ne "auto") {
          if ($v -notmatch '^\d+$') { Show-Bar "Context: a number of tokens, or auto." "warn"; return $null }
          $a += @("-Ctx", $v)
        } elseif ($v -eq "auto") { $a += @("-Ctx", "auto") }
      }
      "chunk" { if ("$v" -match '^\d+$') { $a += @("-Chunk", $v) } }  # (auto: none, the server chooses: a MoE model's 8,192 where it costs nothing, --mtp's smaller ones)
      "drafts" { if ($m.mtp) { $a += @("-MtpDrafts", $v) } }
      "think" { $a += @("-ThinkBudget", [int]$v) }
      "img_min" { if ($m.vision) { $a += @("-ImageMinTokens", [int]$v) } }
      "img_max" { if ($m.vision) { $a += @("-ImageMaxTokens", [int]$v) } }
      "budget" { if ($v -gt 0) { $a += @("-BudgetGb", [string]$v) } }
      "prompt_cache" { $a += @("-SessionCacheGb", [int]$v) }  # (0 = off)
      "prompt_cache_dir" { if ($v) { $a += @("-SessionDir", "`"$v`"") } }
      "prefix_cache" { $a += @("-PrefixCacheGb", [int]$v) }
      "no_prefix" { if ($v) { $a += "-NoPrefixCache" } }
      "no_swap" { if ($v) { $a += "-NoSwap" } }
      "network" { if ($v) { $a += @("-ListenHost", "0.0.0.0") } }
      "host" { if ($v -and -not ($a -contains "-ListenHost")) { $a += @("-ListenHost", "$v") } }  # (Network on wins: 0.0.0.0)
      "expert_ram" { if ($m.moe) { $a += @("-ExpertRamGb", ([double]$v).ToString([Globalization.CultureInfo]::InvariantCulture)) } }
      "expert_cache" { if ($m.moe -and [double]$v -gt 0) { $a += @("-ExpertCacheGb", ([double]$v).ToString([Globalization.CultureInfo]::InvariantCulture)) } }
      "driver_reserve" { $a += @("-DriverReserveGb", ([double]$v).ToString([Globalization.CultureInfo]::InvariantCulture)) }
      "asr_slots" { $a += @("-AsrSlots", [int]$v) }
      "mtp_model" { if ($v) { $a += @("-MtpModel", "`"$v`"") } }
      "mmproj" { if ($v) { $a += @("-Mmproj", "`"$v`"") } }
      "alias" { if ($v) { $a += @("-Alias", "$v") } }
      "log_file" { if ($v) { $a += @("-LogFile", "`"$v`"") } }
      "log_v" { $a += @("-LogVerbosity", "$v") }
      "no_web" { if ($v) { $a += "-NoWebTools" } }
      "no_term" { if ($v) { $a += "-NoTerminal" } }
      "attn_f32" { if ($v) { $a += "-AttnF32" } }
      # (no_mtp and vision: the Drafting and Images switches above; a preset's NoMtp / Vision set them)
      "slots" { $a += @("-Slots", [int]$v) }
      "gpus" { $a += @("-Gpus", [int]$v) }  # (1 too: a running split server goes back to one GPU)
      "npu" { $a += @("-Npu", $v) }  # (off too: a running server's on is kept otherwise)
      "precise" { if ($m.opts.flashnext -and -not $m.precise_only) { $a += $(if ($v) { "-Precise" } else { "-Fast" }) } }  # (on too, as -Npu: a running server keeps -Fast otherwise; Flash Next only)
      "no_stream" { if ($v -and $m.moe) { $a += "-NoStream" } }
      "preserve_think" { if ($v) { $a += "-PreserveThinking" } }
      "whisper" { if ($v) { $a += @("-Whisper", $(if ($v -in "auto", "off") { $v } else { "`"$v`"" })) } }
      "asr" { if ($v) { $a += @("-Asr", $(if ($v -in "auto", "off") { $v } else { "`"$v`"" })) } }
      "whisper_device" { $a += @("-WhisperDevice", $v) }
      "whisper_max" { $a += @("-WhisperMaxMinutes", [int]$v) }
    }
  }
  foreach ($k in $script:extraArgs.Keys) {
    # (one already given by a field above, -ListenHost by Network or -Drafter by Drafting: the field's, once;
    #  PowerShell refuses a parameter given twice, the launcher spec A5)
    if ($a -contains "-$k") { continue }
    $v = $script:extraArgs[$k]
    if ($v -is [bool]) { if ($v) { $a += "-$k" } } else { $a += @("-$k", $(if ("$v" -match '[\s\\/]') { "`"$v`"" } else { "$v" })) }
  }
  # (the switches whose choice differs from the one zero-server keeps in options.json: on / off / back to the
  #  default; Precise mode goes as -Precise above, a value option has its own field)
  $saved = Read-OptChoice
  $on = @(); $off = @(); $def = @()
  # (a greyed switch is left as options.json has it: Copy drafting on, then Drafting off, sent -Opt copy_draft while
  #  the box said "turn Drafting on first"; the Researcher's review of df7379c)
  $st = $(if ($Speech) { $null } else { Get-SwState })
  foreach ($o in $script:catalog) {
    $id = $o.id
    if ($id -eq "fn_precise" -or (Test-SwValue $o)) { continue }
    if ($st -and (Get-SwReason $script:catalog $o $st)) { continue }
    $want = $(if ($script:optChoice.ContainsKey($id)) { "$($script:optChoice[$id])" } else { "default" })
    $have = $(if ($saved.ContainsKey($id)) { "$($saved[$id])" } else { "default" })
    if ($want -eq $have) { continue }
    switch ($want) { "off" { $off += $id } "default" { $def += $id } default { $on += $(if ($want -eq "on") { $id } else { "$id=$want" }) } }
  }
  if ($on.Count) { $a += @("-Opt", ($on -join ",")) }
  if ($off.Count) { $a += @("-NoOpt", ($off -join ",")) }
  if ($def.Count) { $a += @("-DefaultOpt", ($def -join ",")) }
  , $a
}
function Start-Selected([switch]$Speech) {
  # (not the model being packed: the server's open file would keep its expert file from moving into the package)
  if ($script:packing -and -not $Speech -and $grid.SelectedRows.Count -and $grid.SelectedRows[0].Tag.path -and
      [IO.Path]::GetFullPath($grid.SelectedRows[0].Tag.path) -eq [IO.Path]::GetFullPath($script:packSrc)) {
    Show-Bar "$($script:packFile) is being packed: start it when the package is ready." "info"
    return
  }
  $a = Get-StartArgs -Speech:$Speech
  if ($null -eq $a) { return }
  $wh = if ((Get-Field "whisper") -notin "", "off") { " and Whisper" } else { "" }
  $msg.Text = if ($Speech) { "Starting speech to text only (Qwen3-ASR$wh)..." } else { "Starting $($a[2])... (loading takes 5-40 s; 35B up to 1-3 min)" }
  Start-Action $a
}
# issue #6: the command line that starts the server the way Start would (selected model, options as set), copied to
# the clipboard, for a script or shortcut of one's own. It works from cmd and PowerShell alike, wherever it is run.
function Copy-StartCommand {
  $a = Get-StartArgs
  if ($null -eq $a) { if ($grid.SelectedRows.Count -eq 0) { $msg.Text = "Select a model first." }; return }
  $a += @("-Port", [int]$portBox.Value)
  $cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$arcServer`" " + ($a -join " ")
  try { [System.Windows.Forms.Clipboard]::SetText($cmd) } catch {}
  $d = "$cmd`r`n`r`nThe same with the zero-server command (this folder's zero-server.cmd, or on PATH): zero-server " + ($a -join " ") +
    "`r`n`r`nzero-server help lists every option. The command line of the running bin\serverz.exe is in the Running server tab."
  Show-Bar "Start command copied to the clipboard: it starts the server like Start does, with the options as set now." "info" -Details $d
  $msg.Text = "Start command copied to the clipboard."
}

# ---- presets: %USERPROFILE%\.inferzero\presets\<name>.json in zero-server's option names ({"Model": "35b", "Ctx":
# "65536", "Npu": "on", "NoMtp": true}), so zero-server start -Preset <name> starts the same
$paramToField = [ordered]@{ Ctx = "ctx"; Chunk = "chunk"; MtpDrafts = "drafts"; ThinkBudget = "think"; ImageMinTokens = "img_min"
  ImageMaxTokens = "img_max"; BudgetGb = "budget"; SessionCacheGb = "prompt_cache"; SessionDir = "prompt_cache_dir"
  PrefixCacheGb = "prefix_cache"; NoPrefixCache = "no_prefix"; NoSwap = "no_swap"; AttnF32 = "attn_f32"; NoMtp = "no_mtp"
  Vision = "vision"; Slots = "slots"; Gpus = "gpus"; Npu = "npu"; NoStream = "no_stream"; PreserveThinking = "preserve_think"
  Whisper = "whisper"; Asr = "asr"; WhisperDevice = "whisper_device"; WhisperMaxMinutes = "whisper_max"; Port = "port"
  ExpertRamGb = "expert_ram"; ExpertCacheGb = "expert_cache"; DriverReserveGb = "driver_reserve"; AsrSlots = "asr_slots"
  MtpModel = "mtp_model"; Mmproj = "mmproj"; Alias = "alias"; LogFile = "log_file"; LogVerbosity = "log_v"; NoWebTools = "no_web"
  NoTerminal = "no_term" }
$script:presetName = ""
$script:presetFilling = $false
function Load-PresetList([string]$select = "") {
  $script:presetFilling = $true
  try {
    $presetBox.Items.Clear()
    [void]$presetBox.Items.Add("Automatic (best measured)")  # (index 0, the default: best-setups.tsv, Apply-Auto)
    [void]$presetBox.Items.Add("(none: the engine's defaults)")
    foreach ($f in @(Get-ChildItem -LiteralPath $presetDir -Filter *.json -ErrorAction SilentlyContinue | Sort-Object Name)) { [void]$presetBox.Items.Add($f.BaseName) }
    $i = if ($select) { $presetBox.Items.IndexOf($select) } else { $(if ($script:autoOn) { 0 } else { 1 }) }
    $presetBox.SelectedIndex = [math]::Max(0, $i)
  } finally { $script:presetFilling = $false }
  $presetDel.Enabled = $presetBox.SelectedIndex -gt 1
}
function Apply-Preset([string]$name) {
  $pp = Join-Path $presetDir "$name.json"
  try { $o = Get-Content -Encoding UTF8 -LiteralPath $pp -Raw | ConvertFrom-Json } catch { Show-Bar "Preset '$name' could not be read: $($_.Exception.Message)" "error" -Details $pp; return }
  $script:over = @{}; $script:extraArgs = [ordered]@{}
  $model = $null; $unknown = @()
  foreach ($p in $o.PSObject.Properties) {
    $k = $p.Name; $v = $p.Value
    if ($k -eq "Model") { $model = "$v"; continue }
    if ($k -eq "ListenHost") { if ("$v" -eq "0.0.0.0") { $script:over["network"] = $true } else { $script:over["host"] = "$v" }; continue }
    if ($k -eq "Precise") { $script:over["precise"] = [bool]$v; continue }
    if ($k -in "NoPrecise", "Fast") { if ($v) { $script:over["precise"] = $false }; continue }
    $fk = $null; foreach ($pk in $paramToField.Keys) { if ($pk -eq $k) { $fk = $paramToField[$pk] } }
    if ($fk) {
      $c = $fields[$fk]
      $script:over[$fk] = switch ($c.GetType().Name) { "NumericUpDown" { [double]$v } "CheckBox" { [bool]$v } default { "$v" } }
    } elseif ($k -in "DFlash", "Drafter") { $script:extraArgs[$k] = $v }
    elseif ($k -in "Opt", "NoOpt", "DefaultOpt") {
      # (a preset's switches by id, as zero-server takes them: -Opt id or id=value, -NoOpt id, -DefaultOpt id; they
      #  were dropped as unknown, the launcher spec A5, 2026-11-22)
      foreach ($e in @($v)) {
        $id, $val = "$e" -split '=', 2
        if (-not (Get-SwRow $script:catalog $id)) { $unknown += "$k $e"; continue }
        if ($k -eq "Opt") { $script:optChoice[$id] = $(if ($val) { $val } else { "on" }) }
        elseif ($k -eq "NoOpt") { $script:optChoice[$id] = "off" }
        else { $script:optChoice.Remove($id) }
      }
    }
    else { $unknown += $k }
  }
  $script:presetName = $name
  if ($model) {
    $r = Find-Row $model
    if ($r) { $grid.ClearSelection(); $r.Selected = $true; $grid.FirstDisplayedScrollingRowIndex = $r.Index }
    else { Show-Bar "Preset '$name': its model '$model' is not on this PC (the options are set)." "warn" }
  }
  # (a preset's Vision, NoMtp and Drafter: the file's Images and Drafting switches)
  $g = Sel-Group
  if ($g) {
    if ($script:over.ContainsKey("vision")) { $g.img = "$($script:over['vision'])" -ne "off"; $g.imgWant = $g.img }
    if ($script:over["no_mtp"]) { $g.draft = "off" }
    $dp = "$($script:extraArgs['Drafter'])".ToLower()
    if ($dp -in "auto", "mtp", "dflash", "off") {
      $want = $(if ($dp -eq "auto") { "both" } else { $dp })
      if (@($g.entries | Where-Object { (Get-DraftMode $_) -eq $want }).Count) { $g.draft = $want }
      elseif ($dp -eq "auto" -and @($g.entries | Where-Object { (Get-DraftMode $_) -eq "mtp" }).Count) { $g.draft = "mtp" }
      $script:extraArgs.Remove("Drafter")
    }
  }
  $script:over.Remove("vision"); $script:over.Remove("no_mtp")
  Refresh-Model
  Fill-Defaults
  $ex = if ($script:extraArgs.Count) { "; also " + (($script:extraArgs.Keys | ForEach-Object { "-$_ $($script:extraArgs[$_])" }) -join " ") } else { "" }
  if ($unknown.Count) { Show-Bar "Preset '$name': options the launcher does not know were left out: $($unknown -join ', ')" "warn" }
  $msg.Text = "Preset '$name' loaded$ex. Start uses it; zero-server start -Preset `"$name`" does the same."
}
# a small dialog asking for one line of text
function Read-Line([string]$title, [string]$prompt, [string]$value = "") {
  $d = New-Object System.Windows.Forms.Form
  $d.Text = $title; $d.FormBorderStyle = "FixedDialog"; $d.MaximizeBox = $false; $d.MinimizeBox = $false
  $d.StartPosition = "CenterParent"; $d.AutoScaleMode = "None"; $d.Font = $form.Font; $d.ClientSize = New-Object System.Drawing.Size((S 420), (S 130))
  $d.BackColor = $Pal.bg; $d.ForeColor = $Pal.text
  $l = New-Object System.Windows.Forms.Label; $l.Text = $prompt; $l.AutoSize = $true; $l.Location = New-Object System.Drawing.Point((S 14), (S 14))
  $t = New-Text 390; $t.Text = $value; $t.Location = New-Object System.Drawing.Point((S 14), (S 40))
  $ok = New-Button "OK" 90 -Primary; $ok.Location = New-Object System.Drawing.Point((S 216), (S 84)); $ok.DialogResult = "OK"
  $no = New-Button "Cancel" 90; $no.Location = New-Object System.Drawing.Point((S 314), (S 84)); $no.DialogResult = "Cancel"
  $d.Controls.AddRange(@($l, $t, $ok, $no)); $d.AcceptButton = $ok; $d.CancelButton = $no
  Set-WindowStyle $d
  if ($d.ShowDialog($form) -eq "OK") { return $t.Text.Trim() }
  return $null
}
$presetBox.Add_SelectedIndexChanged({
    if ($script:presetFilling) { return }
    $presetDel.Enabled = $presetBox.SelectedIndex -gt 1
    if ($presetBox.SelectedIndex -gt 1) { $script:autoOn = $false; Undo-Auto; Apply-Preset ([string]$presetBox.SelectedItem) }
    else {
      $script:presetName = ""
      $script:autoOn = $presetBox.SelectedIndex -eq 0
      Apply-Auto; Refresh-Model; Show-Detail; Fill-Defaults
    }
  })
$presetSave.Add_Click({
    $a = Get-StartArgs
    if ($null -eq $a) { if ($grid.SelectedRows.Count -eq 0) { $msg.Text = "Select a model first." }; return }
    $name = Read-Line "Save as preset" "Name of the preset (the model and every option changed are saved):" $script:presetName
    if (-not $name) { return }
    if ($name -notmatch '^[\w][\w .-]{0,63}$') { Show-Bar "A preset name: letters, digits, spaces, dots and dashes." "warn"; return }
    # zero-server's own option names, from the arguments Start would pass
    $o = [ordered]@{}
    for ($i = 1; $i -lt $a.Count; $i++) {
      $k = "$($a[$i])".TrimStart("-")
      if ($i + 1 -lt $a.Count -and "$($a[$i + 1])" -notmatch '^-[A-Za-z]') { $o[$k] = "$($a[$i + 1])".Trim('"'); $i++ } else { $o[$k] = $true }
    }
    if ($script:over.ContainsKey("port")) { $o["Port"] = [int]$portBox.Value }
    $pp = Join-Path $presetDir "$name.json"
    if ((Test-Path -LiteralPath $pp) -and $name -ne $script:presetName) {
      if ([System.Windows.Forms.MessageBox]::Show("A preset '$name' exists. Replace it?", "inferzero", "YesNo") -ne "Yes") { return }
    }
    New-Item -ItemType Directory -Force $presetDir | Out-Null
    [IO.File]::WriteAllText($pp, (ConvertTo-Json -InputObject $o -Depth 4))
    $script:presetName = $name
    Load-PresetList $name
    $msg.Text = "Preset '$name' saved ($pp). From a terminal: zero-server start -Preset `"$name`""
  })
$presetDel.Add_Click({
    if ($presetBox.SelectedIndex -le 1) { return }
    $name = [string]$presetBox.SelectedItem
    if ([System.Windows.Forms.MessageBox]::Show("Delete the preset '$name'? (Only its settings file; the model stays.)", "inferzero", "YesNo") -ne "Yes") { return }
    Remove-Item -LiteralPath (Join-Path $presetDir "$name.json") -ErrorAction SilentlyContinue
    $script:presetName = ""
    Load-PresetList
    $msg.Text = "Preset '$name' deleted."
  })

# ---- defaults: Set as default keeps the overridden fields (launcher.json, applied whenever the window opens) and
# makes the selected model the default one (last-model.txt: also zero-pi's choice when no model is given)
$settingsFile = Join-Path $stateDir "launcher.json"
$lastFile = Join-Path $stateDir "last-model.txt"
if (Test-Path $settingsFile) {
  try {
    $o = Get-Content -Encoding UTF8 $settingsFile -Raw | ConvertFrom-Json
    foreach ($p in $o.PSObject.Properties) { if ($fields.Contains($p.Name)) { $script:over[$p.Name] = $p.Value } }
  } catch {}
}
# (Advanced stays collapsed, the user's choice of 2026-11-22 for docs/launcher-design.md; the old No MTP and Images
#  fields are the Drafting and Images switches now, per model)
$script:over.Remove("vision"); $script:over.Remove("no_mtp")
function Set-AsDefault {
  $script:over | ConvertTo-Json | Set-Content $settingsFile -Encoding utf8
  $n = $script:over.Count
  $t = $(if ($n) { "$n changed option(s) saved as default." } else { "No changed options: every model uses its best values." })
  if ($grid.SelectedRows.Count -gt 0 -and $grid.SelectedRows[0].Tag.supported) {
    $m = $grid.SelectedRows[0].Tag
    Set-Content $lastFile $m.path -Encoding utf8
    $t = "Default model: $($m.alias) (also for zero-pi without a model). " + $t
  }
  $msg.Text = $t
  Update-TrayMenu
}
function Reset-Options {
  $script:over = @{}; $script:extraArgs = [ordered]@{}; $script:presetName = ""
  Load-PresetList
  Reset-Switches
  $msg.Text = "Options back to the best values for each model (Set as default to keep it that way)."
}

# ---- updates: once a day at most, the latest GitHub release against this package (zero-server version -Check);
# off in More when not wanted. Only a link: nothing is downloaded by itself.
$script:upd = @{ enabled = $true; last = ""; latest = ""; url = ""; newer = $false; pkg = "" }
# (the package a check was made for: the data folder is shared by every unpacked package, so a check made by an
#  older one said "newer" in the new one too for 20 hours: issue #8. Its folder and serverz.exe's time)
$pkgKey = "$root|" + $(try { (Get-Item -LiteralPath (Join-Path $root "bin\serverz.exe") -ErrorAction Stop).LastWriteTimeUtc.Ticks } catch { "" })
if (Test-Path $updateFile) { try { $u = Get-Content -Encoding UTF8 $updateFile -Raw | ConvertFrom-Json; foreach ($p in $u.PSObject.Properties) { $script:upd[$p.Name] = $p.Value } } catch {} }
function Save-Upd { try { ConvertTo-Json -InputObject $script:upd | Set-Content $updateFile -Encoding utf8 } catch {} }
function Check-Update([switch]$Manual) {
  if (-not $Manual) {
    if (-not $script:upd.enabled) { return }
    try { if ($script:upd.last -and $script:upd.pkg -eq $pkgKey -and ((Get-Date) - [datetime]$script:upd.last).TotalHours -lt 20) { if ($script:upd.newer) { Show-UpdateBar }; return } } catch {}
  }
  $script:manualUpd = [bool]$Manual
  Start-Job2 @("version", "-Check", "-Json") {
    param($code, $text)
    $v = $null; try { $v = $text | ConvertFrom-Json } catch {}
    if (-not $v -or -not $v.update -or $v.update.error) {
      if ($script:manualUpd) { Show-Bar "Could not ask GitHub for the latest release$(if ($v -and $v.update.error) { ': ' + $v.update.error })." "warn" }
      return
    }
    $script:upd.last = (Get-Date).ToString("s"); $script:upd.latest = $v.update.latest; $script:upd.url = $v.update.url; $script:upd.newer = [bool]$v.update.newer; $script:upd.pkg = $pkgKey
    Save-Upd
    if ($v.update.newer) { Show-UpdateBar } elseif ($script:manualUpd) { Show-Bar "Up to date: this is the latest package ($($v.update.latest)) or newer." "info" }
  }
}
function Show-UpdateBar {
  Show-Bar "A newer inferzero package is out: $($script:upd.latest). Unpack it into a new folder; your chats, settings and models stay where they are (what changed: the release notes)." "info" `
    -ActionText "Download page" -Action { Start-Process $script:upd.url }
}

# ---- doctor: zero-server doctor's report in a window with Copy (for a GitHub issue)
function Show-TextDialog([string]$title, [string]$text) {
  $d = New-Object System.Windows.Forms.Form
  $d.Text = $title; $d.StartPosition = "CenterParent"; $d.AutoScaleMode = "None"; $d.Font = $form.Font
  $d.Size = New-Object System.Drawing.Size((S 900), (S 560)); $d.BackColor = $Pal.bg; $d.ForeColor = $Pal.text
  $t = New-Info; $t.Text = $text; $t.WordWrap = $false; $t.ScrollBars = "Both"
  $p = New-Object System.Windows.Forms.FlowLayoutPanel; $p.Dock = "Bottom"; $p.Height = S 46; $p.FlowDirection = "RightToLeft"
  $p.Padding = New-Object System.Windows.Forms.Padding((S 8))
  $close = New-Button "Close" 90; $close.DialogResult = "Cancel"
  $copy = New-Button "Copy" 90 -Primary; $copy.Add_Click({ try { [System.Windows.Forms.Clipboard]::SetText($t.Text) } catch {} })
  $p.Controls.AddRange(@($close, $copy))
  $card = New-Card "" $t
  $d.Controls.Add($card); $d.Controls.Add($p); $d.CancelButton = $close
  Set-WindowStyle $d
  [void]$d.ShowDialog($form)
}
function Run-Doctor {
  $msg.Text = "Checking this PC (zero-server doctor)..."
  Start-Job2 @("doctor", "-Port", [int]$portBox.Value) { param($code, $text) $msg.Text = "Doctor report ready."; Show-TextDialog "inferzero doctor" ($text.Trim() -replace "(?<!`r)`n", "`r`n") }
}

# ---- meeting recording (zero-server record / record-stop; only with bin\zero_recorder.exe): App = the meeting app whose
# sound is "Others" (empty: everything the speakers play)
$hasRec = (Test-Path (Join-Path $root "bin\zero_recorder.exe")) -or (Test-Path (Join-Path $root "bin\arc_recorder.exe"))  # (arc_: a build before 2026-11-10)
function Start-Record {
  $app = Read-Line "Record a meeting" "The meeting app whose sound is 'Others' (Zoom.exe, Teams.exe, chrome.exe; empty: all the speakers play):" ""
  if ($null -eq $app) { return }
  $a = @("record")
  $app = $app.Replace('"', '')
  if ($app) { $a += @("-App", "`"$app`"") }
  $msg.Text = "Starting the recording..."
  Start-Job2 $a { param($code, $text) if ($code -ne 0) { Show-Bar "Recording failed (exit code $code)." "error" -Details $text } else { $msg.Text = ($text -split "`r?`n" | Where-Object { $_ }) -join "  " } }
}

# ---- the tray icon: on / off at a glance, Start the default model, Stop, the web UI; minimized, the window waits here
function New-ArcIcon([bool]$on) {
  $n = 32
  $bmp = New-Object System.Drawing.Bitmap($n, $n)
  $g = [System.Drawing.Graphics]::FromImage($bmp); $g.SmoothingMode = "AntiAlias"
  # (the mark as site/icon.svg, in the taskbar's light or dark colours, lit while a model runs: its Z in the icon's
  #  blue, grey when stopped; until 2026-11-17 the Windows accent colour and half-transparent dim tiles)
  if ($taskbarDark) { $lit = if ($on) { RGB 89 167 227 } else { RGB 128 128 128 }; $dim = RGB 43 52 64 }
  else { $lit = if ($on) { RGB 0 120 212 } else { RGB 128 128 128 }; $dim = RGB 201 211 222 }
  Draw-ArcMark $g 0 0 $n $lit $dim
  # (running: a small green dot over the last tile)
  if ($on) { $g.FillEllipse((New-Object System.Drawing.SolidBrush((RGB 16 180 60))), $n - 11, $n - 11, 10, 10); $g.DrawEllipse((New-Object System.Drawing.Pen([System.Drawing.Color]::White, 1.5)), $n - 11, $n - 11, 10, 10) }
  $g.Dispose()
  [System.Drawing.Icon]::FromHandle($bmp.GetHicon())
}
$iconOn = New-ArcIcon $true; $iconOff = New-ArcIcon $false
$form.Icon = $iconOff
$tray = New-Object System.Windows.Forms.NotifyIcon
$tray.Icon = $iconOff; $tray.Text = "inferzero"; $tray.Visible = -not $SelfTest
$trayMenu = New-Object System.Windows.Forms.ContextMenuStrip; $trayMenu.Font = $form.Font
$trayState = New-Object System.Windows.Forms.ToolStripMenuItem("inferzero"); $trayState.Enabled = $false
$trayShow = New-Object System.Windows.Forms.ToolStripMenuItem("Show the window")
$trayStart = New-Object System.Windows.Forms.ToolStripMenuItem("Start the default model")
$trayStop = New-Object System.Windows.Forms.ToolStripMenuItem("Stop the server")
$trayWeb = New-Object System.Windows.Forms.ToolStripMenuItem("Open the web UI")
$trayQuit = New-Object System.Windows.Forms.ToolStripMenuItem("Close the window (the server keeps running)")
[void]$trayMenu.Items.AddRange(@($trayState, (New-Object System.Windows.Forms.ToolStripSeparator), $trayShow, $trayStart, $trayStop, $trayWeb,
    (New-Object System.Windows.Forms.ToolStripSeparator), $trayQuit))
$tray.ContextMenuStrip = $trayMenu
function Show-Window { $form.Show(); $form.WindowState = "Normal"; $form.Activate() }
$tray.Add_DoubleClick({ Show-Window })
$trayShow.Add_Click({ Show-Window })
$trayWeb.Add_Click({ Start-Process "$(Base)/" })
$trayStop.Add_Click({ $msg.Text = "Stopping the server..."; Start-Action @("stop") })
$trayQuit.Add_Click({ $form.Close() })
$trayStart.Add_Click({
    $r = $null
    if (Test-Path $lastFile) { $r = Find-Row ((Get-Content -Encoding UTF8 $lastFile -Raw).Trim()) }
    if (-not $r) { Show-Window; $msg.Text = "No default model yet: select one and press Start."; return }
    $grid.ClearSelection(); $r.Selected = $true
    Start-Selected
  })
$script:trayWas = $null
function Update-Tray($id, $busy) {
  $key = "$id|$busy"
  if ($key -eq $script:trayWas) { return }
  $script:trayWas = $key
  $tray.Icon = if ($id) { $iconOn } else { $iconOff }
  $form.Icon = $tray.Icon
  $t = if ($id) { "inferzero: $id" } elseif ($busy) { "inferzero: loading..." } else { "inferzero: off" }
  $tray.Text = $t.Substring(0, [math]::Min(63, $t.Length))
  $trayState.Text = $t
  $trayStop.Enabled = [bool]$id -and -not $busy
  $trayStart.Enabled = -not $busy
}
function Update-TrayMenu {
  $r = $null
  if (Test-Path $lastFile) { try { $r = Find-Row ((Get-Content -Encoding UTF8 $lastFile -Raw).Trim()) } catch {} }
  $trayStart.Text = if ($r) { "Start $($r.Tag.alias)" } else { "Start the default model" }
}
$script:trayHint = $false
$form.Add_Resize({
    if ($form.WindowState -eq "Minimized" -and $tray.Visible) {
      $form.Hide()
      if (-not $script:trayHint) { $script:trayHint = $true; $tray.ShowBalloonTip(3000, "inferzero", "The launcher is in the tray; the server keeps running.", "Info") }
    }
  })

# ---- the More menu's items
$script:speechItem = Add-Item $more "Start speech to text only" { Start-Selected -Speech }
[void](Add-Item $more "Copy the start command" { Copy-StartCommand })
Add-Sep $more
# the folders searched for models besides the Hugging Face cache (model-dirs.txt in the data folder, one a line, '#'
# comments; serverz reads it, tools/server_catalog.h): listed, with Add (a folder picker) and Remove; the model list
# read again when it changed (2026-11-23, #81: a D:\ai\models folder had no way in from the window)
function Edit-ModelDirs {
  $f = Join-Path $stateDir "model-dirs.txt"
  $lines = New-Object System.Collections.ArrayList
  if (Test-Path -LiteralPath $f) { foreach ($l in (Get-Content -Encoding UTF8 -LiteralPath $f)) { [void]$lines.Add($l.TrimStart([char]0xFEFF)) } }
  $dlg = New-Object System.Windows.Forms.Form
  $dlg.Text = "Model folders"; $dlg.StartPosition = "CenterParent"; $dlg.Width = 620; $dlg.Height = 340
  $dlg.FormBorderStyle = "FixedDialog"; $dlg.MaximizeBox = $false; $dlg.MinimizeBox = $false
  $lbl = New-Object System.Windows.Forms.Label
  $lbl.Text = "Searched for .gguf files (and Hugging Face cache folders), with the Hugging Face cache and $(Join-Path $stateDir 'models'):"
  $lbl.SetBounds(12, 10, 590, 34)
  $box = New-Object System.Windows.Forms.ListBox
  $box.SetBounds(12, 46, 480, 230)
  $fill = { $box.Items.Clear(); foreach ($l in $lines) { $t = $l.Trim(); if ($t -and -not $t.StartsWith("#")) { [void]$box.Items.Add($t) } } }
  & $fill
  $st = @{ changed = $false }  # (a table: the click handlers change it in place)
  $save = {
    New-Item -ItemType Directory -Force $stateDir | Out-Null
    [IO.File]::WriteAllText($f, ((@($lines) -join "`n") + "`n"), (New-Object Text.UTF8Encoding $false))
  }
  $addB = New-Object System.Windows.Forms.Button
  $addB.Text = "Add..."; $addB.SetBounds(504, 46, 90, 28)
  $addB.Add_Click({
      $fb = New-Object System.Windows.Forms.FolderBrowserDialog
      $fb.Description = "A folder with .gguf models"
      if ($fb.ShowDialog($dlg) -eq "OK" -and $fb.SelectedPath) {
        $p = $fb.SelectedPath
        if (@($box.Items | Where-Object { $_.TrimEnd('\').ToLowerInvariant() -eq $p.TrimEnd('\').ToLowerInvariant() }).Count -eq 0) {
          [void]$lines.Add($p); & $save; & $fill; $st.changed = $true
        }
      }
    })
  $remB = New-Object System.Windows.Forms.Button
  $remB.Text = "Remove"; $remB.SetBounds(504, 80, 90, 28)
  $remB.Add_Click({
      if ($null -eq $box.SelectedItem) { return }
      $s = [string]$box.SelectedItem
      foreach ($l in @($lines)) { if ($l.Trim() -eq $s) { $lines.Remove($l) } }
      & $save; & $fill; $st.changed = $true
    })
  $okB = New-Object System.Windows.Forms.Button
  $okB.Text = "Close"; $okB.SetBounds(504, 248, 90, 28); $okB.DialogResult = "OK"
  $dlg.Controls.AddRange(@($lbl, $box, $addB, $remB, $okB))
  $dlg.AcceptButton = $okB
  [void]$dlg.ShowDialog($form)
  $dlg.Dispose()
  if ($st.changed) { Start-List }
}
[void](Add-Item $more "Model folders..." { Edit-ModelDirs })
[void](Add-Item $more "Set as default (model and options)" { Set-AsDefault })
[void](Add-Item $more "Reset options" { Reset-Options })
[void](Add-Item $more "Refresh the model list" { Start-List })
[void](Add-Item $more "Pack the selected MoE model (inferzero only; the original can then go)..." { Start-Pack })
Add-Sep $more
if ($hasRec) {
  [void](Add-Item $more "Record a meeting..." { Start-Record })
  [void](Add-Item $more "Stop recording (the server transcribes it)" {
      $msg.Text = "Ending the recording (the server transcribes it: the Meetings tab)..."
      Start-Job2 @("record-stop") { param($code, $text) if ($code -ne 0) { Show-Bar "Stopping the recording failed (exit code $code)." "error" -Details $text } else { $msg.Text = ($text -split "`r?`n" | Where-Object { $_ }) -join "  " } }
    })
  Add-Sep $more
}
[void](Add-Item $more "Open the log in Notepad" { if (Test-Path $serverLog) { Start-Process notepad.exe -ArgumentList "`"$serverLog`"" } })
[void](Add-Item $more "Open the data folder" { Start-Process explorer.exe -ArgumentList "`"$stateDir`"" })
[void](Add-Item $more "Open the presets folder" { New-Item -ItemType Directory -Force $presetDir | Out-Null; Start-Process explorer.exe -ArgumentList "`"$presetDir`"" })
Add-Sep $more
[void](Add-Item $more "Check this PC (doctor report)" { Run-Doctor })
[void](Add-Item $more "Benchmark this GPU (matrix, vector, memory)..." { Start-GpuBench })
[void](Add-Item $more "Check for updates now" { Check-Update -Manual })
$updItem = Add-Item $more "Check for updates when the window opens" { $script:upd.enabled = -not $script:upd.enabled; $updItem.Checked = [bool]$script:upd.enabled; Save-Upd }
$updItem.Checked = [bool]$script:upd.enabled

$startBtn.Add_Click({ Start-Selected })
$webBtn.Add_Click({ Start-Process "$(Base)/" })
$grid.Add_CellDoubleClick({ param($s, $e) if ($e.RowIndex -ge 0) { Start-Selected } })
$grid.Add_SelectionChanged({ if ($script:listing) { return }; Apply-Auto; Refresh-Model; Show-Detail; Fill-Defaults; $startBtn.Enabled = -not ($script:actProc -and -not $script:actProc.HasExited) -and $grid.SelectedRows.Count -gt 0 })
$stopBtn.Add_Click({
    if ([System.Windows.Forms.MessageBox]::Show("Stop the server ($($script:running))? Its GPU memory is freed.", "inferzero", "YesNo") -eq "Yes") {
      $msg.Text = "Stopping the server..."
      Start-Action @("stop")
    }
  })

# ---- Windows 11 styling: a dark or light title bar as the window, rounded corners, Mica behind the title bar
# (DwmSetWindowAttribute; older Windows ignore what they do not know), dark scroll bars on a dark window
function Set-WindowStyle($f) {
  $f.Add_HandleCreated({
      param($s)
      $h = $s.Handle
      [void][ArcWin]::Dwm($h, 20, [int]$dark)  # DWMWA_USE_IMMERSIVE_DARK_MODE
      [void][ArcWin]::Dwm($h, 33, 2)           # DWMWA_WINDOW_CORNER_PREFERENCE = round
      [void][ArcWin]::Dwm($h, 38, 2)           # DWMWA_SYSTEMBACKDROP_TYPE = Mica (Windows 11 22H2+)
    })
  if ($dark) { $f.Add_Shown({ param($s) Set-DarkScroll $s }) }
}
# (text boxes, the grid and scrolling panels only: number fields and combo boxes themed this way drew red borders and
#  ended in "Visual Style handle creation operation did not succeed")
function Set-DarkScroll($ctl) {
  foreach ($c in $ctl.Controls) {
    if ($c -is [System.Windows.Forms.UpDownBase] -or $c -is [System.Windows.Forms.ComboBox]) { continue }
    if (($c -is [System.Windows.Forms.TextBox] -and $c.Multiline) -or $c -is [System.Windows.Forms.DataGridView] -or
      ($c -is [System.Windows.Forms.ScrollableControl] -and $c.AutoScroll)) {
      try { [void][ArcWin]::SetWindowTheme($c.Handle, "DarkMode_Explorer", $null) } catch {}
    }
    if ($c.Controls.Count -and $c -isnot [System.Windows.Forms.DataGridView]) { Set-DarkScroll $c }
  }
}
Set-WindowStyle $form
# the plain parts take the window's colours (labels and check boxes take their panel's)
foreach ($p in @($header, $buttons, $split, $split.Panel1, $split.Panel2, $bottom)) { $p.BackColor = $Pal.bg }
$tabStrip.BackColor = $Pal.card; $tabBody.BackColor = $Pal.card; $opt.BackColor = $Pal.card; $logTools.BackColor = $Pal.card
$memLbl.ForeColor = $Pal.sub

# ---- timer: collect background results; the server state every second
# the bar from the server's last "progress:" line, once zero-server has started the new server (it writes server.json
# right after: a switch first stops the running server, whose log still holds its own load and gets its shutdown
# lines, so the log's own time says nothing); the step in the message line under the list
function Show-LoadBar {
  $line = $null
  try {
    $sf = Get-Item $stateFile -ErrorAction Stop
    if ($sf.LastWriteTime -ge $script:actStart) {
      $line = Get-Content -Encoding UTF8 $serverLog -Tail 40 -ErrorAction SilentlyContinue | Where-Object { $_ -match '^progress: ' } | Select-Object -Last 1
    }
  } catch {}
  $loadBar.Visible = $true
  $secs = [int]((Get-Date) - $script:actStart).TotalSeconds
  if ($line -match '^progress: (-?\d+) % (.*)$') {
    $pct = [int]$Matches[1]
    if ($pct -ge 0) {
      $loadBar.Style = "Continuous"; $loadBar.Value = [math]::Min(100, $pct)
      $msg.Text = "Loading: $($Matches[2])  ($pct %, $secs s)"
    } else {
      $loadBar.Style = "Marquee"
      $msg.Text = "Loading: $($Matches[2])  ($secs s)"
    }
  } else {
    $loadBar.Style = "Marquee"
  }
}
# what zero-server's exit code means (scripts\zero-server.ps1: 1 wrong option, 2 stopped while loading, 3 timeout,
# 4 port in use)
function Show-ActionFailed([int]$code) {
  $err = Get-Content -Encoding UTF8 $script:actErr -Raw -ErrorAction SilentlyContinue
  $out = @(Get-Content -Encoding UTF8 $script:actOut -ErrorAction SilentlyContinue | Where-Object { $_ })
  $details = (($out | Select-Object -Last 25) -join "`r`n") + $(if ($err) { "`r`n$err" } else { "" })
  $first = @("$err" -split "`r?`n" | Where-Object { $_ -and $_ -notmatch '^\s*(At |\+|CategoryInfo|FullyQualifiedErrorId)' })[0]
  # (exit 2 after a GPU hang or a lost GPU too, or a start the hang marker refused: then a restart helps, not a smaller
  #  context; the marker or the log's lines say which, audit 2026-11-23 M8)
  $gpuFault = (Test-Path -LiteralPath (Join-Path $stateDir "gpu_hang.txt")) -or
              (@($out | Where-Object { $_ -match 'GPU (hang|lost)|a GPU (hang|lost) at|restart the computer' }).Count -gt 0)
  $what = switch ($code) {
    2 { $(if ($gpuFault) { "The GPU hung or was lost: restart the computer (Restart, not Shut down) before starting the server again (the log tab says which kernel)." }
          else { "The server stopped while loading (often: not enough GPU memory; a smaller context, Images off, or close other GPU programs)." }) }
    3 { "The server did not become ready within 10 minutes (the log tab says how far it got)." }
    4 { $(if ($out.Count) { $out[-1] } else { "The port is used by another program: pick another port (advanced options)." }) }
    default { $(if ($first) { "$first".Trim() } else { "Failed (exit code $code)." }) }
  }
  Show-Bar $what "error" -Details $details -ActionText "Show log" -Action { Select-Tab "Log" }
  if ($form.Visible -eq $false -and $tray.Visible) { $tray.ShowBalloonTip(5000, "inferzero", $what, "Error") }
}
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 500
$script:tick = 0
$script:lastTickError = ""
$timer.Add_Tick({
    try {
      if ($script:listProc -and $script:listProc.HasExited) {
        $script:listProc = $null
        try {
          $script:models = @(Get-Content -Encoding UTF8 $script:listOut -Raw | ConvertFrom-Json | ForEach-Object { $_ })
          Show-List
        } catch { Show-Bar "Could not read the model list: $($_.Exception.Message)" "error" -Details (Get-Content -Encoding UTF8 (Join-Path $env:TEMP "zero-launcher-models.err") -Raw -ErrorAction SilentlyContinue) }
      }
      if ($script:actProc) {
        $last = if (Test-Path $script:actOut) { Get-Content -Encoding UTF8 $script:actOut -Tail 1 -ErrorAction SilentlyContinue } else { $null }
        if ($last) { $msg.Text = "$last" }
        if ($script:actWhat -eq "start" -and -not $script:actProc.HasExited) { Show-LoadBar }
        if ($script:actProc.HasExited) {
          $code = $script:actProc.ExitCode
          $what = $script:actWhat
          $script:actProc = $null
          $loadBar.Visible = $false
          $script:sync.poke = $true
          if ($code -ne 0) { Show-ActionFailed $code }
          elseif ($what -eq "start") {
            if ($form.Visible -eq $false -and $tray.Visible) { $tray.ShowBalloonTip(3000, "inferzero", "Ready: $last", "Info") }
            if ($openWebChk.Checked -or $script:openWeb) {
              $script:openWeb = $false
              Start-Process "$(Base)/"  # (the web UI, in the default browser)
            }
          }
          Update-Status
        }
      }
      Check-Jobs
      if (($script:tick++ % 2) -eq 0) { Update-Status }
      if (($script:tick % 4) -eq 0) { Update-Log }
    } catch {
      # (an error in the timer is shown once in the bar, not as a box every half second: GitHub #5)
      $e = "$($_.Exception.Message)"
      if ($e -ne $script:lastTickError) { $script:lastTickError = $e; Show-Bar "Launcher error: $e" "error" -Details "$($_.ScriptStackTrace)" }
    }
  })

Set-Splash "Checking the package files..." 85
. (Join-Path $PSScriptRoot "launcher-firstrun.ps1")
. (Join-Path $PSScriptRoot "launcher-downloads.ps1")  # (Download models: several at once, a Hugging Face link, pause)
Load-PresetList
Update-TrayMenu
$form.Add_Shown({
    $form.Activate()
    Update-Status
    if (-not $SelfTest -and -not (Test-GpuDriver)) { Show-Bar "No Intel Arc GPU found through its driver: install Intel's graphics driver, restart, open this again." "error" }
    Start-List
    $timer.Start()
    if (-not $SelfTest) { Check-Update }
    if (-not $SelfTest) { try { Test-OldInstall } catch {} }  # (after an update into a new folder: the shortcut / PATH)
  })
# -SelfTest <png>: once the list is in (at most 3 minutes), a picture of the window, the errors seen, and close
if ($SelfTest) {
  $script:selfT0 = Get-Date
  $selfTimer = New-Object System.Windows.Forms.Timer; $selfTimer.Interval = 1000
  $selfTimer.Add_Tick({
      $ready = $grid.Rows.Count -gt 0 -and -not $script:listProc
      if (-not $ready -and ((Get-Date) - $script:selfT0).TotalSeconds -lt 180) { return }
      $selfTimer.Stop()
      if ($SelfTest -match 'log') { Select-Tab "Log" }
      # ($env:ARC_SELFTEST_PRESET: load that preset and print what Start would pass, and the memory estimate)
      if ($env:ARC_SELFTEST_PRESET) {
        $i = $presetBox.Items.IndexOf($env:ARC_SELFTEST_PRESET)
        if ($i -gt 0) { $presetBox.SelectedIndex = $i }
        [Console]::Out.WriteLine("selftest preset: " + ((Get-StartArgs) -join " ") + "  |  fit: " + $fitLbl.Text + "  |  " + $msg.Text)
      }
      # ($env:ARC_SELFTEST_MODEL = an id: that model selected (its row's Drafting and Images); ARC_SELFTEST_DRAFT =
      #  off / mtp / both / dflash and ARC_SELFTEST_IMAGES = on / off: those set; ARC_SELFTEST_OPTS =
      #  "copy_draft=on,xcpu=off": those switches set (on / off / auto). Then each switch's state and reason, the
      #  changes and what Start would pass, printed)
      if ($env:ARC_SELFTEST_MODEL -or $env:ARC_SELFTEST_OPTS -or $env:ARC_SELFTEST_DRAFT -or $env:ARC_SELFTEST_IMAGES) {
        if ($env:ARC_SELFTEST_MODEL) { $r = Find-Row $env:ARC_SELFTEST_MODEL; if ($r) { $grid.ClearSelection(); $r.Selected = $true } }
        if ($env:ARC_SELFTEST_DRAFT) { Set-GroupChoice "Drafting" $env:ARC_SELFTEST_DRAFT }
        if ($env:ARC_SELFTEST_IMAGES) { $imgChk.Checked = $env:ARC_SELFTEST_IMAGES -eq "on" }
        foreach ($kv in @("$env:ARC_SELFTEST_OPTS" -split ',' | Where-Object { $_ })) {
          $id, $v = $kv -split '=', 2
          if ($script:swRows.Contains($id)) { $script:swRows[$id].box.CheckState = $(switch ($v) { "on" { "Checked" } "off" { "Unchecked" } default { "Indeterminate" } }) }
          elseif ($id -in "Attention", "Multi-GPU") { Set-GroupChoice $id $v }
        }
        $sw = @(foreach ($id in $script:swRows.Keys) { $r = $script:swRows[$id]; "$id=$($r.box.CheckState)" + $(if ($r.why.Text) { "($($r.why.Text))" } else { "" }) })
        [Console]::Out.WriteLine("selftest model: " + $modelHead.Text + "; row " + (Sel-Entry).alias + "; drafting " + (Sel-Group).draft + "; images " + $imgChk.Checked + "; " + $draftWhy.Text + $imgWhy.Text)
        [Console]::Out.WriteLine("selftest switches: pc " + ($script:pcKinds -join ",") + "; " + ($sw -join " ") + "; attention: " + $attnWhy.Text + "; multi-gpu: " + $mgWhy.Text)
        [Console]::Out.WriteLine("selftest changed: " + $chgLbl.Text + "  |  fit: " + $fitLbl.Text)
        [Console]::Out.WriteLine("selftest start: " + ((Get-StartArgs) -join " "))
      }
      $form.Refresh(); [System.Windows.Forms.Application]::DoEvents()
      $bmp = New-Object System.Drawing.Bitmap($form.Width, $form.Height)
      $form.DrawToBitmap($bmp, (New-Object System.Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
      $bmp.Save($SelfTest, [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
      # (the -f in its own parentheses: inside a method call its commas split the arguments, GitHub #5)
      [Console]::Out.WriteLine(("selftest: rows {0}, status '{1}', message '{2}', bar '{3}', errors: {4}" -f $grid.Rows.Count, $status.Text, $msg.Text, $(if ($bar.Visible) { $barText.Text } else { "" }), ($script:selfErrors -join " | ")))
      $form.Close()
    })
  $selfTimer.Start()
}
$form.Add_FormClosing({
    $timer.Stop(); $script:sync.quit = $true  # (a running server and a start in progress keep going)
    $tray.Visible = $false; $tray.Dispose()
  })
# (the splash closed first, not in Shown: a ShowDialog took the active window, the splash, as its owner, and closing
#  an owner closes the window it owns: the launcher ended at once)
# (Application.Run, not ShowDialog: hiding a form shown with ShowDialog ends its modal loop, so minimizing to the tray
#  (the Resize handler's Hide) ended the launcher and took the tray icon with it (PascalSoftwares, issue #33,
#  2026-11-16; Windows PowerShell 5.1: the loop returned 0.02 s after the hide, and keeps running with Run))
Set-Splash "Opening..." 100
Close-Splash
[System.Windows.Forms.Application]::Run($form)
try { $pollPs.Dispose(); $pollRs.Dispose() } catch {}
