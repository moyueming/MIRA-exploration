Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Outputs = Join-Path $Root "outputs"
$FinalDir = Join-Path $Outputs "final_results"
[System.IO.Directory]::CreateDirectory($FinalDir) | Out-Null

function P($relative) {
    return Join-Path $Root $relative
}

$Methods = @(
    @{
        Label = "MIRA"; Color = [System.Drawing.Color]::FromArgb(31,119,180)
        TraceFiles = @(1,2,3 | ForEach-Object { P "outputs\our MIRA\galaxy_bile_fixed_seed$_`_final\galaxy_bile_fixed_seed$_`_final_exploration_trace.csv" })
    },
    @{
        Label = "MIRA-noEXT"; Color = [System.Drawing.Color]::FromArgb(255,127,14)
        TraceFiles = @(1,2,3 | ForEach-Object { P "outputs\our MIRA-no extrinsic\galaxy_bile_no_ext_fixed_seed$_`_final\galaxy_bile_no_ext_fixed_seed$_`_final_exploration_trace.csv" })
    },
    @{
        Label = "DORA"; Color = [System.Drawing.Color]::FromArgb(44,160,44)
        TraceFiles = @(
            $(P "outputs\DORA\paper_a3c_seed_1_exploration_trace.csv"),
            $(P "outputs\DORA\paper_a3c_seed_2_exploration_trace.csv"),
            $(P "outputs\DORA\galaxy_paper_a3c_seed3_exploration_trace.csv")
        )
    },
    @{
        Label = "ATENA-ext"; Color = [System.Drawing.Color]::FromArgb(214,39,40)
        TraceFiles = @(
            $(P "outputs\ATENA_ext\atena_ext_seed_1_exploration_trace.csv"),
            $null,
            $(P "outputs\ATENA_ext\galaxy_ATENA_ext_fixed_seed3_final_exploration_trace.csv")
        )
    },
    @{
        Label = "ATENA"; Color = [System.Drawing.Color]::FromArgb(148,103,189)
        TraceFiles = @(
            $(P "outputs\ATENA_pure\atena_pure_seed_1_exploration_trace.csv"),
            $(P "outputs\ATENA_pure\atena_pure_seed_2_exploration_trace.csv"),
            $(P "outputs\ATENA_pure\galaxy_ATENA_pure_fixed_seed3_final_exploration_trace.csv")
        )
    },
    @{
        Label = "A3Cpure"; Color = [System.Drawing.Color]::FromArgb(140,86,75)
        TraceFiles = @(1,2,3 | ForEach-Object { P "outputs\A3Cpure\galaxy_pure_a3c_w5_seed$_\galaxy_pure_a3c_w5_seed$_`_pure_a3c_exploration_trace.csv" })
    },
    @{
        Label = "Random"; Color = [System.Drawing.Color]::FromArgb(127,127,127)
        TraceFiles = @(1,2,3 | ForEach-Object { P "outputs\random\baseline_random_seed_$_`_precomputed_random_exploration_trace.csv" })
    }
)

$Categories = @("Facet","Neighbor","Superset","Distribution","Other")

function Empty-Counts {
    $counts = @{}
    foreach ($cat in $Categories) { $counts[$cat] = 0.0 }
    return $counts
}

function Normalize-Operator($name) {
    $text = ([string]$name).ToLowerInvariant()
    if ($text.Contains("facet")) { return "Facet" }
    if ($text.Contains("neighbor")) { return "Neighbor" }
    if ($text.Contains("superset")) { return "Superset" }
    if ($text.Contains("distribution")) { return "Distribution" }
    return "Other"
}

function Normalize-Counts($counts) {
    $total = ($counts.Values | Measure-Object -Sum).Sum
    if ($total -gt 0) {
        foreach ($cat in @($counts.Keys)) { $counts[$cat] = $counts[$cat] / [double]$total }
    }
    return $counts
}

function Get-TraceStats($file) {
    $all = Empty-Counts
    $byEpisode = @{}
    if ([string]::IsNullOrWhiteSpace($file) -or -not (Test-Path -LiteralPath $file)) {
        return @{ All = $all; Last200 = Empty-Counts; HasData = $false }
    }
    $reader = [System.IO.File]::OpenText($file)
    $maxEp = 0
    try {
        $header = $reader.ReadLine()
        if ($null -eq $header) { return @{ All = $all; Last200 = Empty-Counts; HasData = $false } }
        $cols = $header.Split(',')
        $epIndex = [Array]::IndexOf($cols, "episode")
        $opIndex = [Array]::IndexOf($cols, "operator")
        if ($opIndex -lt 0) { return @{ All = $all; Last200 = Empty-Counts; HasData = $false } }
        while (($line = $reader.ReadLine()) -ne $null) {
            $parts = $line.Split(',')
            if ($parts.Length -le $opIndex) { continue }
            $cat = Normalize-Operator $parts[$opIndex]
            $all[$cat] += 1.0
            if ($epIndex -ge 0 -and $parts.Length -gt $epIndex) {
                $ep = 0
                if ([int]::TryParse($parts[$epIndex], [ref]$ep)) {
                    if ($ep -gt $maxEp) { $maxEp = $ep }
                    if (-not $byEpisode.ContainsKey($ep)) { $byEpisode[$ep] = Empty-Counts }
                    $byEpisode[$ep][$cat] += 1.0
                }
            }
        }
    } finally {
        $reader.Close()
    }
    $last = Empty-Counts
    $minEp = [math]::Max(1, $maxEp - 199)
    foreach ($ep in $byEpisode.Keys) {
        if ([int]$ep -lt $minEp) { continue }
        foreach ($cat in $Categories) { $last[$cat] += [double]$byEpisode[$ep][$cat] }
    }
    return @{ All = (Normalize-Counts $all); Last200 = (Normalize-Counts $last); HasData = (($all.Values | Measure-Object -Sum).Sum -gt 0) }
}

function Mean-Stats($files, $key) {
    $sum = Empty-Counts
    $n = 0
    foreach ($file in $files) {
        $stats = Get-TraceStats $file
        if (-not $stats.HasData) { continue }
        $dist = $stats[$key]
        foreach ($cat in $Categories) { $sum[$cat] += [double]$dist[$cat] }
        $n += 1
    }
    if ($n -gt 0) {
        foreach ($cat in $Categories) { $sum[$cat] = $sum[$cat] / [double]$n }
    }
    return $sum
}

function New-Canvas($width, $height) {
    $bmp = New-Object System.Drawing.Bitmap $width, $height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::White)
    return @{ Bitmap = $bmp; Graphics = $g }
}

function Draw-TextCentered($g, $text, $font, $brush, $x, $y, $w, $h) {
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::Center
    $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
    $g.DrawString($text, $font, $brush, (New-Object System.Drawing.RectangleF $x, $y, $w, $h), $sf)
    $sf.Dispose()
}

function Draw-TextRight($g, $text, $font, $brush, $x, $y, $w, $h) {
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = [System.Drawing.StringAlignment]::Far
    $sf.LineAlignment = [System.Drawing.StringAlignment]::Center
    $g.DrawString($text, $font, $brush, (New-Object System.Drawing.RectangleF $x, $y, $w, $h), $sf)
    $sf.Dispose()
}

function Draw-OperatorPlot($key, $title, $filename) {
    $width = 2500
    $height = 1500
    $left = 210
    $right = 90
    $top = 310
    $bottom = 230
    $plotW = $width - $left - $right
    $plotH = $height - $top - $bottom
    $canvas = New-Canvas $width $height
    $bmp = $canvas.Bitmap
    $g = $canvas.Graphics

    $fontTitle = New-Object System.Drawing.Font "Times New Roman", 48, ([System.Drawing.FontStyle]::Bold)
    $fontAxis = New-Object System.Drawing.Font "Times New Roman", 38
    $fontTick = New-Object System.Drawing.Font "Times New Roman", 30
    $fontLegend = New-Object System.Drawing.Font "Times New Roman", 27
    $brushBlack = [System.Drawing.Brushes]::Black
    Draw-TextCentered $g $title $fontTitle $brushBlack 0 35 $width 70

    $data = @()
    $maxValue = 0.0
    foreach ($method in $Methods) {
        $dist = Mean-Stats $method.TraceFiles $key
        $row = @()
        foreach ($cat in $Categories) {
            $v = [double]$dist[$cat]
            $row += $v
            if ($v -gt $maxValue) { $maxValue = $v }
        }
        $data += ,$row
    }
    $yMax = [math]::Max(0.5, [math]::Min(1.0, [math]::Ceiling(($maxValue + 0.08) * 10) / 10.0))
    $axisPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Black), 3
    $gridPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(210,210,210)), 2
    $gridPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
    for ($tick = 0; $tick -le 5; $tick++) {
        $value = $yMax * $tick / 5.0
        $y = $top + $plotH - ($value / $yMax) * $plotH
        $g.DrawLine($gridPen, $left, [float]$y, $left + $plotW, [float]$y)
        Draw-TextRight $g ("{0:P0}" -f $value) $fontTick $brushBlack 0 ([float]($y - 25)) ($left - 25) 50
    }
    $g.DrawLine($axisPen, $left, $top, $left, $top + $plotH)
    $g.DrawLine($axisPen, $left, $top + $plotH, $left + $plotW, $top + $plotH)
    $groupW = $plotW / [double]$Categories.Count
    $barW = [math]::Min(48, $groupW / ($Methods.Count + 2))
    for ($c = 0; $c -lt $Categories.Count; $c++) {
        $center = $left + ($c + 0.5) * $groupW
        for ($m = 0; $m -lt $Methods.Count; $m++) {
            $v = $data[$m][$c]
            $barH = ($v / $yMax) * $plotH
            $x = $center + ($m - (($Methods.Count - 1) / 2.0)) * $barW - ($barW * 0.42)
            $y = $top + $plotH - $barH
            $brush = New-Object System.Drawing.SolidBrush $Methods[$m].Color
            $g.FillRectangle($brush, [float]$x, [float]$y, [float]($barW * 0.84), [float]$barH)
            $brush.Dispose()
        }
        Draw-TextCentered $g $Categories[$c] $fontTick $brushBlack ([float]($center - $groupW / 2)) ($top + $plotH + 20) ([float]$groupW) 55
    }
    Draw-TextCentered $g "Operator Family" $fontAxis $brushBlack $left ($height - 100) $plotW 60
    $state = $g.Save()
    $g.TranslateTransform(65, $top + $plotH / 2)
    $g.RotateTransform(-90)
    Draw-TextCentered $g "Selection Ratio" $fontAxis $brushBlack -420 -35 840 70
    $g.Restore($state)
    $legendX = 250
    $legendY = 140
    for ($idx = 0; $idx -lt $Methods.Count; $idx++) {
        $row = [math]::Floor($idx / 4)
        $col = $idx % 4
        $x = $legendX + $col * 510
        $y = $legendY + $row * 60
        $brush = New-Object System.Drawing.SolidBrush $Methods[$idx].Color
        $g.FillRectangle($brush, $x, $y + 12, 56, 28)
        $brush.Dispose()
        $g.DrawString($Methods[$idx].Label, $fontLegend, $brushBlack, [float]($x + 72), [float]$y)
    }
    $out = Join-Path $FinalDir $filename
    $bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose()
    $bmp.Dispose()
}

Draw-OperatorPlot "All" "Galaxy Operator Distribution" "galaxy_operator_distribution_evolution.png"
Draw-OperatorPlot "Last200" "Galaxy Operator Distribution in Last 200 Episodes" "galaxy_operator_distribution_last200.png"
Get-ChildItem -LiteralPath $FinalDir -Filter "galaxy_operator*.png" | Select-Object FullName, Length, LastWriteTime
