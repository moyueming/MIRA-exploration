Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Outputs = Join-Path $Root "outputs"
$FinalDir = Join-Path $Outputs "final_results"
$MaxEpisode = 1000
[System.IO.Directory]::CreateDirectory($FinalDir) | Out-Null

$Culture = [System.Globalization.CultureInfo]::InvariantCulture

function P($relative) {
    return Join-Path $Root $relative
}

$Methods = @(
    @{
        Label = "MIRA"; Color = [System.Drawing.Color]::FromArgb(31,119,180)
        RewardFiles = @(1,2,3 | ForEach-Object { P "outputs\our MIRA\galaxy_bile_fixed_seed$_`_final\galaxy_bile_fixed_seed$_`_final_fusion_rewards.csv" })
        TraceFiles = @(1,2,3 | ForEach-Object { P "outputs\our MIRA\galaxy_bile_fixed_seed$_`_final\galaxy_bile_fixed_seed$_`_final_exploration_trace.csv" })
    },
    @{
        Label = "MIRA-noEXT"; Color = [System.Drawing.Color]::FromArgb(255,127,14)
        RewardFiles = @(1,2,3 | ForEach-Object { P "outputs\our MIRA-no extrinsic\galaxy_bile_no_ext_fixed_seed$_`_final\galaxy_bile_no_ext_fixed_seed$_`_final_fusion_rewards.csv" })
        TraceFiles = @(1,2,3 | ForEach-Object { P "outputs\our MIRA-no extrinsic\galaxy_bile_no_ext_fixed_seed$_`_final\galaxy_bile_no_ext_fixed_seed$_`_final_exploration_trace.csv" })
    },
    @{
        Label = "DORA"; Color = [System.Drawing.Color]::FromArgb(44,160,44)
        RewardFiles = @(
            $(P "outputs\DORA\paper_a3c_seed_1_paper_a3c_rewards.csv"),
            $(P "outputs\DORA\paper_a3c_seed_2_paper_a3c_rewards.csv"),
            $(P "outputs\DORA\galaxy_paper_a3c_seed3_paper_a3c_rewards.csv")
        )
        TraceFiles = @(
            $(P "outputs\DORA\paper_a3c_seed_1_exploration_trace.csv"),
            $(P "outputs\DORA\paper_a3c_seed_2_exploration_trace.csv"),
            $(P "outputs\DORA\galaxy_paper_a3c_seed3_exploration_trace.csv")
        )
    },
    @{
        Label = "ATENA-ext"; Color = [System.Drawing.Color]::FromArgb(214,39,40)
        RewardFiles = @(
            $(P "outputs\ATENA_ext\atena_ext_seed_1_fusion_rewards.csv"),
            $(P "outputs\ATENA_ext\galaxy_ATENA_ext_fixed_seed2_final_fusion_rewards.csv"),
            $(P "outputs\ATENA_ext\galaxy_ATENA_ext_fixed_seed3_final_fusion_rewards.csv")
        )
        TraceFiles = @(
            $(P "outputs\ATENA_ext\atena_ext_seed_1_exploration_trace.csv"),
            $null,
            $(P "outputs\ATENA_ext\galaxy_ATENA_ext_fixed_seed3_final_exploration_trace.csv")
        )
    },
    @{
        Label = "ATENA"; Color = [System.Drawing.Color]::FromArgb(148,103,189)
        RewardFiles = @(
            $(P "outputs\ATENA_pure\atena_pure_seed_1_fusion_rewards.csv"),
            $(P "outputs\ATENA_pure\atena_pure_seed_2_fusion_rewards.csv"),
            $(P "outputs\ATENA_pure\galaxy_ATENA_pure_fixed_seed3_final_fusion_rewards.csv")
        )
        TraceFiles = @(
            $(P "outputs\ATENA_pure\atena_pure_seed_1_exploration_trace.csv"),
            $(P "outputs\ATENA_pure\atena_pure_seed_2_exploration_trace.csv"),
            $(P "outputs\ATENA_pure\galaxy_ATENA_pure_fixed_seed3_final_exploration_trace.csv")
        )
    },
    @{
        Label = "A3Cpure"; Color = [System.Drawing.Color]::FromArgb(140,86,75)
        RewardFiles = @(1,2,3 | ForEach-Object { P "outputs\A3Cpure\galaxy_pure_a3c_w5_seed$_\galaxy_pure_a3c_w5_seed$_`_pure_a3c_rewards.csv" })
        TraceFiles = @(1,2,3 | ForEach-Object { P "outputs\A3Cpure\galaxy_pure_a3c_w5_seed$_\galaxy_pure_a3c_w5_seed$_`_pure_a3c_exploration_trace.csv" })
    },
    @{
        Label = "Random"; Color = [System.Drawing.Color]::FromArgb(127,127,127)
        RewardFiles = @(1,2,3 | ForEach-Object { P "outputs\random\baseline_random_seed_$_`_precomputed_random_rewards.csv" })
        TraceFiles = @(1,2,3 | ForEach-Object { P "outputs\random\baseline_random_seed_$_`_precomputed_random_exploration_trace.csv" })
    }
)

function Parse-Double($text) {
    $value = 0.0
    if ([double]::TryParse([string]$text, [System.Globalization.NumberStyles]::Float, $Culture, [ref]$value)) {
        return $value
    }
    return [double]::NaN
}

function Format-Number($value) {
    $abs = [math]::Abs($value)
    if ($abs -ge 1000000) { return ("{0:0.##}M" -f ($value / 1000000.0)) }
    if ($abs -ge 10000) { return ("{0:0.#}K" -f ($value / 1000.0)) }
    if ($abs -ge 1000) { return ("{0:0.##}K" -f ($value / 1000.0)) }
    if ($abs -ge 100) { return ("{0:0}" -f $value) }
    if ($abs -ge 10) { return ("{0:0.#}" -f $value) }
    return ("{0:0.###}" -f $value)
}

function Get-MeanMetric($files, $metric) {
    $sums = New-Object double[] ($MaxEpisode + 1)
    $counts = New-Object int[] ($MaxEpisode + 1)
    foreach ($file in $files) {
        if ([string]::IsNullOrWhiteSpace($file) -or -not (Test-Path -LiteralPath $file)) { continue }
        $rows = Import-Csv -LiteralPath $file
        foreach ($row in $rows) {
            if (-not ($row.PSObject.Properties.Name -contains "episode")) { continue }
            if (-not ($row.PSObject.Properties.Name -contains $metric)) { continue }
            $episode = [int](Parse-Double $row.episode)
            if ($episode -lt 1 -or $episode -gt $MaxEpisode) { continue }
            $value = Parse-Double $row.$metric
            if ([double]::IsNaN($value)) { continue }
            $sums[$episode] += $value
            $counts[$episode] += 1
        }
    }
    $values = New-Object double[] ($MaxEpisode + 1)
    $hasAny = $false
    for ($i = 1; $i -le $MaxEpisode; $i++) {
        if ($counts[$i] -gt 0) {
            $values[$i] = $sums[$i] / [double]$counts[$i]
            $hasAny = $true
        } else {
            $values[$i] = [double]::NaN
        }
    }
    return @{ Values = $values; HasAny = $hasAny }
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

function Draw-MetricPlot($metric, $ylabel, $title, $filename) {
    $width = 2400
    $height = 1500
    $left = 260
    $right = 90
    $top = 280
    $bottom = 190
    $plotW = $width - $left - $right
    $plotH = $height - $top - $bottom
    $canvas = New-Canvas $width $height
    $bmp = $canvas.Bitmap
    $g = $canvas.Graphics

    $fontTitle = New-Object System.Drawing.Font "Times New Roman", 48, ([System.Drawing.FontStyle]::Bold)
    $fontAxis = New-Object System.Drawing.Font "Times New Roman", 38, ([System.Drawing.FontStyle]::Regular)
    $fontTick = New-Object System.Drawing.Font "Times New Roman", 30, ([System.Drawing.FontStyle]::Regular)
    $fontLegend = New-Object System.Drawing.Font "Times New Roman", 28, ([System.Drawing.FontStyle]::Regular)
    $brushBlack = [System.Drawing.Brushes]::Black

    Draw-TextCentered $g $title $fontTitle $brushBlack 0 35 $width 70

    $allSeries = @()
    $yMax = 0.0
    foreach ($method in $Methods) {
        $series = Get-MeanMetric $method.RewardFiles $metric
        if (-not $series.HasAny) { continue }
        for ($i = 1; $i -le $MaxEpisode; $i++) {
            $v = $series.Values[$i]
            if (-not [double]::IsNaN($v) -and $v -gt $yMax) { $yMax = $v }
        }
        $allSeries += @{ Method = $method; Values = $series.Values }
    }
    if ($yMax -le 0) { $yMax = 1.0 }
    $yMax = $yMax * 1.08

    $axisPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::Black), 3
    $gridPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(210,210,210)), 2
    $gridPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash

    for ($tick = 0; $tick -le 5; $tick++) {
        $value = $yMax * $tick / 5.0
        $y = $top + $plotH - ($value / $yMax) * $plotH
        $g.DrawLine($gridPen, $left, [float]$y, $left + $plotW, [float]$y)
        Draw-TextRight $g (Format-Number $value) $fontTick $brushBlack 0 ([float]($y - 25)) ($left - 25) 50
    }
    for ($tick = 0; $tick -le 5; $tick++) {
        $episode = [int]($MaxEpisode * $tick / 5)
        $x = $left + ($episode / [double]$MaxEpisode) * $plotW
        $g.DrawLine($gridPen, [float]$x, $top, [float]$x, $top + $plotH)
        Draw-TextCentered $g ([string]$episode) $fontTick $brushBlack ([float]($x - 70)) ($top + $plotH + 15) 140 50
    }

    $g.DrawLine($axisPen, $left, $top, $left, $top + $plotH)
    $g.DrawLine($axisPen, $left, $top + $plotH, $left + $plotW, $top + $plotH)

    foreach ($item in $allSeries) {
        $method = $item.Method
        $values = $item.Values
        $points = New-Object System.Collections.Generic.List[System.Drawing.PointF]
        for ($i = 1; $i -le $MaxEpisode; $i++) {
            $v = $values[$i]
            if ([double]::IsNaN($v)) { continue }
            $x = $left + (($i - 1) / [double]($MaxEpisode - 1)) * $plotW
            $y = $top + $plotH - ($v / $yMax) * $plotH
            $points.Add((New-Object System.Drawing.PointF ([float]$x), ([float]$y)))
        }
        if ($points.Count -gt 1) {
            $pen = New-Object System.Drawing.Pen $method.Color, 6
            $g.DrawLines($pen, $points.ToArray())
            $pen.Dispose()
        }
    }

    Draw-TextCentered $g "Episode" $fontAxis $brushBlack $left ($height - 95) $plotW 60
    $state = $g.Save()
    $g.TranslateTransform(70, $top + $plotH / 2)
    $g.RotateTransform(-90)
    Draw-TextCentered $g $ylabel $fontAxis $brushBlack -450 -35 900 70
    $g.Restore($state)

    $legendX = 270
    $legendY = 135
    $legendW = 280
    $legendH = 48
    for ($idx = 0; $idx -lt $Methods.Count; $idx++) {
        $row = [math]::Floor($idx / 4)
        $col = $idx % 4
        $x = $legendX + $col * 500
        $y = $legendY + $row * 60
        $pen = New-Object System.Drawing.Pen $Methods[$idx].Color, 8
        $g.DrawLine($pen, $x, $y + 24, $x + 75, $y + 24)
        $pen.Dispose()
        $g.DrawString($Methods[$idx].Label, $fontLegend, $brushBlack, [float]($x + 90), [float]$y)
    }

    $out = Join-Path $FinalDir $filename
    $bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose()
    $bmp.Dispose()
}

function Normalize-Operator($name) {
    $text = ([string]$name).ToLowerInvariant()
    if ($text.Contains("facet")) { return "Facet" }
    if ($text.Contains("neighbor")) { return "Neighbor" }
    if ($text.Contains("superset")) { return "Superset" }
    if ($text.Contains("distribution")) { return "Distribution" }
    return "Other"
}

function Get-MaxEpisodeFromTrace($file) {
    if ([string]::IsNullOrWhiteSpace($file) -or -not (Test-Path -LiteralPath $file)) { return 0 }
    $reader = [System.IO.File]::OpenText($file)
    try {
        $header = $reader.ReadLine()
        if ($null -eq $header) { return 0 }
        $cols = $header.Split(',')
        $epIndex = [Array]::IndexOf($cols, "episode")
        if ($epIndex -lt 0) { return 0 }
        $maxEp = 0
        while (($line = $reader.ReadLine()) -ne $null) {
            $parts = $line.Split(',')
            if ($parts.Length -le $epIndex) { continue }
            $ep = 0
            if ([int]::TryParse($parts[$epIndex], [ref]$ep) -and $ep -gt $maxEp) { $maxEp = $ep }
        }
        return $maxEp
    } finally {
        $reader.Close()
    }
}

function Get-TraceDistribution($file, $lastEpisodes) {
    $counts = @{}
    foreach ($cat in @("Facet","Neighbor","Superset","Distribution","Other")) { $counts[$cat] = 0.0 }
    if ([string]::IsNullOrWhiteSpace($file) -or -not (Test-Path -LiteralPath $file)) { return $counts }
    $maxEp = if ($null -eq $lastEpisodes) { 0 } else { Get-MaxEpisodeFromTrace $file }
    $minEp = if ($null -eq $lastEpisodes) { -1 } else { [math]::Max(1, $maxEp - [int]$lastEpisodes + 1) }
    $reader = [System.IO.File]::OpenText($file)
    try {
        $header = $reader.ReadLine()
        if ($null -eq $header) { return $counts }
        $cols = $header.Split(',')
        $epIndex = [Array]::IndexOf($cols, "episode")
        $opIndex = [Array]::IndexOf($cols, "operator")
        if ($opIndex -lt 0) { return $counts }
        while (($line = $reader.ReadLine()) -ne $null) {
            $parts = $line.Split(',')
            if ($parts.Length -le $opIndex) { continue }
            if ($null -ne $lastEpisodes -and $epIndex -ge 0) {
                $ep = 0
                if (-not [int]::TryParse($parts[$epIndex], [ref]$ep)) { continue }
                if ($ep -lt $minEp) { continue }
            }
            $cat = Normalize-Operator $parts[$opIndex]
            $counts[$cat] += 1.0
        }
    } finally {
        $reader.Close()
    }
    $total = ($counts.Values | Measure-Object -Sum).Sum
    if ($total -gt 0) {
        foreach ($cat in @($counts.Keys)) { $counts[$cat] = $counts[$cat] / [double]$total }
    }
    return $counts
}

function Get-MeanTraceDistribution($files, $lastEpisodes) {
    $cats = @("Facet","Neighbor","Superset","Distribution","Other")
    $sums = @{}
    foreach ($cat in $cats) { $sums[$cat] = 0.0 }
    $n = 0
    foreach ($file in $files) {
        if ([string]::IsNullOrWhiteSpace($file) -or -not (Test-Path -LiteralPath $file)) { continue }
        $dist = Get-TraceDistribution $file $lastEpisodes
        $has = (($dist.Values | Measure-Object -Sum).Sum -gt 0)
        if (-not $has) { continue }
        foreach ($cat in $cats) { $sums[$cat] += [double]$dist[$cat] }
        $n += 1
    }
    if ($n -gt 0) {
        foreach ($cat in $cats) { $sums[$cat] = $sums[$cat] / [double]$n }
    }
    return $sums
}

function Draw-OperatorPlot($lastEpisodes, $title, $filename) {
    $width = 2500
    $height = 1500
    $left = 210
    $right = 90
    $top = 310
    $bottom = 230
    $plotW = $width - $left - $right
    $plotH = $height - $top - $bottom
    $cats = @("Facet","Neighbor","Superset","Distribution","Other")
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
        $dist = Get-MeanTraceDistribution $method.TraceFiles $lastEpisodes
        $row = @()
        foreach ($cat in $cats) {
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

    $groupW = $plotW / [double]$cats.Count
    $barW = [math]::Min(48, $groupW / ($Methods.Count + 2))
    for ($c = 0; $c -lt $cats.Count; $c++) {
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
        Draw-TextCentered $g $cats[$c] $fontTick $brushBlack ([float]($center - $groupW / 2)) ($top + $plotH + 20) ([float]$groupW) 55
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

Draw-MetricPlot "extrinsic_reward" "Extrinsic Reward" "Galaxy Extrinsic Reward" "galaxy_extrinsic_reward.png"
Draw-MetricPlot "cumulative_extrinsic_reward" "Cumulative Extrinsic Reward" "Galaxy Cumulative Extrinsic Reward" "galaxy_cumulative_extrinsic_reward.png"
Draw-MetricPlot "cumulative_unique_sets_viewed" "Cumulative Unique Sets" "Galaxy Cumulative Unique Sets Viewed" "galaxy_cumulative_unique_sets_viewed.png"
Draw-MetricPlot "cumulative_target_efficiency" "Cumulative Target Efficiency" "Galaxy Cumulative Target Efficiency" "galaxy_cumulative_target_efficiency.png"
Draw-OperatorPlot $null "Galaxy Operator Distribution" "galaxy_operator_distribution_evolution.png"
Draw-OperatorPlot 200 "Galaxy Operator Distribution in Last 200 Episodes" "galaxy_operator_distribution_last200.png"

Get-ChildItem -LiteralPath $FinalDir -Filter "galaxy_*.png" | Select-Object FullName, Length, LastWriteTime

