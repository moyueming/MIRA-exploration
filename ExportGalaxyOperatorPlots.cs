using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Linq;

public static class ExportGalaxyOperatorPlots
{
    private static readonly string Root = Directory.GetCurrentDirectory();
    private static readonly string Outputs = Path.Combine(Root, "outputs");
    private static readonly string FinalDir = Path.Combine(Outputs, "final_results");
    private static readonly string[] Categories = { "Facet", "Neighbor", "Superset", "Distribution", "Other" };

    private sealed class MethodSpec
    {
        public string Label = "";
        public Color Color;
        public List<string?> TraceFiles = new List<string?>();
    }

    private sealed class TraceStats
    {
        public readonly Dictionary<string, double> All = EmptyCounts();
        public readonly Dictionary<string, double> Last200 = EmptyCounts();
        public bool HasData;
    }

    public static void Main()
    {
        Directory.CreateDirectory(FinalDir);
        var methods = BuildMethods();
        var cache = new Dictionary<string, TraceStats>(StringComparer.OrdinalIgnoreCase);
        var allData = BuildPlotData(methods, cache, useLast200: false);
        var lastData = BuildPlotData(methods, cache, useLast200: true);
        DrawOperatorPlot(methods, allData, "Galaxy Operator Distribution", "galaxy_operator_distribution_evolution.png");
        DrawOperatorPlot(methods, lastData, "Galaxy Operator Distribution in Last 200 Episodes", "galaxy_operator_distribution_last200.png");
        foreach (var file in Directory.GetFiles(FinalDir, "galaxy_operator*.png"))
        {
            var info = new FileInfo(file);
            Console.WriteLine($"{info.FullName}\t{info.Length}");
        }
    }

    private static List<MethodSpec> BuildMethods()
    {
        string P(string relative) => Path.Combine(Root, relative);
        return new List<MethodSpec>
        {
            new MethodSpec
            {
                Label = "MIRA",
                Color = Color.FromArgb(31,119,180),
                TraceFiles = new List<string?> { 1, 2, 3 }.Select(seed => P($@"outputs\our MIRA\galaxy_bile_fixed_seed{seed}_final\galaxy_bile_fixed_seed{seed}_final_exploration_trace.csv")).ToList()
            },
            new MethodSpec
            {
                Label = "MIRA-noEXT",
                Color = Color.FromArgb(255,127,14),
                TraceFiles = new List<string?> { 1, 2, 3 }.Select(seed => P($@"outputs\our MIRA-no extrinsic\galaxy_bile_no_ext_fixed_seed{seed}_final\galaxy_bile_no_ext_fixed_seed{seed}_final_exploration_trace.csv")).ToList()
            },
            new MethodSpec
            {
                Label = "DORA",
                Color = Color.FromArgb(44,160,44),
                TraceFiles = new List<string?>
                {
                    P(@"outputs\DORA\paper_a3c_seed_1_exploration_trace.csv"),
                    P(@"outputs\DORA\paper_a3c_seed_2_exploration_trace.csv"),
                    P(@"outputs\DORA\galaxy_paper_a3c_seed3_exploration_trace.csv")
                }
            },
            new MethodSpec
            {
                Label = "ATENA-ext",
                Color = Color.FromArgb(214,39,40),
                TraceFiles = new List<string?>
                {
                    P(@"outputs\ATENA_ext\atena_ext_seed_1_exploration_trace.csv"),
                    null,
                    P(@"outputs\ATENA_ext\galaxy_ATENA_ext_fixed_seed3_final_exploration_trace.csv")
                }
            },
            new MethodSpec
            {
                Label = "ATENA",
                Color = Color.FromArgb(148,103,189),
                TraceFiles = new List<string?>
                {
                    P(@"outputs\ATENA_pure\atena_pure_seed_1_exploration_trace.csv"),
                    P(@"outputs\ATENA_pure\atena_pure_seed_2_exploration_trace.csv"),
                    P(@"outputs\ATENA_pure\galaxy_ATENA_pure_fixed_seed3_final_exploration_trace.csv")
                }
            },
            new MethodSpec
            {
                Label = "A3Cpure",
                Color = Color.FromArgb(140,86,75),
                TraceFiles = new List<string?> { 1, 2, 3 }.Select(seed => P($@"outputs\A3Cpure\galaxy_pure_a3c_w5_seed{seed}\galaxy_pure_a3c_w5_seed{seed}_pure_a3c_exploration_trace.csv")).ToList()
            },
            new MethodSpec
            {
                Label = "Random",
                Color = Color.FromArgb(127,127,127),
                TraceFiles = new List<string?> { 1, 2, 3 }.Select(seed => P($@"outputs\random\baseline_random_seed_{seed}_precomputed_random_exploration_trace.csv")).ToList()
            }
        };
    }

    private static double[][] BuildPlotData(List<MethodSpec> methods, Dictionary<string, TraceStats> cache, bool useLast200)
    {
        var data = new double[methods.Count][];
        for (int methodIndex = 0; methodIndex < methods.Count; methodIndex++)
        {
            var sums = EmptyCounts();
            int count = 0;
            foreach (var file in methods[methodIndex].TraceFiles)
            {
                if (string.IsNullOrWhiteSpace(file) || !File.Exists(file)) continue;
                if (!cache.TryGetValue(file, out var stats))
                {
                    stats = ReadTraceStats(file);
                    cache[file] = stats;
                }
                if (!stats.HasData) continue;
                var source = useLast200 ? stats.Last200 : stats.All;
                foreach (var cat in Categories) sums[cat] += source[cat];
                count++;
            }
            data[methodIndex] = new double[Categories.Length];
            if (count > 0)
            {
                for (int i = 0; i < Categories.Length; i++) data[methodIndex][i] = sums[Categories[i]] / count;
            }
        }
        return data;
    }

    private static TraceStats ReadTraceStats(string file)
    {
        var stats = new TraceStats();
        var byEpisode = new Dictionary<int, Dictionary<string, double>>();
        int maxEpisode = 0;
        using (var reader = new StreamReader(file))
        {
            var header = reader.ReadLine();
            if (string.IsNullOrEmpty(header)) return stats;
            var cols = header.Split(',');
            int epIndex = Array.IndexOf(cols, "episode");
            int opIndex = Array.IndexOf(cols, "operator");
            if (opIndex < 0) return stats;
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                var parts = line.Split(',');
                if (parts.Length <= opIndex) continue;
                var cat = NormalizeOperator(parts[opIndex]);
                stats.All[cat] += 1.0;
                stats.HasData = true;
                if (epIndex >= 0 && parts.Length > epIndex && int.TryParse(parts[epIndex], NumberStyles.Integer, CultureInfo.InvariantCulture, out int ep))
                {
                    if (ep > maxEpisode) maxEpisode = ep;
                    if (!byEpisode.TryGetValue(ep, out var counts))
                    {
                        counts = EmptyCounts();
                        byEpisode[ep] = counts;
                    }
                    counts[cat] += 1.0;
                }
            }
        }
        NormalizeCounts(stats.All);
        int minEpisode = Math.Max(1, maxEpisode - 199);
        foreach (var pair in byEpisode)
        {
            if (pair.Key < minEpisode) continue;
            foreach (var cat in Categories) stats.Last200[cat] += pair.Value[cat];
        }
        NormalizeCounts(stats.Last200);
        return stats;
    }

    private static Dictionary<string, double> EmptyCounts()
    {
        return Categories.ToDictionary(cat => cat, _ => 0.0);
    }

    private static void NormalizeCounts(Dictionary<string, double> counts)
    {
        double total = counts.Values.Sum();
        if (total <= 0.0) return;
        foreach (var cat in Categories) counts[cat] /= total;
    }

    private static string NormalizeOperator(string name)
    {
        var lower = (name ?? "").ToLowerInvariant();
        if (lower.Contains("facet")) return "Facet";
        if (lower.Contains("neighbor")) return "Neighbor";
        if (lower.Contains("superset")) return "Superset";
        if (lower.Contains("distribution")) return "Distribution";
        return "Other";
    }

    private static void DrawOperatorPlot(List<MethodSpec> methods, double[][] data, string title, string filename)
    {
        const int width = 2500;
        const int height = 1500;
        const int left = 210;
        const int right = 90;
        const int top = 310;
        const int bottom = 230;
        int plotW = width - left - right;
        int plotH = height - top - bottom;
        using var bitmap = new Bitmap(width, height);
        using var g = Graphics.FromImage(bitmap);
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAliasGridFit;
        g.Clear(Color.White);
        using var titleFont = new Font("Times New Roman", 48, FontStyle.Bold);
        using var axisFont = new Font("Times New Roman", 38);
        using var tickFont = new Font("Times New Roman", 30);
        using var legendFont = new Font("Times New Roman", 27);
        using var blackBrush = new SolidBrush(Color.Black);

        DrawCentered(g, title, titleFont, blackBrush, 0, 35, width, 70);
        double maxValue = data.SelectMany(row => row).DefaultIfEmpty(0.0).Max();
        double yMax = Math.Max(0.5, Math.Min(1.0, Math.Ceiling((maxValue + 0.08) * 10.0) / 10.0));
        using var axisPen = new Pen(Color.Black, 3);
        using var gridPen = new Pen(Color.FromArgb(210, 210, 210), 2) { DashStyle = DashStyle.Dash };
        for (int tick = 0; tick <= 5; tick++)
        {
            double value = yMax * tick / 5.0;
            float y = (float)(top + plotH - (value / yMax) * plotH);
            g.DrawLine(gridPen, left, y, left + plotW, y);
            DrawRight(g, value.ToString("P0", CultureInfo.InvariantCulture), tickFont, blackBrush, 0, y - 25, left - 25, 50);
        }
        g.DrawLine(axisPen, left, top, left, top + plotH);
        g.DrawLine(axisPen, left, top + plotH, left + plotW, top + plotH);

        double groupW = plotW / (double)Categories.Length;
        double barW = Math.Min(48.0, groupW / (methods.Count + 2.0));
        for (int c = 0; c < Categories.Length; c++)
        {
            double center = left + (c + 0.5) * groupW;
            for (int m = 0; m < methods.Count; m++)
            {
                double v = data[m][c];
                double barH = (v / yMax) * plotH;
                double x = center + (m - ((methods.Count - 1) / 2.0)) * barW - (barW * 0.42);
                double y = top + plotH - barH;
                using var brush = new SolidBrush(methods[m].Color);
                g.FillRectangle(brush, (float)x, (float)y, (float)(barW * 0.84), (float)barH);
            }
            DrawCentered(g, Categories[c], tickFont, blackBrush, (float)(center - groupW / 2.0), top + plotH + 20, (float)groupW, 55);
        }

        DrawCentered(g, "Operator Family", axisFont, blackBrush, left, height - 100, plotW, 60);
        var state = g.Save();
        g.TranslateTransform(65, top + plotH / 2.0f);
        g.RotateTransform(-90);
        DrawCentered(g, "Selection Ratio", axisFont, blackBrush, -420, -35, 840, 70);
        g.Restore(state);

        int legendX = 250;
        int legendY = 140;
        for (int idx = 0; idx < methods.Count; idx++)
        {
            int row = idx / 4;
            int col = idx % 4;
            int x = legendX + col * 510;
            int y = legendY + row * 60;
            using var brush = new SolidBrush(methods[idx].Color);
            g.FillRectangle(brush, x, y + 12, 56, 28);
            g.DrawString(methods[idx].Label, legendFont, blackBrush, x + 72, y);
        }
        bitmap.Save(Path.Combine(FinalDir, filename), ImageFormat.Png);
    }

    private static void DrawCentered(Graphics g, string text, Font font, Brush brush, float x, float y, float w, float h)
    {
        using var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center };
        g.DrawString(text, font, brush, new RectangleF(x, y, w, h), sf);
    }

    private static void DrawRight(Graphics g, string text, Font font, Brush brush, float x, float y, float w, float h)
    {
        using var sf = new StringFormat { Alignment = StringAlignment.Far, LineAlignment = StringAlignment.Center };
        g.DrawString(text, font, brush, new RectangleF(x, y, w, h), sf);
    }
}
