"""
trend_analyzer.py — GazeStyle Multi-Session Trend Analyzer
===========================================================
Run this script AFTER exporting one or more CSV files from main.py:

    python trend_analyzer.py

It scans the current directory for all gaze_data_multiperson_*.csv files,
aggregates the data across sessions, computes trends, and produces
a self-contained  trend_report.html  file you can open in any browser.
"""

import os
import csv
import glob
import json
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_sessions(directory: str) -> list[dict]:
    """Load all exported CSVs and return a list of session dicts."""
    pattern = os.path.join(directory, "gaze_data_multiperson_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        return []

    sessions = []
    for path in files:
        fname = os.path.basename(path)
        try:
            dt_str  = fname.replace("gaze_data_multiperson_", "").replace(".csv", "")
            dt_obj  = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
            dt_label = dt_obj.strftime("%b %d, %Y  %H:%M")
        except Exception:
            dt_label = fname

        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "product":    row.get("Product", "").strip(),
                    "fixations":  int(float(row.get("Total Fixations (All Persons)", 0) or 0)),
                    "avg_gaze":   float(row.get("Avg Gaze Time (s)", 0) or 0),
                    "top_emotion": row.get("Top Emotion", "neutral").strip(),
                })

        sessions.append({"label": dt_label, "file": fname, "rows": rows})

    return sessions


# ─────────────────────────────────────────────────────────────
# AGGREGATION & TREND COMPUTATION
# ─────────────────────────────────────────────────────────────

def aggregate(sessions: list[dict]) -> dict:
    """Compute per-product totals, averages, and session-over-session trends."""

    # Collect all known products (preserve insertion order across sessions)
    all_products = list(dict.fromkeys(
        row["product"]
        for s in sessions
        for row in s["rows"]
        if row["product"]
    ))

    # Per-product per-session table
    # product -> [fixations_per_session]
    fix_by_session:  dict[str, list[int]]   = defaultdict(list)
    time_by_session: dict[str, list[float]] = defaultdict(list)
    emotions:        dict[str, dict]        = defaultdict(lambda: defaultdict(int))

    for s in sessions:
        present = {row["product"] for row in s["rows"]}
        for p in all_products:
            row_match = next((r for r in s["rows"] if r["product"] == p), None)
            fix_by_session[p].append(row_match["fixations"]  if row_match else 0)
            time_by_session[p].append(row_match["avg_gaze"]   if row_match else 0.0)
            if row_match and row_match["top_emotion"]:
                emotions[p][row_match["top_emotion"]] += 1

    # Build summary per product
    summary = []
    for p in all_products:
        fixes  = fix_by_session[p]
        times  = time_by_session[p]
        total_fix  = sum(fixes)
        avg_fix    = total_fix / len(fixes) if fixes else 0
        avg_time   = sum(times) / len(times) if times else 0

        # Trend: % change from first to last session that has data
        nonzero = [f for f in fixes if f > 0]
        if len(nonzero) >= 2:
            trend_pct = ((nonzero[-1] - nonzero[0]) / nonzero[0]) * 100
        elif len(fixes) >= 2 and fixes[-1] != fixes[-2]:
            trend_pct = ((fixes[-1] - fixes[-2]) / max(fixes[-2], 1)) * 100
        else:
            trend_pct = 0.0

        top_emotion = max(emotions[p], key=emotions[p].get) if emotions[p] else "neutral"

        summary.append({
            "product":      p,
            "total_fix":    total_fix,
            "avg_fix":      round(avg_fix, 1),
            "avg_time":     round(avg_time, 2),
            "trend_pct":    round(trend_pct, 1),
            "top_emotion":  top_emotion,
            "fix_series":   fixes,
            "time_series":  [round(t, 2) for t in times],
        })

    # Sort by total fixations descending
    summary.sort(key=lambda x: x["total_fix"], reverse=True)
    return {"products": summary, "session_labels": [s["label"] for s in sessions]}


# ─────────────────────────────────────────────────────────────
# HTML REPORT GENERATION
# ─────────────────────────────────────────────────────────────

EMOTION_COLORS = {
    "happy":    "#10b981",
    "neutral":  "#3b82f6",
    "surprise": "#f59e0b",
    "sad":      "#60a5fa",
    "angry":    "#f43f5e",
    "fear":     "#a78bfa",
    "disgust":  "#6b7280",
}

PRODUCT_COLORS = [
    "#7c3aed", "#10b981", "#f97316", "#ec4899",
    "#3b82f6", "#eab308", "#ef4444", "#a855f7",
    "#06b6d4", "#84cc16", "#f43f5e", "#8b5cf6",
]


def trend_arrow(pct: float) -> str:
    if pct > 5:
        return f'<span style="color:#10b981">▲ +{pct}%</span>'
    elif pct < -5:
        return f'<span style="color:#f43f5e">▼ {pct}%</span>'
    else:
        return f'<span style="color:#90b8d4">→ {pct}%</span>'


def emotion_badge(emotion: str) -> str:
    c = EMOTION_COLORS.get(emotion, "#8b89b0")
    return f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;font-size:.72rem;font-family:monospace;font-weight:600;background:{c}22;color:{c};border:1px solid {c}44">{emotion}</span>'


def generate_report(sessions: list[dict], agg: dict, out_path: str):
    products     = agg["products"]
    sess_labels  = agg["session_labels"]
    n_sessions   = len(sessions)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # JSON blobs for Chart.js
    chart_labels_json   = json.dumps(sess_labels)
    fix_datasets        = []
    time_datasets       = []
    for i, p in enumerate(products):
        c = PRODUCT_COLORS[i % len(PRODUCT_COLORS)]
        fix_datasets.append({
            "label": p["product"],
            "data":  p["fix_series"],
            "borderColor": c,
            "backgroundColor": c + "33",
            "fill": False,
            "tension": 0.4,
            "pointRadius": 5,
        })
        time_datasets.append({
            "label": p["product"],
            "data":  p["time_series"],
            "borderColor": c,
            "backgroundColor": c + "33",
            "fill": False,
            "tension": 0.4,
            "pointRadius": 5,
        })

    # Summary bar chart
    bar_labels = [p["product"] for p in products]
    bar_data   = [p["total_fix"] for p in products]
    bar_colors = [PRODUCT_COLORS[i % len(PRODUCT_COLORS)] for i in range(len(products))]

    # Table rows
    rows_html = ""
    for i, p in enumerate(products):
        rows_html += f"""
        <tr class="{'rank-1' if i == 0 else ''}">
          <td>{'🥇 ' if i==0 else ''}{p['product']}</td>
          <td><strong style="color:#00e5ff">{p['total_fix']}</strong></td>
          <td>{p['avg_fix']}</td>
          <td>{p['avg_time']}s</td>
          <td>{trend_arrow(p['trend_pct'])}</td>
          <td>{emotion_badge(p['top_emotion'])}</td>
        </tr>"""

    # Session table
    sess_rows_html = ""
    for s in sessions:
        total_fix = sum(r["fixations"] for r in s["rows"])
        sess_rows_html += f"""
        <tr>
          <td>{s['label']}</td>
          <td>{s['file']}</td>
          <td>{len(s['rows'])} products</td>
          <td>{total_fix}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>GazeStyle · Trend Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:#04080f; --bg-card:#070d16; --bg-card2:#0a1220;
      --border:#0e1e30; --border-glow:#0097a7;
      --cyan:#00e5ff; --cyan-dim:#0097a7; --cyan-ghost:rgba(0,229,255,.07);
      --purple:#7c3aed; --success:#10b981; --warning:#f59e0b; --danger:#f43f5e;
      --text:#e8f4fd; --text-dim:#90b8d4; --text-muted:#3d6278;
      --mono:'JetBrains Mono',monospace;
    }}
    body {{ background:var(--bg); color:var(--text); font-family:'Inter',sans-serif; min-height:100vh; overflow-x:hidden; }}
    body::before {{
      content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
      background-image: linear-gradient(rgba(0,229,255,.025) 1px,transparent 1px),
                        linear-gradient(90deg,rgba(0,229,255,.025) 1px,transparent 1px);
      background-size:55px 55px;
    }}
    .container {{ position:relative; z-index:1; max-width:1400px; margin:0 auto; padding:0 28px 80px; }}
    header {{ display:flex; align-items:center; justify-content:space-between; padding:28px 0 20px; border-bottom:1px solid var(--border-glow); }}
    .logo {{ display:flex; align-items:center; gap:14px; }}
    .logo-icon {{ font-size:2.2rem; filter:drop-shadow(0 0 8px var(--cyan)); }}
    .logo-text h1 {{ font-size:1.6rem; font-weight:800; letter-spacing:-.5px; }}
    .logo-text p  {{ font-size:.78rem; color:var(--text-dim); font-family:var(--mono); margin-top:2px; }}
    .meta {{ font-size:.73rem; font-family:var(--mono); color:var(--text-muted); text-align:right; }}
    .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin:28px 0; }}
    .stat-card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:20px 22px; position:relative; overflow:hidden; }}
    .stat-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--accent-line,var(--cyan-dim)); }}
    .stat-label {{ font-size:.72rem; font-family:var(--mono); color:var(--text-muted); text-transform:uppercase; letter-spacing:.08em; }}
    .stat-value {{ font-size:2rem; font-weight:800; margin-top:6px; line-height:1; }}
    .stat-icon  {{ position:absolute; right:16px; top:50%; transform:translateY(-50%); font-size:2rem; opacity:.1; }}
    .card {{ background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:22px 24px; margin-bottom:22px; }}
    .card-title {{ font-size:.78rem; font-family:var(--mono); color:var(--cyan-dim); text-transform:uppercase; letter-spacing:.1em; margin-bottom:18px; }}
    .charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:22px; }}
    @media(max-width:900px){{ .charts-row{{ grid-template-columns:1fr; }} }}
    .chart-wrap {{ position:relative; height:300px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th {{ background:var(--bg-card2); padding:12px 20px; text-align:left; font-size:.72rem; font-family:var(--mono); color:var(--text-muted); text-transform:uppercase; letter-spacing:.08em; border-bottom:1px solid var(--border); }}
    td {{ padding:14px 20px; font-size:.88rem; border-bottom:1px solid var(--border); color:var(--text-dim); }}
    tr:last-child td {{ border-bottom:none; }}
    tr:hover td {{ background:var(--cyan-ghost); color:var(--text); }}
    .rank-1 td:first-child {{ color:var(--cyan); font-weight:700; }}
    .footer {{ text-align:center; padding:40px 0 20px; font-size:.73rem; font-family:var(--mono); color:var(--text-muted); border-top:1px solid var(--border); margin-top:40px; }}
    ::-webkit-scrollbar {{ width:6px; }} ::-webkit-scrollbar-track {{ background:var(--bg); }} ::-webkit-scrollbar-thumb {{ background:var(--border-glow); border-radius:3px; }}
  </style>
</head>
<body>
<div class="container">

  <header>
    <div class="logo">
      <span class="logo-icon">📈</span>
      <div class="logo-text">
        <h1>GazeStyle · Trend Report</h1>
        <p>Multi-Session Retail Attention Analysis</p>
      </div>
    </div>
    <div class="meta">
      Generated: {generated_at}<br/>
      Sessions analysed: {n_sessions}
    </div>
  </header>

  <div class="stat-grid">
    <div class="stat-card" style="--accent-line:#7c3aed">
      <div class="stat-label">Sessions</div>
      <div class="stat-value" style="color:#9d68f5">{n_sessions}</div>
      <div class="stat-icon">📅</div>
    </div>
    <div class="stat-card" style="--accent-line:#10b981">
      <div class="stat-label">Products Tracked</div>
      <div class="stat-value" style="color:#10b981">{len(products)}</div>
      <div class="stat-icon">📦</div>
    </div>
    <div class="stat-card" style="--accent-line:#f59e0b">
      <div class="stat-label">All-Time Fixations</div>
      <div class="stat-value" style="color:#f59e0b">{sum(p['total_fix'] for p in products)}</div>
      <div class="stat-icon">👁</div>
    </div>
    <div class="stat-card" style="--accent-line:#00e5ff">
      <div class="stat-label">Top Product (Overall)</div>
      <div class="stat-value" style="font-size:1.1rem;color:#00e5ff">{ products[0]['product'] if products else '—' }</div>
      <div class="stat-icon">🏆</div>
    </div>
  </div>

  <!-- Bar: total fixations -->
  <div class="card">
    <div class="card-title">📊 Total Fixations per Product (all sessions)</div>
    <div class="chart-wrap"><canvas id="barChart"></canvas></div>
  </div>

  <!-- Line charts -->
  <div class="charts-row">
    <div class="card">
      <div class="card-title">📈 Fixations over Sessions</div>
      <div class="chart-wrap"><canvas id="fixLine"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">⏱ Avg Gaze Time over Sessions</div>
      <div class="chart-wrap"><canvas id="timeLine"></canvas></div>
    </div>
  </div>

  <!-- Summary table -->
  <div class="card">
    <div class="card-title">📋 Product Summary & Trends</div>
    <table>
      <thead><tr>
        <th>Product</th><th>Total Fix.</th><th>Avg Fix/Session</th>
        <th>Avg Gaze (s)</th><th>Trend</th><th>Top Emotion</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <!-- Session log -->
  <div class="card">
    <div class="card-title">🗓 Session Log</div>
    <table>
      <thead><tr><th>Date / Time</th><th>File</th><th>Products</th><th>Total Fixations</th></tr></thead>
      <tbody>{sess_rows_html}</tbody>
    </table>
  </div>

  <div class="footer">
    © 2026 GazeStyle &nbsp;·&nbsp; Powered by MediaPipe & DeepFace &nbsp;·&nbsp; trend_analyzer.py
  </div>
</div>

<script>
const sessLabels = {chart_labels_json};
const fixDS      = {json.dumps(fix_datasets)};
const timeDS     = {json.dumps(time_datasets)};
const barLabels  = {json.dumps(bar_labels)};
const barData    = {json.dumps(bar_data)};
const barColors  = {json.dumps(bar_colors)};

const gridColor = 'rgba(14,30,48,.9)';
const tickColor = '#90b8d4';

// Bar chart
new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: barLabels,
    datasets: [{{
      label: 'Total Fixations',
      data:  barData,
      backgroundColor: barColors.map(c => c + 'cc'),
      borderColor: barColors,
      borderWidth: 1, borderRadius: 6,
    }}]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ display:false }} }},
    scales:{{
      x:{{ ticks:{{color:tickColor,font:{{size:11}}}}, grid:{{color:gridColor}} }},
      y:{{ ticks:{{color:tickColor,font:{{size:11}}}}, grid:{{color:gridColor}} }}
    }}
  }}
}});

// Fixations line chart
new Chart(document.getElementById('fixLine'), {{
  type: 'line',
  data: {{ labels: sessLabels, datasets: fixDS }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ labels:{{color:tickColor,font:{{size:10}},padding:10}} }} }},
    scales:{{
      x:{{ ticks:{{color:tickColor,font:{{size:10}}}}, grid:{{color:gridColor}} }},
      y:{{ ticks:{{color:tickColor,font:{{size:10}}}}, grid:{{color:gridColor}} }}
    }}
  }}
}});

// Avg gaze time line chart
new Chart(document.getElementById('timeLine'), {{
  type: 'line',
  data: {{ labels: sessLabels, datasets: timeDS }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ labels:{{color:tickColor,font:{{size:10}},padding:10}} }} }},
    scales:{{
      x:{{ ticks:{{color:tickColor,font:{{size:10}}}}, grid:{{color:gridColor}} }},
      y:{{ ticks:{{color:tickColor,font:{{size:10}},suffix:'s'}}, grid:{{color:gridColor}} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    directory = os.path.dirname(os.path.abspath(__file__))
    out_path  = os.path.join(directory, "trend_report.html")

    print("=" * 60)
    print("  GazeStyle — Multi-Session Trend Analyzer")
    print(f"  Scanning: {directory}")
    print("=" * 60)

    sessions = load_sessions(directory)

    if not sessions:
        print("\n  ⚠  No CSV files found.")
        print("  Export at least one session from main.py first.")
        print("  (File pattern: gaze_data_multiperson_*.csv)")
        exit(1)

    print(f"\n  Found {len(sessions)} session(s):")
    for s in sessions:
        total = sum(r["fixations"] for r in s["rows"])
        print(f"    · {s['label']}  —  {total} total fixations  ({s['file']})")

    agg = aggregate(sessions)
    generate_report(sessions, agg, out_path)

    print(f"\n  ✅  Report generated: {out_path}")
    print("  Open trend_report.html in your browser to view.\n")
