"""
╔══════════════════════════════════════════════════════════════════╗
║   MESS INTELLIGENCE PLATFORM  —  v2.0                          ║
║   DWBI Analytics Suite for Institutional Food Systems           ║
║   Built for: IISER Thiruvananthapuram  |  Startup-Grade         ║
╚══════════════════════════════════════════════════════════════════╝

Deploy:  python app.py
Deps:    pip install gradio pandas numpy plotly mlxtend networkx scipy
GitHub:  Add requirements.txt + this file to repo root
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────
DATA: pd.DataFrame | None = None

# ─────────────────────────────────────────────────────────────────
# THEME PALETTE  (CSS injected via theme + custom_css)
# ─────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,17,26,0.6)",
    font=dict(family="'DM Mono', monospace", color="#E2E8F0"),
    title_font=dict(family="'Syne', sans-serif", size=16, color="#F8FAFC"),
    colorway=["#6EE7B7", "#38BDF8", "#F472B6", "#FBBF24", "#A78BFA", "#34D399"],
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
)

CUSTOM_CSS = """
/* ── Google Fonts ────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root Tokens ─────────────────────────────────────── */
:root {
    --bg-void:    #080B14;
    --bg-surface: #0F1220;
    --bg-card:    #161B2E;
    --bg-glass:   rgba(22,27,46,0.7);
    --border:     rgba(110,231,183,0.12);
    --accent-1:   #6EE7B7;
    --accent-2:   #38BDF8;
    --accent-3:   #F472B6;
    --text-hi:    #F8FAFC;
    --text-mid:   #94A3B8;
    --text-lo:    #475569;
    --radius:     12px;
    --glow:       0 0 32px rgba(110,231,183,0.15);
}

/* ── Body / App Shell ────────────────────────────────── */
body, .gradio-container {
    background: var(--bg-void) !important;
    font-family: 'DM Mono', monospace !important;
    color: var(--text-hi) !important;
}

/* Animated grid background */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(110,231,183,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(110,231,183,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Header Hero ─────────────────────────────────────── */
#header-hero {
    background: linear-gradient(135deg, #0D1525 0%, #111827 50%, #0A1628 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow), inset 0 1px 0 rgba(255,255,255,0.05);
}
#header-hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(110,231,183,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
#header-hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #6EE7B7, #38BDF8, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0 !important;
    letter-spacing: -0.5px;
}
#header-hero p {
    color: var(--text-mid) !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Tab Bar ─────────────────────────────────────────── */
.tabs > .tab-nav {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 6px !important;
    gap: 4px !important;
    margin-bottom: 1rem !important;
}
.tabs > .tab-nav button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-mid) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
}
.tabs > .tab-nav button:hover {
    color: var(--text-hi) !important;
    background: rgba(110,231,183,0.08) !important;
}
.tabs > .tab-nav button.selected {
    color: var(--bg-void) !important;
    background: var(--accent-1) !important;
    box-shadow: 0 0 16px rgba(110,231,183,0.35) !important;
}

/* ── Cards / Panels ──────────────────────────────────── */
.panel, .gr-group, .gr-box {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Buttons ─────────────────────────────────────────── */
button.primary, .gr-button-primary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    color: var(--bg-void) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 16px rgba(110,231,183,0.2) !important;
}
button.primary:hover { transform: translateY(-1px) !important; box-shadow: 0 0 24px rgba(110,231,183,0.4) !important; }

/* ── Inputs & Dropdowns ──────────────────────────────── */
input, select, textarea, .gr-input, .gr-dropdown {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-hi) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s ease !important;
}
input:focus, select:focus { border-color: var(--accent-1) !important; outline: none !important; box-shadow: 0 0 12px rgba(110,231,183,0.15) !important; }

/* ── KPI Cards ───────────────────────────────────────── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: var(--glow); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.kpi-card.k1::before { background: linear-gradient(90deg, var(--accent-1), var(--accent-2)); }
.kpi-card.k2::before { background: linear-gradient(90deg, var(--accent-2), var(--accent-3)); }
.kpi-card.k3::before { background: linear-gradient(90deg, var(--accent-3), #FBBF24); }
.kpi-card.k4::before { background: linear-gradient(90deg, #FBBF24, var(--accent-1)); }
.kpi-label { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-lo); margin-bottom: 0.5rem; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 700; color: var(--text-hi); }
.kpi-sub { font-size: 0.72rem; color: var(--text-mid); margin-top: 0.25rem; }

/* ── Section Labels ──────────────────────────────────── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent-1);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Status / Badge ──────────────────────────────────── */
.status-ok  { color: var(--accent-1) !important; }
.status-err { color: #F87171 !important; }

/* ── Dataframe ───────────────────────────────────────── */
.gr-dataframe table { font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important; }
.gr-dataframe thead th { background: var(--bg-surface) !important; color: var(--accent-1) !important; font-size: 0.7rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
.gr-dataframe tbody tr:hover td { background: rgba(110,231,183,0.04) !important; }

/* ── Sliders ─────────────────────────────────────────── */
.gr-slider input[type=range]::-webkit-slider-thumb { background: var(--accent-1) !important; }
.gr-slider input[type=range]::-webkit-slider-runnable-track { background: var(--border) !important; }

/* ── Markdown / Text ─────────────────────────────────── */
.gr-markdown, .prose { color: var(--text-mid) !important; font-size: 0.82rem !important; line-height: 1.6 !important; }
.gr-markdown strong { color: var(--text-hi) !important; }
.gr-markdown h3 { font-family: 'Syne', sans-serif !important; color: var(--text-hi) !important; font-size: 1rem !important; margin: 0.3rem 0 !important; }

/* ── Upload Zone ─────────────────────────────────────── */
.gr-file-upload {
    background: var(--bg-surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s ease !important;
}
.gr-file-upload:hover { border-color: var(--accent-1) !important; }

/* ── Alert Box ───────────────────────────────────────── */
.alert-box {
    background: rgba(110,231,183,0.06);
    border: 1px solid rgba(110,231,183,0.2);
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    font-size: 0.82rem;
    color: var(--accent-1);
}
.alert-box.warn { background: rgba(251,191,36,0.06); border-color: rgba(251,191,36,0.2); color: #FBBF24; }
.alert-box.danger { background: rgba(248,113,113,0.06); border-color: rgba(248,113,113,0.2); color: #F87171; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-1); }

/* ── Responsive ──────────────────────────────────────── */
@media (max-width: 768px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    #header-hero h1 { font-size: 1.5rem !important; }
}

/* ══════════════════════════════════════════════════════
   ASSOCIATION RULES — Google Material + Doodle Style
   ══════════════════════════════════════════════════════ */

#arm-banner {
    background: linear-gradient(135deg, #1a237e 0%, #283593 40%, #1565c0 100%);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(66,133,244,0.3);
}
#arm-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -30px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(251,188,5,0.3) 0%, transparent 70%);
}
#arm-banner::after {
    content: '';
    position: absolute;
    bottom: -50px; left: 20%;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(52,168,83,0.2) 0%, transparent 70%);
}
#arm-banner h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 0 0.3rem !important;
    letter-spacing: -0.3px;
    position: relative; z-index: 1;
}
#arm-banner p {
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.78rem !important;
    margin: 0 !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    position: relative; z-index: 1;
}

.arm-stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin: 1rem 0;
}
.arm-stat-card {
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.arm-stat-card:hover { transform: translateY(-3px); }
.arm-stat-card.blue  { background: linear-gradient(135deg, rgba(66,133,244,0.18), rgba(66,133,244,0.06)); border: 1.5px solid rgba(66,133,244,0.35); }
.arm-stat-card.green { background: linear-gradient(135deg, rgba(52,168,83,0.18), rgba(52,168,83,0.06));   border: 1.5px solid rgba(52,168,83,0.35); }
.arm-stat-card.amber { background: linear-gradient(135deg, rgba(251,188,5,0.18), rgba(251,188,5,0.06));   border: 1.5px solid rgba(251,188,5,0.35); }
.arm-stat-card .stat-icon  { font-size: 1.3rem; margin-bottom: 0.35rem; }
.arm-stat-card .stat-label { font-size: 0.67rem; letter-spacing: 0.1em; text-transform: uppercase; color: #64748B; margin-bottom: 0.3rem; }
.arm-stat-card .stat-val   { font-family: 'Syne', sans-serif; font-size: 1.55rem; font-weight: 700; }
.arm-stat-card.blue  .stat-val { color: #60A5FA; }
.arm-stat-card.green .stat-val { color: #4ADE80; }
.arm-stat-card.amber .stat-val { color: #FCD34D; }

#arm-mine-btn button {
    background: linear-gradient(135deg, #4285F4, #1a73e8) !important;
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 32px !important;
    box-shadow: 0 4px 20px rgba(66,133,244,0.45) !important;
    transition: all 0.2s ease !important;
    text-transform: none !important;
}
#arm-mine-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(66,133,244,0.6) !important;
}

#arm-table .gr-dataframe thead th {
    background: rgba(66,133,244,0.15) !important;
    color: #90CAF9 !important;
    font-size: 0.67rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid rgba(66,133,244,0.3) !important;
}
#arm-table .gr-dataframe tbody tr:hover td {
    background: rgba(66,133,244,0.07) !important;
}
"""

# ─────────────────────────────────────────────────────────────────
# HELPER — apply plotly theme
# ─────────────────────────────────────────────────────────────────
def style_fig(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        height=height,
        title=dict(text=title, font=dict(family="'Syne',sans-serif", size=15, color="#F8FAFC"), x=0.02),
        margin=dict(l=16, r=16, t=48, b=16),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_data(file):
    global DATA
    if file is None:
        return "⚠ No file selected.", None, None

    try:
        path = file.name if hasattr(file, "name") else file
        df = pd.read_csv(path)
    except Exception as e:
        return f"❌ Could not read CSV: {e}", None, None

    df.columns = [c.lower().strip() for c in df.columns]

    # ── flexible column detection ──────────────────────────────
    date_col   = next((c for c in df.columns if "date" in c), None)
    amount_col = next((c for c in df.columns if "amount" in c or "expense" in c), None)
    vendor_col = next((c for c in df.columns if "vendor" in c or "name" in c), None)
    mess_col   = next((c for c in df.columns if "mess" in c or "unit" in c), None)

    if date_col:   df["date"]   = pd.to_datetime(df[date_col],   errors="coerce")
    if amount_col: df["amount"] = pd.to_numeric(df[amount_col],  errors="coerce")
    if vendor_col: df["vendor"] = df[vendor_col].astype(str).str.strip().str.upper()
    if mess_col:   df["mess"]   = df[mess_col].astype(str).str.strip().str.upper()
    else:          df["mess"]   = "UNKNOWN"

    df = df.dropna(subset=["date", "amount"])
    df["amount"] = df["amount"].abs()
    DATA = df

    # ── summary stats ──────────────────────────────────────────
    total_txn     = len(df)
    total_spend   = df["amount"].sum()
    months_active = df["date"].dt.to_period("M").nunique()
    vendors_count = df["vendor"].nunique() if "vendor" in df.columns else 0

    msg = (
        f"✅ **{total_txn:,} transactions** loaded  |  "
        f"₹{total_spend:,.0f} total expenditure  |  "
        f"{months_active} months  |  "
        f"{vendors_count} unique vendors"
    )

    months  = ["All"] + sorted(df["date"].dt.month.dropna().unique().astype(int).astype(str).tolist())
    units   = ["All"] + sorted(df["mess"].dropna().unique().tolist())

    return msg, gr.update(choices=months, value="All"), gr.update(choices=units, value="All")


# ─────────────────────────────────────────────────────────────────
# 2. FILTER HELPER
# ─────────────────────────────────────────────────────────────────
def get_filtered(month: str, mess: str) -> pd.DataFrame:
    if DATA is None:
        return pd.DataFrame()
    df = DATA.copy()
    if month != "All":
        df = df[df["date"].dt.month == int(month)]
    if mess != "All":
        df = df[df["mess"] == mess]
    return df


# ─────────────────────────────────────────────────────────────────
# 3. KPI CARDS HTML
# ─────────────────────────────────────────────────────────────────
def build_kpis(month: str, mess: str) -> str:
    df = get_filtered(month, mess)
    if df.empty:
        return "<p style='color:#475569;font-size:0.85rem'>Load data to see KPIs.</p>"

    total   = df["amount"].sum()
    daily   = df.groupby(df["date"].dt.date)["amount"].sum()
    avg_day = daily.mean()
    peak    = daily.max()
    txn_cnt = len(df)

    top_vendor = "—"
    if "vendor" in df.columns:
        top_vendor = df.groupby("vendor")["amount"].sum().idxmax()
        top_vendor = top_vendor[:22] + "…" if len(top_vendor) > 22 else top_vendor

    # wastage estimate (100 kg baseline)
    monthly_avg = daily.mean()
    total_waste = (daily / monthly_avg * 100).sum()

    html = f"""
<div class="kpi-grid">
  <div class="kpi-card k1">
    <div class="kpi-label">Total Expenditure</div>
    <div class="kpi-value">₹{total/1e6:.2f}M</div>
    <div class="kpi-sub">{txn_cnt:,} transactions</div>
  </div>
  <div class="kpi-card k2">
    <div class="kpi-label">Avg Daily Spend</div>
    <div class="kpi-value">₹{avg_day:,.0f}</div>
    <div class="kpi-sub">Peak ₹{peak:,.0f}</div>
  </div>
  <div class="kpi-card k3">
    <div class="kpi-label">Est. Food Wastage</div>
    <div class="kpi-value">{total_waste:,.0f} kg</div>
    <div class="kpi-sub">Proxy model @ 100 kg baseline</div>
  </div>
  <div class="kpi-card k4">
    <div class="kpi-label">Top Vendor</div>
    <div class="kpi-value" style="font-size:1.1rem">{top_vendor}</div>
    <div class="kpi-sub">by total spend</div>
  </div>
</div>"""
    return html


# ─────────────────────────────────────────────────────────────────
# 4. TREND CHART
# ─────────────────────────────────────────────────────────────────
def build_trend(month: str, mess: str) -> go.Figure:
    df = get_filtered(month, mess)
    if df.empty:
        return go.Figure()

    daily = df.groupby(df["date"].dt.date)["amount"].sum().reset_index()
    daily.columns = ["date", "amount"]
    daily["date"] = pd.to_datetime(daily["date"])

    # monthly avg line for wastage threshold
    monthly_avg = daily["amount"].mean()

    # rolling 7-day average
    daily = daily.sort_values("date")
    daily["ma7"] = daily["amount"].rolling(7, min_periods=1).mean()

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["amount"],
        name="Daily Spend",
        fill="tozeroy",
        line=dict(color="#6EE7B7", width=2),
        fillcolor="rgba(110,231,183,0.08)",
        mode="lines+markers",
        marker=dict(size=4, color="#6EE7B7"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.0f}<extra></extra>"
    ))

    # 7-day MA
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ma7"],
        name="7-Day MA",
        line=dict(color="#38BDF8", width=2, dash="dot"),
        hovertemplate="7-Day MA: ₹%{y:,.0f}<extra></extra>"
    ))

    # Threshold line
    fig.add_hline(
        y=monthly_avg,
        line=dict(color="#F472B6", width=1.5, dash="dash"),
        annotation_text="Avg",
        annotation_font_color="#F472B6"
    )

    style_fig(fig, "Daily Expenditure Trend", height=380)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ─────────────────────────────────────────────────────────────────
# 5. MESS COMPARISON
# ─────────────────────────────────────────────────────────────────
def build_mess_comparison(month: str, _: str) -> go.Figure:
    df = get_filtered(month, "All")
    if df.empty or "mess" not in df.columns:
        return go.Figure()

    grp = df.groupby("mess")["amount"].agg(["sum", "mean", "count"]).reset_index()
    grp.columns = ["mess", "total", "avg_daily", "txn"]
    grp = grp[grp["mess"] != "UNKNOWN"].sort_values("total", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=grp["mess"], x=grp["total"],
        name="Total Spend",
        orientation="h",
        marker=dict(
            color=grp["total"],
            colorscale=[[0, "#1E293B"], [0.5, "#38BDF8"], [1, "#6EE7B7"]],
            showscale=False
        ),
        hovertemplate="<b>%{y}</b><br>Total: ₹%{x:,.0f}<extra></extra>"
    ))

    style_fig(fig, "Expenditure by Mess Unit", height=300)
    fig.update_layout(xaxis_title="Total Spend (₹)", yaxis_title="")
    return fig


# ─────────────────────────────────────────────────────────────────
# 6. TOP VENDORS
# ─────────────────────────────────────────────────────────────────
def build_vendor_chart(month: str, mess: str) -> go.Figure:
    df = get_filtered(month, mess)
    if df.empty or "vendor" not in df.columns:
        return go.Figure()

    top = (
        df.groupby("vendor")["amount"].sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top.columns = ["vendor", "amount"]
    top = top.sort_values("amount")
    top["short"] = top["vendor"].str[:30]

    colors = px.colors.sequential.Teal
    n = len(top)
    bar_colors = [colors[int(i / n * (len(colors) - 1))] for i in range(n)]

    fig = go.Figure(go.Bar(
        y=top["short"], x=top["amount"],
        orientation="h",
        marker=dict(color=bar_colors),
        hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>"
    ))
    style_fig(fig, "Top 15 Vendors by Spend", height=460)
    fig.update_layout(xaxis_title="Total Spend (₹)", yaxis_title="")
    return fig


# ─────────────────────────────────────────────────────────────────
# 7. WASTAGE CALENDAR HEATMAP
# ─────────────────────────────────────────────────────────────────
def build_wastage_heatmap(month: str, mess: str) -> go.Figure:
    df = get_filtered(month, mess)
    if df.empty:
        return go.Figure()

    daily = df.groupby(df["date"].dt.date)["amount"].sum().reset_index()
    daily.columns = ["date", "amount"]
    daily["date"] = pd.to_datetime(daily["date"])
    monthly_avg = daily["amount"].mean()
    daily["wastage"] = (daily["amount"] / monthly_avg * 100).round(1)
    daily["week"] = daily["date"].dt.isocalendar().week
    daily["dow"]  = daily["date"].dt.dayofweek
    daily["label"] = daily["date"].dt.strftime("%d %b") + "<br>" + daily["wastage"].astype(str) + " kg"

    pivot = daily.pivot(index="dow", columns="week", values="wastage")
    text  = daily.pivot(index="dow", columns="week", values="label")

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    y_labels = [days[i] for i in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=y_labels,
        text=text.values,
        texttemplate="%{text}",
        colorscale=[[0, "#0F1220"], [0.5, "#0EA5E9"], [1, "#6EE7B7"]],
        showscale=True,
        hovertemplate="Week %{x} — %{y}<br>Estimated Wastage: %{z:.1f} kg<extra></extra>",
        colorbar=dict(title="kg", tickfont=dict(color="#94A3B8"), titlefont=dict(color="#94A3B8"))
    ))
    style_fig(fig, "Estimated Wastage Heatmap (kg) by Week × Day", height=360)
    fig.update_layout(xaxis_title="Week No.", yaxis_title="")
    return fig


# ─────────────────────────────────────────────────────────────────
# 8. BENFORD'S LAW
# ─────────────────────────────────────────────────────────────────
def build_benford() -> tuple[go.Figure, str]:
    if DATA is None:
        return go.Figure(), "Load data first."

    df = DATA.copy()
    amounts = df["amount"].dropna()
    amounts = amounts[amounts > 0]

    first_digits = amounts.astype(str).str.replace(r"[^0-9]", "", regex=True).str.lstrip("0").str[0]
    first_digits = pd.to_numeric(first_digits, errors="coerce").dropna().astype(int)
    first_digits = first_digits[first_digits.between(1, 9)]

    obs_counts = first_digits.value_counts().sort_index()
    digits = list(range(1, 10))
    N = len(first_digits)
    expected_freq = [np.log10(1 + 1 / d) for d in digits]
    expected_counts = [f * N for f in expected_freq]
    observed = [obs_counts.get(d, 0) for d in digits]

    chi2_stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected_counts))
    df_val  = 8
    p_val   = 1 - chi2.cdf(chi2_stat, df_val)
    crit    = chi2.ppf(0.95, df_val)
    verdict = "❌ Deviates" if chi2_stat > crit else "✅ Conforms"

    # ── Figure ─────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=digits, y=[o / N * 100 for o in observed],
        name="Observed %",
        marker=dict(color="#38BDF8", opacity=0.85),
        hovertemplate="Digit %{x}<br>Observed: %{y:.1f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=digits, y=[e / N * 100 for e in expected_counts],
        name="Benford Expected %",
        mode="lines+markers",
        line=dict(color="#F472B6", width=2.5),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="Digit %{x}<br>Expected: %{y:.1f}%<extra></extra>"
    ))

    style_fig(fig, "Benford's Law — First-Digit Distribution", height=400)
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=digits, title="Leading Digit"),
        yaxis=dict(title="Frequency (%)"),
        bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    detail = ("Data deviates significantly from Benford's Law. Potential anomalies detected — recommend deeper audit."
              if chi2_stat > crit else
              "Data broadly follows Benford's Law. No immediate anomaly signal.")
    summary = f"""
**Chi-Square Statistic:** `{chi2_stat:.2f}`  
**Critical Value (α=0.05, df=8):** `{crit:.2f}`  
**P-Value:** `{p_val:.4f}`  
**Verdict:** {verdict} — {detail}

> *Benford's Law is a forensic heuristic; deviations may reflect legitimate batch-payment practices, not fraud.*
"""
    return fig, summary


# ─────────────────────────────────────────────────────────────────
# 9. NETWORK ANALYSIS
# ─────────────────────────────────────────────────────────────────
def build_network() -> tuple[go.Figure, str]:
    if DATA is None or "vendor" not in DATA.columns:
        return go.Figure(), "Load data with vendor column first."

    df = DATA.copy()
    G  = nx.Graph()

    edge_weights: dict[tuple, float] = {}
    for _, row in df.iterrows():
        mess   = str(row.get("mess", "UNKNOWN"))
        vendor = str(row.get("vendor", "?"))
        amt    = float(row.get("amount", 0))
        key    = (mess, vendor)
        edge_weights[key] = edge_weights.get(key, 0) + amt

    for (m, v), w in edge_weights.items():
        G.add_edge(m, v, weight=w)

    pos = nx.spring_layout(G, k=2.5, seed=42, weight="weight")

    # Degree centrality
    deg_cent   = nx.degree_centrality(G)
    wt_degree  = {n: sum(d.get("weight", 1) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
    communities = nx.community.greedy_modularity_communities(G)
    node_comm   = {}
    for i, c in enumerate(communities):
        for n in c:
            node_comm[n] = i

    comm_colors = ["#6EE7B7", "#38BDF8", "#F472B6", "#FBBF24", "#A78BFA"]
    mess_units  = {"CDH-1", "CDH-2", "CAFE", "UNKNOWN"} | set(df["mess"].dropna().unique())

    node_x, node_y, node_text, node_size, node_color, node_hover = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        is_mess = node in mess_units
        short   = node[:22] + "…" if len(str(node)) > 22 else str(node)
        node_text.append(short if is_mess else "")
        node_size.append(28 if is_mess else max(8, min(22, deg_cent[node] * 120)))
        node_color.append(comm_colors[node_comm.get(node, 0) % len(comm_colors)])
        node_hover.append(f"<b>{node}</b><br>Connections: {G.degree(node)}<br>Total Flow: ₹{wt_degree[node]:,.0f}")

    edge_traces = []
    for m, v, d in G.edges(data=True):
        x0, y0 = pos[m]; x1, y1 = pos[v]
        w = d.get("weight", 1)
        opacity = min(0.6, max(0.05, w / (max(wt_degree.values()) * 0.15)))
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(0.5, w / max(wt_degree.values()) * 6), color=f"rgba(110,231,183,{opacity})"),
            hoverinfo="none", showlegend=False
        ))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color="#E2E8F0"),
        marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color="rgba(255,255,255,0.2)")),
        hovertext=node_hover,
        hoverinfo="text",
        name="Nodes"
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    style_fig(fig, f"Mess–Vendor Financial Network  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)", height=520)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False
    )

    # Community stats
    modularity = nx.community.modularity(G, communities)
    top5 = sorted(wt_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_md = "\n".join([f"- **{k[:35]}**: ₹{v:,.0f}" for k, v in top5])
    summary = f"""
**Network Stats**  
- Nodes: `{G.number_of_nodes()}` · Edges: `{G.number_of_edges()}`  
- Communities detected: `{len(communities)}` (Modularity Q = `{modularity:.3f}`)  
- High modularity → Strong internal structure within vendor clusters  

**Top Financial Dependency Nodes**  
{top5_md}
"""
    return fig, summary


# ─────────────────────────────────────────────────────────────────
# 10. ASSOCIATION RULES
# ─────────────────────────────────────────────────────────────────
def build_rules(min_support: float, min_confidence: float, min_lift: float = 1.0) -> tuple[pd.DataFrame, str, go.Figure]:
    if DATA is None or "vendor" not in DATA.columns:
        empty_summary = """
<div id='arm-banner'>
  <h2>🔗 Association Rules Mining</h2>
  <p>FP-Growth · Vendor Co-occurrence Analysis</p>
</div>
<p style='color:#F87171;font-size:0.85rem'>⚠ Load data with a vendor column first, then click Mine.</p>
"""
        return pd.DataFrame(), empty_summary, go.Figure()

    df = DATA.copy()
    df["day_mess"] = df["date"].dt.date.astype(str) + "_" + df["mess"].astype(str)

    basket_raw = df.groupby(["day_mess", "vendor"]).size().unstack(fill_value=0)
    basket     = (basket_raw > 0).astype(bool)

    try:
        freq_items = fpgrowth(basket, min_support=min_support, use_colnames=True)
        if freq_items.empty:
            return pd.DataFrame(), "⚠ No frequent itemsets found. Try lowering the support threshold.", go.Figure()

        # ── mlxtend compat: newer versions require num_items ────
        try:
            rules_df = association_rules(
                freq_items,
                metric="confidence",
                min_threshold=min_confidence,
                num_items=len(basket.columns),
            )
        except TypeError:
            rules_df = association_rules(
                freq_items,
                metric="confidence",
                min_threshold=min_confidence,
            )

        rules_df = rules_df[rules_df["lift"] >= min_lift].sort_values("lift", ascending=False)

        if rules_df.empty:
            return pd.DataFrame(), "⚠ No rules meet the lift threshold. Try adjusting sliders.", go.Figure()

        rules_df["antecedents"] = rules_df["antecedents"].apply(
            lambda x: ", ".join(sorted(str(i)[:20] for i in x))
        )
        rules_df["consequents"] = rules_df["consequents"].apply(
            lambda x: ", ".join(sorted(str(i)[:20] for i in x))
        )

        out = rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
        out["support"]    = out["support"].round(3)
        out["confidence"] = out["confidence"].round(3)
        out["lift"]       = out["lift"].round(3)
        out = out.head(50)

        # ── stats for summary card ───────────────────────────────
        n_rules   = len(out)
        avg_lift  = out["lift"].mean()
        max_conf  = out["confidence"].max()

        # ── Scatter plot (bubble = lift) — Google palette ───────
        lift_vals  = out["lift"].tolist()          # FIX: Series → list
        conf_vals  = out["confidence"].tolist()
        sup_vals   = out["support"].tolist()
        size_vals  = [max(8, v * 14) for v in lift_vals]   # FIX: scalar list
        hover_text = (out["antecedents"] + " → " + out["consequents"]).tolist()

        # Google-brand colour ramp: blue → green → yellow
        colorscale = [
            [0.0,  "#4285F4"],
            [0.33, "#34A853"],
            [0.66, "#FBBC05"],
            [1.0,  "#EA4335"],
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sup_vals,
            y=conf_vals,
            mode="markers",
            marker=dict(
                size=size_vals,
                color=lift_vals,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="Lift",
                    tickfont=dict(color="#94A3B8", size=10),
                    titlefont=dict(color="#94A3B8", size=11),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(66,133,244,0.2)",
                ),
                line=dict(width=1.5, color="rgba(255,255,255,0.15)"),
                opacity=0.88,
                sizemode="diameter",
            ),
            text=hover_text,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Support: %{x:.3f}<br>"
                "Confidence: %{y:.3f}<br>"
                "Lift: %{marker.color:.3f}<extra></extra>"
            ),
        ))

        # Reference lines
        fig.add_hline(y=min_confidence, line=dict(color="rgba(66,133,244,0.4)", width=1.5, dash="dot"),
                      annotation_text=f"min conf {min_confidence}", annotation_font_color="#90CAF9",
                      annotation_font_size=10)
        fig.add_vline(x=min_support, line=dict(color="rgba(52,168,83,0.4)", width=1.5, dash="dot"),
                      annotation_text=f"min sup {min_support}", annotation_font_color="#86EFAC",
                      annotation_font_size=10)

        style_fig(fig, "Association Rules — Support vs Confidence  (bubble size & colour = Lift)", height=460)
        fig.update_layout(
            xaxis_title="Support",
            yaxis_title="Confidence",
            xaxis=dict(gridcolor="rgba(66,133,244,0.08)"),
            yaxis=dict(gridcolor="rgba(66,133,244,0.08)"),
        )

        # ── Summary HTML (Google doodle style) ──────────────────
        summary = f"""
<div id="arm-banner">
  <h2>🔗 Association Rules Mining</h2>
  <p>FP-Growth · Vendor Co-occurrence · Results Ready</p>
</div>
<div class="arm-stats-grid">
  <div class="arm-stat-card blue">
    <div class="stat-icon">📋</div>
    <div class="stat-label">Rules Found</div>
    <div class="stat-val">{n_rules}</div>
  </div>
  <div class="arm-stat-card green">
    <div class="stat-icon">📈</div>
    <div class="stat-label">Avg Lift</div>
    <div class="stat-val">{avg_lift:.2f}×</div>
  </div>
  <div class="arm-stat-card amber">
    <div class="stat-icon">🎯</div>
    <div class="stat-label">Max Confidence</div>
    <div class="stat-val">{max_conf:.0%}</div>
  </div>
</div>
<p style="font-size:0.78rem;color:#64748B;margin:0.3rem 0 0.8rem">
  Min support <strong style="color:#60A5FA">{min_support}</strong> ·
  Min confidence <strong style="color:#4ADE80">{min_confidence}</strong> ·
  Min lift <strong style="color:#FCD34D">{min_lift}</strong>
</p>
"""
        return out, summary, fig

    except Exception as e:
        return pd.DataFrame(), f"❌ Error: {e}", go.Figure()


# ─────────────────────────────────────────────────────────────────
# 11. ANOMALY DETECTION (ZSCORE)
# ─────────────────────────────────────────────────────────────────
def build_anomaly(month: str, mess: str, zscore_thresh: float) -> tuple[go.Figure, pd.DataFrame]:
    df = get_filtered(month, mess)
    if df.empty:
        return go.Figure(), pd.DataFrame()

    daily = df.groupby(df["date"].dt.date)["amount"].sum().reset_index()
    daily.columns = ["date", "amount"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    mean = daily["amount"].mean()
    std  = daily["amount"].std()
    daily["zscore"] = (daily["amount"] - mean) / std
    daily["anomaly"] = daily["zscore"].abs() > zscore_thresh

    fig = go.Figure()

    # Normal days
    normal = daily[~daily["anomaly"]]
    fig.add_trace(go.Scatter(
        x=normal["date"], y=normal["amount"],
        mode="markers+lines",
        name="Normal",
        line=dict(color="#38BDF8", width=1.5),
        marker=dict(size=5, color="#38BDF8"),
        hovertemplate="%{x|%d %b}<br>₹%{y:,.0f}<extra></extra>"
    ))

    # Anomaly days
    anom = daily[daily["anomaly"]]
    fig.add_trace(go.Scatter(
        x=anom["date"], y=anom["amount"],
        mode="markers",
        name=f"Anomaly (|Z|>{zscore_thresh})",
        marker=dict(size=12, color="#F472B6", symbol="star",
                    line=dict(width=1.5, color="rgba(244,114,182,0.5)")),
        hovertemplate="⚠ <b>%{x|%d %b %Y}</b><br>₹%{y:,.0f}<extra></extra>"
    ))

    # Threshold bands
    fig.add_hline(y=mean + zscore_thresh * std, line=dict(color="#FBBF24", width=1, dash="dot"), annotation_text=f"+{zscore_thresh}σ")
    fig.add_hline(y=max(0, mean - zscore_thresh * std), line=dict(color="#FBBF24", width=1, dash="dot"), annotation_text=f"-{zscore_thresh}σ")

    style_fig(fig, f"Anomaly Detection — Z-Score Threshold ±{zscore_thresh}σ", height=400)

    anom_table = anom[["date", "amount", "zscore"]].copy()
    anom_table["date"]   = anom_table["date"].dt.strftime("%Y-%m-%d")
    anom_table["amount"] = anom_table["amount"].apply(lambda x: f"₹{x:,.0f}")
    anom_table["zscore"] = anom_table["zscore"].round(2)
    anom_table.columns   = ["Date", "Amount", "Z-Score"]

    return fig, anom_table


# ─────────────────────────────────────────────────────────────────
# 12. MONTHLY SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────
def build_monthly_summary() -> pd.DataFrame:
    if DATA is None:
        return pd.DataFrame()

    df = DATA.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    daily = df.groupby(df["date"].dt.date)["amount"].sum()
    monthly_avg = daily.mean()

    grp = df.groupby("month")["amount"].agg(["sum", "mean", "count", "max"])
    grp.columns = ["Total (₹)", "Avg Daily (₹)", "Transactions", "Peak Day (₹)"]
    grp["Est. Wastage (kg)"] = (grp["Total (₹)"] / (monthly_avg * 30) * 100 * 30).round(0)

    for col in ["Total (₹)", "Avg Daily (₹)", "Peak Day (₹)"]:
        grp[col] = grp[col].apply(lambda x: f"₹{x:,.0f}")
    grp["Est. Wastage (kg)"] = grp["Est. Wastage (kg)"].apply(lambda x: f"{x:.0f} kg")

    return grp.reset_index().rename(columns={"month": "Month"})


# ─────────────────────────────────────────────────────────────────
# 13. ABOUT / README HTML
# ─────────────────────────────────────────────────────────────────
ABOUT_HTML = """
<div style="font-family:'DM Mono',monospace;color:#94A3B8;font-size:0.82rem;line-height:1.8;padding:0.5rem">

<h2 style="font-family:'Syne',sans-serif;color:#F8FAFC;font-size:1.4rem;margin-bottom:1rem">
  Mess Intelligence Platform
</h2>

<div class="alert-box" style="margin-bottom:1rem">
  📊 A startup-grade DWBI analytics suite for institutional food system optimization.
  Built for IISER Thiruvananthapuram — generalized for any institutional mess.
</div>

<h3 style="color:#6EE7B7;font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;margin-top:1.5rem">CSV Format</h3>
<p>Your CSV must contain these columns (flexible naming detected automatically):</p>
<table style="width:100%;border-collapse:collapse;margin:0.8rem 0">
  <thead><tr style="border-bottom:1px solid rgba(110,231,183,0.2)">
    <th style="text-align:left;padding:6px;color:#6EE7B7">Column</th>
    <th style="text-align:left;padding:6px;color:#6EE7B7">Description</th>
    <th style="text-align:left;padding:6px;color:#6EE7B7">Example</th>
  </tr></thead>
  <tbody>
    <tr><td style="padding:6px">date</td><td style="padding:6px">Payment/bill date</td><td style="padding:6px">2025-08-01</td></tr>
    <tr><td style="padding:6px">amount</td><td style="padding:6px">Transaction amount (₹)</td><td style="padding:6px">150000</td></tr>
    <tr><td style="padding:6px">vendor</td><td style="padding:6px">Vendor/supplier name</td><td style="padding:6px">Amma Vegetables</td></tr>
    <tr><td style="padding:6px">mess / unit</td><td style="padding:6px">Mess unit name</td><td style="padding:6px">CDH-1</td></tr>
  </tbody>
</table>

<h3 style="color:#6EE7B7;font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;margin-top:1.5rem">Modules</h3>
<ul>
  <li><strong style="color:#F8FAFC">Overview</strong> — KPI cards, daily trend, mess comparison, top vendors</li>
  <li><strong style="color:#F8FAFC">Wastage Analysis</strong> — Calendar heatmap + monthly summary with proxy wastage model</li>
  <li><strong style="color:#F8FAFC">Anomaly Detection</strong> — Z-score based statistical outlier flagging</li>
  <li><strong style="color:#F8FAFC">Benford's Law</strong> — Forensic first-digit analysis with χ² test</li>
  <li><strong style="color:#F8FAFC">Network Analysis</strong> — Mess–vendor graph with community detection</li>
  <li><strong style="color:#F8FAFC">Association Rules</strong> — FP-Growth frequent pattern mining</li>
</ul>

<h3 style="color:#6EE7B7;font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;margin-top:1.5rem">Deploy on GitHub</h3>
<pre style="background:rgba(0,0,0,0.4);border:1px solid rgba(110,231,183,0.15);border-radius:8px;padding:1rem;overflow-x:auto">
git init
git add app.py requirements.txt README.md
git commit -m "feat: launch Mess Intelligence Platform"
git remote add origin https://github.com/YOUR_USER/mess-analytics
git push -u origin main

# Then deploy on Hugging Face Spaces (Gradio SDK)
# Or: railway up / fly deploy</pre>

<h3 style="color:#6EE7B7;font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;margin-top:1.5rem">Methodology</h3>
<p>
  Wastage proxy: <code style="background:rgba(0,0,0,0.3);padding:2px 6px;border-radius:4px">W(d) = E(d)/E̅_month × k</code> where k=100 kg baseline.<br>
  Benford test: χ² goodness-of-fit, df=8, α=0.05.<br>
  Network: Louvain community detection, weighted bipartite graph.<br>
  ARM: FP-Growth with configurable support/confidence/lift thresholds.
</p>

<p style="margin-top:2rem;color:#475569;font-size:0.75rem">
  © 2025 Maitreya Sameer Ganu · IISER Thiruvananthapuram · Course Project DWBI<br>
  Supervisor: Dr. Zakaria Laskar · School of Data Science
</p>
</div>
"""


# ─────────────────────────────────────────────────────────────────
# UI — GRADIO BLOCKS
# ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("DM Mono"), "monospace"],
    ),
    css=CUSTOM_CSS,
    title="Mess Intelligence Platform",
    analytics_enabled=False,
) as demo:

    # ── HEADER ─────────────────────────────────────────────────
    gr.HTML("""
    <div id="header-hero">
      <h1>Mess Intelligence Platform</h1>
      <p>Data Warehousing & Business Intelligence · Institutional Food Systems Analytics</p>
    </div>
    """)

    # ── DATA LOAD ROW ───────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            file_input  = gr.File(label="Upload Mess Expense CSV", file_types=[".csv"])
        with gr.Column(scale=1):
            load_btn    = gr.Button("⚡ Load & Analyse", variant="primary", size="lg")
    load_status = gr.Markdown("*Upload a CSV file and click Load to begin.*")

    # ── FILTERS ─────────────────────────────────────────────────
    with gr.Row():
        month_dd = gr.Dropdown(["All"], value="All", label="Month Filter", interactive=True)
        mess_dd  = gr.Dropdown(["All"], value="All", label="Mess Unit",   interactive=True)

    # ── KPI CARDS ───────────────────────────────────────────────
    kpi_html = gr.HTML()

    # ── TABS ────────────────────────────────────────────────────
    with gr.Tabs():

        # ── TAB 1 · OVERVIEW ─────────────────────────────────
        with gr.Tab("📈 Overview"):
            with gr.Row():
                trend_plot = gr.Plot(show_label=False)
            with gr.Row():
                mess_plot  = gr.Plot(show_label=False)
                vendor_plot = gr.Plot(show_label=False)

        # ── TAB 2 · WASTAGE ──────────────────────────────────
        with gr.Tab("🗑 Wastage Analysis"):
            waste_heatmap = gr.Plot(show_label=False)
            gr.Markdown("### Monthly Summary")
            monthly_table = gr.Dataframe(interactive=False)

        # ── TAB 3 · ANOMALY ──────────────────────────────────
        with gr.Tab("🔍 Anomaly Detection"):
            with gr.Row():
                zscore_slider = gr.Slider(1.0, 3.5, value=2.0, step=0.1, label="Z-Score Threshold (σ)")
            anom_plot  = gr.Plot(show_label=False)
            anom_table = gr.Dataframe(label="Flagged Anomaly Dates", interactive=False)

        # ── TAB 4 · BENFORD ──────────────────────────────────
        with gr.Tab("📐 Benford's Law"):
            with gr.Row():
                benford_run_btn = gr.Button("Run Benford Analysis", variant="primary")
            benford_plot    = gr.Plot(show_label=False)
            benford_summary = gr.Markdown()

        # ── TAB 5 · NETWORK ──────────────────────────────────
        with gr.Tab("🕸 Network Analysis"):
            with gr.Row():
                network_run_btn = gr.Button("Build Financial Network", variant="primary")
            network_plot    = gr.Plot(show_label=False)
            network_summary = gr.Markdown()

        # ── TAB 6 · ASSOCIATION RULES ────────────────────────
        with gr.Tab("🔗 Association Rules"):
            gr.HTML("""
            <div id="arm-banner">
              <h2>🔗 Association Rules Mining</h2>
              <p>FP-Growth · Vendor Co-occurrence Pattern Analysis</p>
            </div>
            """)
            with gr.Group(elem_id="arm-controls"):
                with gr.Row():
                    sup_slider  = gr.Slider(0.01, 0.5,  value=0.05, step=0.01, label="🔵 Min Support")
                    conf_slider = gr.Slider(0.1,  1.0,  value=0.60, step=0.05, label="🟢 Min Confidence")
                    lift_slider = gr.Slider(1.0,  5.0,  value=1.0,  step=0.1,  label="🟡 Min Lift")
            with gr.Row():
                with gr.Column(scale=1):
                    rules_run_btn = gr.Button("⚡ Mine Association Rules", variant="primary", elem_id="arm-mine-btn")
            rules_summary = gr.HTML()
            with gr.Group(elem_id="arm-table"):
                rules_table = gr.Dataframe(label="Strong Rules (top 50)", interactive=False)
            rules_scatter = gr.Plot(show_label=False)

        # ── TAB 7 · ABOUT ────────────────────────────────────
        with gr.Tab("ℹ About / Deploy"):
            gr.HTML(ABOUT_HTML)

    # ─────────────────────────────────────────────────────────
    # WIRING — EVENT HANDLERS
    # ─────────────────────────────────────────────────────────

    def on_load(file):
        msg, m_update, u_update = load_data(file)
        return msg, m_update, u_update

    def refresh_overview(month, mess):
        return (
            build_kpis(month, mess),
            build_trend(month, mess),
            build_mess_comparison(month, mess),
            build_vendor_chart(month, mess),
        )

    def refresh_wastage(month, mess):
        return build_wastage_heatmap(month, mess), build_monthly_summary()

    def refresh_anomaly(month, mess, zs):
        return build_anomaly(month, mess, zs)

    # Load button
    load_btn.click(
        fn=on_load,
        inputs=[file_input],
        outputs=[load_status, month_dd, mess_dd],
    ).then(
        fn=refresh_overview,
        inputs=[month_dd, mess_dd],
        outputs=[kpi_html, trend_plot, mess_plot, vendor_plot],
    ).then(
        fn=refresh_wastage,
        inputs=[month_dd, mess_dd],
        outputs=[waste_heatmap, monthly_table],
    )

    # Filter changes
    for trigger in [month_dd, mess_dd]:
        trigger.change(
            fn=refresh_overview,
            inputs=[month_dd, mess_dd],
            outputs=[kpi_html, trend_plot, mess_plot, vendor_plot],
        )
        trigger.change(
            fn=refresh_wastage,
            inputs=[month_dd, mess_dd],
            outputs=[waste_heatmap, monthly_table],
        )
        trigger.change(
            fn=refresh_anomaly,
            inputs=[month_dd, mess_dd, zscore_slider],
            outputs=[anom_plot, anom_table],
        )

    zscore_slider.change(
        fn=refresh_anomaly,
        inputs=[month_dd, mess_dd, zscore_slider],
        outputs=[anom_plot, anom_table],
    )

    # Benford
    benford_run_btn.click(
        fn=build_benford,
        inputs=[],
        outputs=[benford_plot, benford_summary],
    )

    # Network
    network_run_btn.click(
        fn=build_network,
        inputs=[],
        outputs=[network_plot, network_summary],
    )

    # ARM
    rules_run_btn.click(
        fn=build_rules,
        inputs=[sup_slider, conf_slider, lift_slider],
        outputs=[rules_table, rules_summary, rules_scatter],
    )


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # needed for Docker / cloud deploys
        server_port=7860,
        share=False,            # set True for temporary public URL
        favicon_path=None,
        show_error=True,
    )
