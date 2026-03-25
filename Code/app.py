
import os
import re
import warnings
from datetime import datetime

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from scipy import stats
from scipy.stats import chi2, norm, shapiro, kstest

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# SOLSTICE COLOR PALETTE & THEMING
# ──────────────────────────────────────────────────────────────────
THEME = {
    "bg_app":      "#0F111A",
    "bg_card":     "#181B28",
    "bg_plot":     "#181B28",
    "border":      "#2A2D3D",
    "text_main":   "#F0F2F5",
    "text_muted":  "#8F95B2",
    "grid":        "#222533",
    "coral":       "#FF6B6B",
    "gold":        "#FCA311",
    "teal":        "#4ECDC4",
    "purple":      "#9D4EDD",
    "blue":        "#3A86FF"
}

ACCENTS = [
    THEME["teal"], THEME["coral"], THEME["gold"], THEME["purple"], THEME["blue"],
    "#FF9F43", "#EE5A24", "#009432", "#1289A7", "#C4E538",
    "#FDA7DF", "#D980FA", "#12CBC4", "#F79F1F", "#A3CB38",
    "#ED4C67", "#B53471", "#5758BB", "#1B1464", "#006266",
]

BENFORD_EXPECTED = {d: np.log10(1 + 1 / d) * 100 for d in range(1, 10)}


# ──────────────────────────────────────────────────────────────────
# HELPERS: COLORS & LAYOUT
# ──────────────────────────────────────────────────────────────────
def get_color_sequence(n: int) -> list:
    """Returns exactly n distinct colors, cycling through the ACCENTS palette."""
    return [ACCENTS[i % len(ACCENTS)] for i in range(n)]


def _layout(**kwargs):
    """Enforces the Solstice theme on all Plotly charts."""
    base = dict(
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text_main"], family="Inter, sans-serif"),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(
            gridcolor=THEME["grid"],
            zerolinecolor=THEME["grid"],
            tickfont=dict(color=THEME["text_muted"])
        ),
        yaxis=dict(
            gridcolor=THEME["grid"],
            zerolinecolor=THEME["grid"],
            tickfont=dict(color=THEME["text_muted"])
        ),
        legend=dict(
            bgcolor="rgba(24, 27, 40, 0.8)",
            font=dict(color=THEME["text_muted"])
        )
    )
    for k, v in kwargs.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def _fmt_inr(val: float) -> str:
    """Format a rupee amount with adaptive precision so bars and labels always agree."""
    if val >= 1e7:
        return f"₹{val/1e7:.2f}Cr"
    elif val >= 1e5:
        return f"₹{val/1e5:.2f}L"
    elif val >= 1e3:
        return f"₹{val/1e3:.1f}K"
    else:
        return f"₹{val:,.0f}"


# ──────────────────────────────────────────────────────────────────
# UI COMPONENT HELPERS
# ──────────────────────────────────────────────────────────────────
def _pill(icon_label, value, color, delay=0):
    anim_style = f"animation: fadeInUp 0.6s ease-out {delay}s forwards; opacity: 0; transform: translateY(20px);"
    return (
        f'<div class="kpi-card hover-glow" style="border-bottom: 3px solid {color}; {anim_style}">'
        f'<div style="font-size: 12px; font-weight: 700; color: {THEME["text_muted"]}; '
        f'text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">{icon_label}</div>'
        f'<div style="font-size: 28px; font-weight: 900; color: {THEME["text_main"]}; '
        f'text-shadow: 0 0 15px {color}40;">{value}</div>'
        f'</div>'
    )

def _pills_row(*items):
    pills = "".join(_pill(*i) for i in items)
    return f'<div style="display:flex;flex-wrap:wrap;gap:20px;padding:10px 0 24px 0;">{pills}</div>'


# ──────────────────────────────────────────────────────────────────
# DATA NORMALIZATION & INGESTION
# ──────────────────────────────────────────────────────────────────
def normalize_vendor(name: str) -> str:
    """
    Generic vendor name normalization — removes special characters,
    normalises whitespace, and title-cases. No dataset-specific aliases.
    """
    name = str(name).lower().strip()
    name = re.sub(r'[-_/()]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.title()


def load_and_validate(file_obj):
    if file_obj is None:
        return None, "No dataset uploaded."
    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"Data parsing error: {e}"

    required = {"date", "amount", "vendor", "classes"}
    cols_lower = set(df.columns.str.lower().str.strip())
    missing = required - cols_lower
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}"

    df.columns = df.columns.str.lower().str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])
    df = df[df["amount"] > 0]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["class_label"]  = df["classes"].fillna("UNCATEGORIZED").astype(str).str.strip().str.upper()
    df["vendor"]       = df["vendor"].fillna("UNKNOWN").astype(str)
    df["vendor_clean"] = df["vendor"].apply(normalize_vendor)
    df["month_name"]   = df["date"].dt.strftime("%b %Y")
    df["day"]          = df["date"].dt.date
    return df, ""


# ──────────────────────────────────────────────────────────────────
# ANALYTICS MODULES
# ──────────────────────────────────────────────────────────────────
def build_overview(df: pd.DataFrame):
    total  = df["amount"].sum()
    n_tx   = len(df)
    avg_d  = df.groupby("day")["amount"].sum().mean()
    n_vend = df["vendor_clean"].nunique()

    kpi_html = _pills_row(
        ("💸 Total Expenditure", _fmt_inr(total),       THEME["teal"],   0.1),
        ("📋 Transactions",      f"{n_tx:,}",            THEME["purple"], 0.2),
        ("📅 Avg Daily Spend",   _fmt_inr(avg_d),        THEME["gold"],   0.3),
        ("🏪 Unique Vendors",    str(n_vend),            THEME["coral"],  0.4),
    )

    # ── Monthly bar chart ──────────────────────────────────────────
    months_order = sorted(
        df["month_name"].unique(),
        key=lambda x: datetime.strptime(x, "%b %Y")
    )
    monthly = df.groupby("month_name")["amount"].sum().reindex(months_order)

    fig_monthly = go.Figure(go.Bar(
        x=list(monthly.index),
        y=monthly.values / 1e5,
        marker=dict(color=THEME["teal"], opacity=0.9),
        # FIX: use adaptive formatter so label exactly matches bar height
        text=[_fmt_inr(v) for v in monthly.values],
        textposition="outside",
    ))
    fig_monthly.update_layout(**_layout(
        title="Monthly Financial Velocity",
        yaxis=dict(title="Volume (₹ Lakhs)"),
        xaxis=dict(tickangle=-45),
        showlegend=False,
        height=400,
    ))

    # ── Pie chart ─────────────────────────────────────────────────
    by_class = df.groupby("class_label")["amount"].sum().reset_index()
    by_class = by_class.sort_values("amount", ascending=False)
    n_slices = len(by_class)

    # FIX: always supply exactly as many colours as there are slices
    pie_colors = get_color_sequence(n_slices)

    fig_pie = go.Figure(go.Pie(
        labels=by_class["class_label"],
        values=by_class["amount"],          # true rupee sums drive the %
        hole=0.55,
        marker=dict(
            colors=pie_colors,              # length == n_slices, no mismatch
            line=dict(color=THEME["bg_card"], width=3)
        ),
        textinfo="label+percent",
        insidetextorientation="horizontal",
        hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    fig_pie.update_layout(**_layout(
        title="Expenditure Distribution by Unit",
        height=420,
        legend=dict(orientation="v", x=1.02, y=0.5),
    ))

    # ── Top-10 vendor bar chart ────────────────────────────────────
    top10 = (
        df.groupby("vendor_clean")["amount"]
        .sum()
        .sort_values(ascending=True)
        .tail(10)
        .reset_index()
    )

    # FIX: adaptive labels so every label matches its bar precisely
    bar_labels = [_fmt_inr(v) for v in top10["amount"]]

    fig_vend = go.Figure(go.Bar(
        x=top10["amount"] / 1e5,
        y=top10["vendor_clean"],
        orientation="h",
        marker=dict(color=THEME["purple"], opacity=0.9),
        text=bar_labels,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Amount: ₹%{x:.4f}L<extra></extra>",
    ))
    fig_vend.update_layout(**_layout(
        title="Top 10 Tier-1 Vendors",
        xaxis=dict(title="Volume (₹ Lakhs)"),
        showlegend=False,
        height=420,
    ))

    return kpi_html, fig_monthly, fig_pie, fig_vend


# ──────────────────────────────────────────────────────────────────
def build_wastage(df: pd.DataFrame):
    daily = df.groupby(["day", "month_name"])["amount"].sum().reset_index()
    monthly_avg = daily.groupby("month_name")["amount"].transform("mean").replace(0, np.nan)
    daily["wastage_kg"] = (daily["amount"] / monthly_avg) * 100
    daily["date"]       = pd.to_datetime(daily["day"])
    daily["high_risk"]  = daily["amount"] > monthly_avg
    daily = daily.sort_values("date")

    colors = [THEME["coral"] if r else THEME["blue"] for r in daily["high_risk"]]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily["date"], y=daily["wastage_kg"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(58, 134, 255, 0.1)",
        line=dict(color=THEME["blue"], width=2),
        marker=dict(color=colors, size=6, line=dict(width=0)),
        name="Risk Index",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Proxy Index: %{y:.1f}<extra></extra>",
    ))
    fig_trend.add_hline(
        y=100,
        line=dict(color=THEME["gold"], dash="dash", width=2),
        annotation_text="Baseline Risk (100)",
        annotation_font=dict(color=THEME["gold"]),
    )
    fig_trend.update_layout(**_layout(
        title="Daily Operational Inefficiency & Risk Proxy",
        yaxis=dict(title="Risk Index (Units)"),
        xaxis=dict(title="Timeline"),
        height=400,
    ))

    dclass     = df.groupby(["day", "month_name", "class_label"])["amount"].sum().reset_index()
    overall_mean = dclass["amount"].mean() or 1
    gmean2     = dclass.groupby(["month_name", "class_label"])["amount"].mean().reset_index()
    gmean2["wastage_kg"] = gmean2["amount"] / overall_mean * 100
    gmean2["month_sort"] = gmean2["month_name"].apply(lambda x: datetime.strptime(x, "%b %Y"))
    gmean2 = gmean2.sort_values("month_sort")

    n_classes  = gmean2["class_label"].nunique()
    fig_cmp = px.bar(
        gmean2, x="month_name", y="wastage_kg",
        color="class_label", barmode="group",
        color_discrete_sequence=get_color_sequence(n_classes),
        labels={"month_name": "Month", "wastage_kg": "Risk Index", "class_label": "Unit"},
        title="Risk Profile by Unit",
    )
    fig_cmp.update_layout(**_layout(height=400))

    hr = (daily[daily["high_risk"]][["date", "month_name", "amount", "wastage_kg"]]
          .sort_values("wastage_kg", ascending=False).head(20).copy())
    hr["date"]       = hr["date"].dt.strftime("%Y-%m-%d")
    hr["amount"]     = hr["amount"].apply(_fmt_inr)
    hr["wastage_kg"] = hr["wastage_kg"].apply(lambda x: f"{x:.1f}")
    hr.columns       = ["Date", "Month", "Expenditure Outflow", "Risk Index"]

    return fig_trend, fig_cmp, hr


# ──────────────────────────────────────────────────────────────────
def build_network(df: pd.DataFrame):
    from networkx.algorithms.community import greedy_modularity_communities
    from networkx.algorithms.community.quality import modularity

    G = nx.Graph()
    for _, row in df.iterrows():
        vendor = row["vendor"]
        mess   = row["class_label"]
        weight = row["amount"]
        G.add_node(vendor, node_type="vendor")
        G.add_node(mess,   node_type="mess")
        if G.has_edge(vendor, mess):
            G[vendor][mess]["weight"] += weight
        else:
            G.add_edge(vendor, mess, weight=weight)

    node_weights = {
        node: sum(d["weight"] for _, _, d in G.edges(node, data=True))
        for node in G.nodes()
    }

    communities      = list(greedy_modularity_communities(G, weight="weight"))
    modularity_score = modularity(G, communities, weight="weight")
    community_map    = {n: i for i, comm in enumerate(communities) for n in comm}
    pos              = nx.spring_layout(G, seed=42, k=2.0, iterations=80)
    total_spend      = df["amount"].sum() or 1

    fig_net = go.Figure()
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        w   = data.get("weight", 1)
        opc = min(0.7, 0.15 + w / total_spend * 8)
        fig_net.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(0.5, min(4.0, w / 2e5)),
                      color=f"rgba(143,149,178,{opc:.2f})"),
            hoverinfo="none", showlegend=False,
        ))

    for node in G.nodes():
        x, y = pos[node]
        nw   = node_weights.get(node, 0)
        cid  = community_map[node]
        fig_net.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(
                size=max(10, min(35, nw / 4e5)),
                color=ACCENTS[cid % len(ACCENTS)],
                line=dict(width=2, color=THEME["bg_card"]),
                opacity=0.95,
            ),
            text=node if nw > 8e5 else "",
            textposition="top center",
            showlegend=False,
            hovertemplate=f"<b>{node}</b><br>Community: {cid}<br>Total Flow: ₹{nw:,.0f}<extra></extra>",
        ))

    fig_net.update_layout(**_layout(
        title="Network Analysis (Greedy Modularity Communities)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=600,
    ))

    stats_html = _pills_row(
        ("System Nodes", str(G.number_of_nodes()), THEME["teal"],   0.1),
        ("Active Edges", str(G.number_of_edges()), THEME["blue"],   0.2),
        ("Communities",  str(len(communities)),     THEME["purple"], 0.3),
        ("Modularity",   f"{modularity_score:.3f}", THEME["gold"],   0.4),
    )
    return fig_net, stats_html


# ──────────────────────────────────────────────────────────────────
def build_arm(df: pd.DataFrame, min_sup=0.05, min_conf=0.60, min_lift=1.0):
    transactions = (
        df.groupby(["day", "class_label"])["vendor_clean"]
        .apply(lambda x: list(set(x)))
        .tolist()
    )

    te    = TransactionEncoder()
    arr   = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(arr, columns=te.columns_)

    try:
        frequent = fpgrowth(basket, min_support=min_sup, use_colnames=True)
        rules    = association_rules(frequent, metric="confidence", min_threshold=min_conf)
        rules    = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]}), None, None, f"Error: {e}"

    if rules.empty:
        return pd.DataFrame({"Info": ["No structural rules at current thresholds."]}), None, None, "No patterns detected."

    r_disp = rules.copy()
    r_disp["antecedents"] = r_disp["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    r_disp["consequents"] = r_disp["consequents"].apply(lambda x: ", ".join(sorted(x)))
    r_disp = r_disp[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)
    r_disp["support"]    = r_disp["support"].apply(lambda x: f"{x:.3f}")
    r_disp["confidence"] = r_disp["confidence"].apply(lambda x: f"{x:.3%}")
    r_disp["lift"]       = r_disp["lift"].apply(lambda x: f"{x:.3f}")
    r_disp.index += 1
    r_disp.columns = ["If (Antecedents)", "Then (Consequents)", "Support", "Confidence", "Lift"]

    top_items = basket.sum().sort_values(ascending=False).head(10)
    fig_freq  = go.Figure(go.Bar(
        x=top_items.values, y=top_items.index,
        orientation="h",
        marker=dict(color=THEME["coral"], opacity=0.9),
    ))
    fig_freq.update_layout(**_layout(
        title="Vendor Transaction Frequency (Top 10)",
        xaxis=dict(title="Transaction Count"),
        yaxis=dict(autorange="reversed"),
        height=400,
    ))

    top_rules = rules.head(10)
    labels    = [f"{list(a)} → {list(c)}" for a, c in zip(top_rules["antecedents"], top_rules["consequents"])]
    fig_rules = go.Figure(go.Bar(
        x=top_rules["lift"], y=labels,
        orientation="h",
        marker=dict(color=THEME["gold"], opacity=0.9),
    ))
    fig_rules.update_layout(**_layout(
        title="Top 10 Structural Dependencies (by Lift)",
        xaxis=dict(title="Lift Metric"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        height=400,
    ))

    stats_html = _pills_row(
        ("Total Transactions", str(len(transactions)), THEME["teal"],   0.1),
        ("Frequent Sets",      str(len(frequent)),     THEME["purple"], 0.2),
        ("Strong Rules",       str(len(rules)),        THEME["coral"],  0.3),
        ("Max Lift Detected",  f"{rules['lift'].max():.2f}", THEME["gold"], 0.4),
    )
    return r_disp, fig_freq, fig_rules, stats_html


# ──────────────────────────────────────────────────────────────────
def build_benford(df: pd.DataFrame):
    amounts = df["amount"].dropna()
    amounts = amounts[amounts > 0]

    def first_digit(x):
        s = str(abs(int(x)))
        return int(s[0]) if s and s[0] != "0" else None

    fd  = amounts.apply(first_digit).dropna().astype(int)
    n   = len(fd)

    obs_cnt = {d: int((fd == d).sum()) for d in range(1, 10)}
    exp_cnt = {d: max(1, int(np.round(BENFORD_EXPECTED[d] / 100 * n))) for d in range(1, 10)}
    obs_pct = {d: obs_cnt[d] / n * 100 for d in range(1, 10)}

    chi2_stat = sum((obs_cnt[d] - exp_cnt[d]) ** 2 / exp_cnt[d] for d in range(1, 10))
    p_value   = 1 - chi2.cdf(chi2_stat, df=8)
    critical  = chi2.ppf(0.95, df=8)
    conforms  = chi2_stat <= critical

    digits = list(range(1, 10))
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=digits, y=[obs_pct[d] for d in digits],
        name="Observed",
        marker=dict(color=THEME["teal"], opacity=0.9),
    ))
    fig_bar.add_trace(go.Bar(
        x=digits, y=[BENFORD_EXPECTED[d] for d in digits],
        name="Expected (Benford)",
        marker=dict(color=THEME["purple"], opacity=0.7),
    ))
    fig_bar.update_layout(**_layout(
        barmode="group",
        title="Forensic Audit: Benford's Law Distribution",
        xaxis=dict(title="Leading Digit", tickvals=digits),
        yaxis=dict(title="Frequency (%)"),
        height=450,
    ))

    sc   = THEME["teal"] if conforms else THEME["coral"]
    icon = "VERIFIED" if conforms else "ANOMALY DETECTED"
    st   = "System Conforms to Benford's Law" if conforms else "Structural Deviation Detected"
    desc = (
        "Financial distribution aligns naturally with expected macroscopic economic models."
        if conforms else
        "Distributional anomalies present. Recommend deep-dive audit on vendor batching practices."
    )

    result_html = (
        f'<div class="hover-glow" style="background-color:{THEME["bg_card"]};border:1px solid {sc}50;'
        f'border-left:4px solid {sc};border-radius:12px;padding:24px;margin-bottom:24px;'
        f'animation:fadeInUp 0.6s ease-out forwards;">'
        f'<div style="font-size:12px;font-weight:800;color:{sc};letter-spacing:.15em;'
        f'text-transform:uppercase;margin-bottom:8px">[{icon}]</div>'
        f'<div style="font-size:24px;font-weight:900;color:{THEME["text_main"]};margin-bottom:8px;">{st}</div>'
        f'<div style="font-size:14px;color:{THEME["text_muted"]};">{desc}</div>'
        f'</div>'
        + _pills_row(
            ("Chi-sq Stat",       f"{chi2_stat:.2f}", THEME["blue"],   0.1),
            ("Critical (α=0.05)", f"{critical:.2f}",  THEME["gold"],   0.2),
            ("p-value",           f"{p_value:.4f}",   THEME["teal"],   0.3),
            ("Analyzed Rows",     str(n),             THEME["purple"], 0.4),
        )
    )

    freq_table = pd.DataFrame({
        "First Digit": digits,
        "Observed %":  [f"{obs_pct[d]:.2f}%" for d in digits],
        "Expected %":  [f"{BENFORD_EXPECTED[d]:.2f}%" for d in digits],
        "Deviation":   [f"{obs_pct[d] - BENFORD_EXPECTED[d]:+.2f}%" for d in digits],
    })
    return fig_bar, result_html, freq_table


# ──────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS MODULE
# ──────────────────────────────────────────────────────────────────
STAT_CHOICES = [
    "Mean", "Median", "Mode", "Std Dev", "Variance",
    "Min", "Max", "Q1 (25th %ile)", "Q3 (75th %ile)", "IQR",
    "Skewness", "Kurtosis",
    "Histogram", "Box Plot", "Violin Plot",
    "Q-Q Plot (vs Normal)", "P-P Plot (vs Normal)",
    "CDF Plot", "Outlier Analysis (IQR)", "Log-scale Histogram",
]


def build_summary_stats(df: pd.DataFrame, selected: list):
    if not selected:
        return "<p style='color:#8F95B2'>Select at least one statistic or plot.</p>", *([None] * 8)

    amounts = df["amount"].dropna().values
    amounts = amounts[amounts > 0]
    n       = len(amounts)

    if n == 0:
        return "<p style='color:#FF6B6B'>No valid amount data found.</p>", *([None] * 8)

    # ── Scalar statistics ────────────────────────────────────────
    scalar_map = {}
    mean_val   = np.mean(amounts)
    std_val    = np.std(amounts, ddof=1)
    q1         = np.percentile(amounts, 25)
    q3         = np.percentile(amounts, 75)

    if "Mean"            in selected: scalar_map["Mean"]            = _fmt_inr(mean_val)
    if "Median"          in selected: scalar_map["Median"]          = _fmt_inr(np.median(amounts))
    if "Mode"            in selected:
        mode_res = stats.mode(amounts, keepdims=True)
        scalar_map["Mode"] = _fmt_inr(float(mode_res.mode[0]))
    if "Std Dev"         in selected: scalar_map["Std Dev"]         = _fmt_inr(std_val)
    if "Variance"        in selected: scalar_map["Variance"]        = _fmt_inr(np.var(amounts, ddof=1))
    if "Min"             in selected: scalar_map["Min"]             = _fmt_inr(np.min(amounts))
    if "Max"             in selected: scalar_map["Max"]             = _fmt_inr(np.max(amounts))
    if "Q1 (25th %ile)"  in selected: scalar_map["Q1 (25th %ile)"] = _fmt_inr(q1)
    if "Q3 (75th %ile)"  in selected: scalar_map["Q3 (75th %ile)"] = _fmt_inr(q3)
    if "IQR"             in selected: scalar_map["IQR"]             = _fmt_inr(q3 - q1)
    if "Skewness"        in selected: scalar_map["Skewness"]        = f"{stats.skew(amounts):.4f}"
    if "Kurtosis"        in selected: scalar_map["Kurtosis"]        = f"{stats.kurtosis(amounts):.4f}"

    # Build KPI HTML
    accent_cycle = [THEME["teal"], THEME["blue"], THEME["gold"],
                    THEME["purple"], THEME["coral"]]
    pill_items   = [(k, v, accent_cycle[i % len(accent_cycle)], 0.05 * i)
                    for i, (k, v) in enumerate(scalar_map.items())]
    kpi_html     = _pills_row(*pill_items) if pill_items else ""

    # Additional normality test
    if len(amounts) >= 3 and ("Q-Q Plot (vs Normal)" in selected or "P-P Plot (vs Normal)" in selected):
        if n <= 5000:
            sw_stat, sw_p = shapiro(amounts[:5000])
        else:
            sw_stat, sw_p = kstest(
                (amounts - mean_val) / std_val, "norm"
            )
        normal_note = (
            f'<div style="margin: 8px 0 16px 0; padding: 10px 16px; border-radius: 8px; '
            f'border-left: 3px solid {THEME["gold"]}; background: {THEME["bg_card"]}; '
            f'color: {THEME["text_muted"]}; font-size: 13px;">'
            f'Normality test: stat={sw_stat:.4f}, p={sw_p:.4f} — '
            f'{"<b>likely normal</b>" if sw_p > 0.05 else "<b>non-normal distribution</b>"}</div>'
        )
        kpi_html += normal_note

    # ── Plots ────────────────────────────────────────────────────
    plot_slots = [None] * 8
    slot_idx   = 0

    def _next_slot(fig):
        nonlocal slot_idx
        if slot_idx < 8:
            plot_slots[slot_idx] = fig
            slot_idx += 1

    # Histogram
    if "Histogram" in selected:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=amounts,
            nbinsx=min(60, max(10, n // 20)),
            marker=dict(color=THEME["teal"], opacity=0.85,
                        line=dict(color=THEME["bg_card"], width=0.5)),
            name="Amount",
        ))
        fig.add_vline(x=mean_val,            line=dict(color=THEME["coral"], dash="dash", width=2),
                      annotation_text="Mean",   annotation_font_color=THEME["coral"])
        fig.add_vline(x=np.median(amounts),  line=dict(color=THEME["gold"],  dash="dot",  width=2),
                      annotation_text="Median", annotation_font_color=THEME["gold"])
        fig.update_layout(**_layout(title="Histogram of Transaction Amounts",
                                    xaxis=dict(title="Amount (₹)"),
                                    yaxis=dict(title="Frequency"), height=400))
        _next_slot(fig)

    # Box Plot
    if "Box Plot" in selected:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=amounts, name="Amounts",
            marker_color=THEME["purple"],
            line=dict(color=THEME["purple"]),
            fillcolor=f"rgba(157,78,221,0.25)",
            boxpoints="outliers",
            jitter=0.3,
            pointpos=-1.8,
        ))
        fig.update_layout(**_layout(title="Box Plot of Transaction Amounts",
                                    yaxis=dict(title="Amount (₹)"), height=400))
        _next_slot(fig)

    # Violin Plot
    if "Violin Plot" in selected:
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=amounts, name="Amounts",
            box_visible=True,
            meanline_visible=True,
            fillcolor=f"rgba(78,205,196,0.3)",
            line_color=THEME["teal"],
        ))
        fig.update_layout(**_layout(title="Violin Plot of Transaction Amounts",
                                    yaxis=dict(title="Amount (₹)"), height=400))
        _next_slot(fig)

    # Log-scale Histogram
    if "Log-scale Histogram" in selected:
        log_amounts = np.log10(amounts)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=log_amounts,
            nbinsx=min(60, max(10, n // 20)),
            marker=dict(color=THEME["gold"], opacity=0.85,
                        line=dict(color=THEME["bg_card"], width=0.5)),
            name="log₁₀(Amount)",
        ))
        fig.update_layout(**_layout(title="Log₁₀-scale Histogram of Amounts",
                                    xaxis=dict(title="log₁₀(Amount)"),
                                    yaxis=dict(title="Frequency"), height=400))
        _next_slot(fig)

    # Q-Q Plot
    if "Q-Q Plot (vs Normal)" in selected:
        sorted_data  = np.sort(amounts)
        probs        = (np.arange(1, n + 1) - 0.5) / n
        theoretical  = norm.ppf(probs, loc=mean_val, scale=std_val)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical, y=sorted_data,
            mode="markers",
            marker=dict(color=THEME["blue"], size=4, opacity=0.6),
            name="Data Quantiles",
        ))
        # 45° reference line
        ref_min = min(theoretical.min(), sorted_data.min())
        ref_max = max(theoretical.max(), sorted_data.max())
        fig.add_trace(go.Scatter(
            x=[ref_min, ref_max], y=[ref_min, ref_max],
            mode="lines",
            line=dict(color=THEME["coral"], dash="dash", width=2),
            name="Normal Reference",
        ))
        fig.update_layout(**_layout(
            title="Q-Q Plot: Amount vs Normal Distribution",
            xaxis=dict(title="Theoretical Quantiles (₹)"),
            yaxis=dict(title="Sample Quantiles (₹)"),
            height=420,
        ))
        _next_slot(fig)

    # P-P Plot
    if "P-P Plot (vs Normal)" in selected:
        sorted_data   = np.sort(amounts)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = norm.cdf(sorted_data, loc=mean_val, scale=std_val)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical_cdf, y=empirical_cdf,
            mode="markers",
            marker=dict(color=THEME["purple"], size=4, opacity=0.6),
            name="Empirical vs Theoretical",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color=THEME["coral"], dash="dash", width=2),
            name="Perfect Normal",
        ))
        fig.update_layout(**_layout(
            title="P-P Plot: Amount vs Normal Distribution",
            xaxis=dict(title="Theoretical CDF", range=[-0.02, 1.02]),
            yaxis=dict(title="Empirical CDF",   range=[-0.02, 1.02]),
            height=420,
        ))
        _next_slot(fig)

    # CDF Plot
    if "CDF Plot" in selected:
        sorted_data   = np.sort(amounts)
        empirical_cdf = np.arange(1, n + 1) / n

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted_data, y=empirical_cdf,
            mode="lines",
            line=dict(color=THEME["teal"], width=2),
            fill="tozeroy",
            fillcolor=f"rgba(78,205,196,0.15)",
            name="Empirical CDF",
        ))
        fig.add_vline(x=mean_val,           line=dict(color=THEME["coral"], dash="dash", width=1.5),
                      annotation_text="Mean")
        fig.add_vline(x=np.median(amounts), line=dict(color=THEME["gold"],  dash="dot",  width=1.5),
                      annotation_text="Median")
        fig.update_layout(**_layout(
            title="Cumulative Distribution Function (CDF)",
            xaxis=dict(title="Amount (₹)"),
            yaxis=dict(title="Cumulative Probability", range=[0, 1.05]),
            height=400,
        ))
        _next_slot(fig)

    # Outlier Analysis
    if "Outlier Analysis (IQR)" in selected:
        lower_fence = q1 - 1.5 * (q3 - q1)
        upper_fence = q3 + 1.5 * (q3 - q1)
        outliers    = amounts[(amounts < lower_fence) | (amounts > upper_fence)]
        normal_pts  = amounts[(amounts >= lower_fence) & (amounts <= upper_fence)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(normal_pts)), y=np.sort(normal_pts),
            mode="markers",
            marker=dict(color=THEME["blue"], size=4, opacity=0.5),
            name=f"Normal ({len(normal_pts):,})",
        ))
        if len(outliers) > 0:
            fig.add_trace(go.Scatter(
                x=np.arange(len(outliers)), y=np.sort(outliers),
                mode="markers",
                marker=dict(color=THEME["coral"], size=7, symbol="x"),
                name=f"Outliers ({len(outliers):,})",
            ))
        fig.add_hline(y=upper_fence, line=dict(color=THEME["gold"], dash="dash", width=1.5),
                      annotation_text=f"Upper Fence ({_fmt_inr(upper_fence)})")
        fig.add_hline(y=lower_fence, line=dict(color=THEME["gold"], dash="dash", width=1.5),
                      annotation_text=f"Lower Fence ({_fmt_inr(lower_fence)})" if lower_fence > 0 else "")
        fig.update_layout(**_layout(
            title=f"Outlier Analysis — IQR Method  ({len(outliers):,} outliers / {n:,} total)",
            xaxis=dict(title="Rank (sorted)"),
            yaxis=dict(title="Amount (₹)"),
            height=420,
        ))
        _next_slot(fig)

    return kpi_html, *plot_slots


# ──────────────────────────────────────────────────────────────────
# GLOBAL STATE & DISPATCHER
# ──────────────────────────────────────────────────────────────────
_CACHE: dict = {}

def run_pipeline(file_obj):
    df, err = load_and_validate(file_obj)
    if err:
        return f"**System Error:** {err}"
    _CACHE["df"] = df
    n_classes = df["class_label"].nunique()
    n_vendors = df["vendor_clean"].nunique()
    return (
        f"✅ **Data Engine Synced.** "
        f"Ingested **{len(df):,}** records · "
        f"**{n_vendors}** vendors · "
        f"**{n_classes}** operational units."
    )

def _guard():
    df = _CACHE.get("df")
    return df, df is None

def get_overview():
    df, missing = _guard()
    if missing: return "Upload dataset to initialize.", None, None, None
    return build_overview(df)

def get_wastage():
    df, missing = _guard()
    if missing: return None, None, pd.DataFrame()
    return build_wastage(df)

def get_network():
    df, missing = _guard()
    if missing: return None, "Upload dataset to initialize."
    return build_network(df)

def get_arm(min_sup, min_conf, min_lift):
    df, missing = _guard()
    if missing: return pd.DataFrame(), None, None, "Upload dataset to initialize."
    return build_arm(df, float(min_sup), float(min_conf), float(min_lift))

def get_benford():
    df, missing = _guard()
    if missing: return None, "Upload dataset to initialize.", pd.DataFrame()
    return build_benford(df)

def get_summary_stats(selected):
    df, missing = _guard()
    if missing:
        return "Upload dataset to initialize.", *([None] * 8)
    return build_summary_stats(df, selected)


# ──────────────────────────────────────────────────────────────────
# FRONT-END  (HTML / CSS)
# ──────────────────────────────────────────────────────────────────
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');

body, .gradio-container {{
    font-family: 'Inter', sans-serif !important;
    background-color: {THEME["bg_app"]} !important;
    color: {THEME["text_main"]} !important;
}}
.tabs {{
    background: transparent !important;
    border: none !important;
}}
.tab-nav {{
    background-color: {THEME["bg_card"]} !important;
    border: 1px solid {THEME["border"]} !important;
    border-radius: 16px !important;
    padding: 8px !important;
    margin-bottom: 30px !important;
    display: flex !important;
    justify-content: space-evenly !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
    gap: 8px !important;
}}
.tab-nav button {{
    background: transparent !important;
    border: none !important;
    color: {THEME["text_muted"]} !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}
.tab-nav button:hover {{
    color: {THEME["text_main"]} !important;
    background: rgba(255,255,255,0.05) !important;
    transform: translateY(-2px);
}}
.tab-nav button.selected {{
    color: {THEME["bg_app"]} !important;
    background: linear-gradient(135deg, {THEME["teal"]}, {THEME["blue"]}) !important;
    box-shadow: 0 4px 15px {THEME["teal"]}60 !important;
    font-weight: 800 !important;
}}
.tabitem {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}}
.kpi-card {{
    background-color: {THEME["bg_card"]};
    border: 1px solid {THEME["border"]};
    border-radius: 16px;
    padding: 24px;
    flex: 1;
    min-width: 200px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}}
@keyframes fadeInUp {{
    0%   {{ opacity: 0; transform: translateY(25px) scale(0.98); }}
    100% {{ opacity: 1; transform: translateY(0)    scale(1);    }}
}}
@keyframes pulseGlow {{
    0%   {{ box-shadow: 0 0 15px {THEME["blue"]}20; }}
    50%  {{ box-shadow: 0 0 25px {THEME["blue"]}50; }}
    100% {{ box-shadow: 0 0 15px {THEME["blue"]}20; }}
}}
.hover-glow {{
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}}
.hover-glow:hover {{
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 15px 35px rgba(0,0,0,0.4), 0 0 20px {THEME["blue"]}30;
    border-color: {THEME["blue"]} !important;
}}
.plot-container {{
    background-color: {THEME["bg_card"]} !important;
    border-radius: 16px !important;
    border: 1px solid {THEME["border"]} !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15) !important;
    overflow: hidden;
    animation: fadeInUp 0.8s ease-out forwards;
}}
button.primary {{
    background: linear-gradient(135deg, {THEME["purple"]}, {THEME["coral"]}) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px {THEME["purple"]}50 !important;
}}
button.primary:hover {{
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 25px {THEME["coral"]}60 !important;
}}
input, select, textarea, .table-wrap {{
    background-color: {THEME["bg_card"]} !important;
    border: 1px solid {THEME["border"]} !important;
    color: {THEME["text_main"]} !important;
    border-radius: 12px !important;
}}
thead tr {{
    background-color: {THEME["bg_app"]} !important;
    color: {THEME["teal"]} !important;
}}
"""

HEADER_HTML = f"""
<div style="text-align:center;margin-bottom:40px;animation:fadeInUp 0.8s ease-out;">
    <h1 style="font-size:42px;font-weight:900;margin-bottom:10px;
        background:linear-gradient(90deg,{THEME["teal"]},{THEME["blue"]},{THEME["purple"]});
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        ENTERPRISE SYSTEMS INTELLIGENCE
    </h1>
    <p style="color:{THEME["text_muted"]};font-size:16px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;">
        Next-Generation Operations &amp; Financial Forensics
    </p>
</div>
"""

# ──────────────────────────────────────────────────────────────────
# GRADIO APPLICATION
# ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Systems Intelligence", css=CSS, theme=gr.themes.Base()) as app:

    gr.HTML(HEADER_HTML)

    with gr.Row(elem_classes="hover-glow"):
        with gr.Column(scale=3):
            file_input = gr.File(
                label="Initialize Data Engine (Upload CSV)",
                file_types=[".csv"],
                type="filepath",
            )
        with gr.Column(scale=2):
            upload_status = gr.Markdown(
                f"<div style='padding:20px;color:{THEME['text_muted']};border:1px dashed {THEME['border']};"
                f"border-radius:12px;text-align:center;margin-top:25px;'>"
                "System Standby. Awaiting data ingestion...</div>"
            )

    with gr.Tabs():

        # ── 1. OVERVIEW ──────────────────────────────────────────
        with gr.Tab("📊 Financial Overview"):
            overview_btn = gr.Button("⚡ Generate Executive Summary", variant="primary")
            kpi_out      = gr.HTML()
            with gr.Row():
                fig_mo_out  = gr.Plot()
                fig_pie_out = gr.Plot()
            fig_vend_out = gr.Plot()
            overview_btn.click(get_overview, outputs=[kpi_out, fig_mo_out, fig_pie_out, fig_vend_out])

        # ── 2. RISK ──────────────────────────────────────────────
        with gr.Tab("♻️ Risk & Inefficiency"):
            wastage_btn  = gr.Button("⚡ Execute Risk Modeling", variant="primary")
            fig_wt_out   = gr.Plot()
            fig_cmp_out  = gr.Plot()
            gr.Markdown(f"### <span style='color:{THEME['coral']}'>Critical High-Risk Operational Days</span>")
            hr_table_out = gr.Dataframe(wrap=True)
            wastage_btn.click(get_wastage, outputs=[fig_wt_out, fig_cmp_out, hr_table_out])

        # ── 3. NETWORK ───────────────────────────────────────────
        with gr.Tab("🕸️ Topology Network"):
            net_btn       = gr.Button("⚡ Render Supply Chain Topology", variant="primary")
            net_stats_out = gr.HTML()
            fig_net_out   = gr.Plot()
            net_btn.click(get_network, outputs=[fig_net_out, net_stats_out])

        # ── 4. ASSOCIATION RULES ──────────────────────────────────
        with gr.Tab("🔗 Association Intelligence"):
            gr.Markdown(
                f"<div style='color:{THEME['text_muted']};margin-bottom:20px;'>"
                "<b>FP-Growth Engine:</b> Uncovers latent administrative batching and structural "
                "supply chain coupling. Default thresholds: Support 0.05 · Confidence 0.60 · Lift 1.0."
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    sup_s   = gr.Slider(0.01, 0.30, value=0.05, step=0.01, label="Minimum Support Threshold")
                    conf_s  = gr.Slider(0.30, 1.00, value=0.60, step=0.05, label="Minimum Confidence Threshold")
                    lift_s  = gr.Slider(1.00, 5.00, value=1.00, step=0.10, label="Minimum Lift Multiplier")
                    arm_btn = gr.Button("⚡ Mine Structural Logic", variant="primary")
                with gr.Column(scale=3):
                    arm_stats_out = gr.HTML()
            with gr.Row():
                fig_freq_out  = gr.Plot()
                fig_rules_out = gr.Plot()
            gr.Markdown(f"### <span style='color:{THEME['teal']}'>Discovered Structural Constraints</span>")
            arm_table_out = gr.Dataframe(wrap=True)
            arm_btn.click(get_arm, inputs=[sup_s, conf_s, lift_s],
                          outputs=[arm_table_out, fig_freq_out, fig_rules_out, arm_stats_out])

        # ── 5. BENFORD ────────────────────────────────────────────
        with gr.Tab("⚖️ Forensic Audit (Benford)"):
            ben_btn        = gr.Button("⚡ Execute Benford Verification", variant="primary")
            ben_result_out = gr.HTML()
            fig_ben_bar    = gr.Plot()
            gr.Markdown(f"### <span style='color:{THEME['purple']}'>Distribution Ledger</span>")
            ben_table_out  = gr.Dataframe()
            ben_btn.click(get_benford, outputs=[fig_ben_bar, ben_result_out, ben_table_out])

        # ── 6. SUMMARY STATISTICS ─────────────────────────────────
        with gr.Tab("📐 Summary Statistics"):
            gr.Markdown(
                f"<div style='color:{THEME['text_muted']};margin-bottom:20px;'>"
                "Select any combination of scalar statistics and distribution plots to analyse the "
                "<b>Amount</b> column. All plots are rendered with interactive Plotly."
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    stat_selector = gr.CheckboxGroup(
                        choices=STAT_CHOICES,
                        value=["Mean", "Median", "Std Dev", "Min", "Max", "Histogram"],
                        label="📋 Choose Statistics & Plots",
                    )
                    stats_btn = gr.Button("⚡ Compute & Visualize", variant="primary")

            stats_kpi_out = gr.HTML()

            # 8 plot slots — shown only when populated
            with gr.Row():
                sp0 = gr.Plot(visible=True)
                sp1 = gr.Plot(visible=True)
            with gr.Row():
                sp2 = gr.Plot(visible=True)
                sp3 = gr.Plot(visible=True)
            with gr.Row():
                sp4 = gr.Plot(visible=True)
                sp5 = gr.Plot(visible=True)
            with gr.Row():
                sp6 = gr.Plot(visible=True)
                sp7 = gr.Plot(visible=True)

            stats_btn.click(
                fn=get_summary_stats,
                inputs=[stat_selector],
                outputs=[stats_kpi_out, sp0, sp1, sp2, sp3, sp4, sp5, sp6, sp7],
            )

    # Trigger
    file_input.change(fn=run_pipeline, inputs=[file_input], outputs=[upload_status])

if __name__ == "__main__":
    #app.launch(server_name="0.0.0.0", server_port=1234, share=False)
    app.launch()
