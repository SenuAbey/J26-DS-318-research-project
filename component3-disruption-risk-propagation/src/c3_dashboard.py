"""
Component 3 - demo dashboard: Supply Chain Risk Propagation (Streamlit + Plotly)

Run from the component folder (venv active):
    pip install streamlit plotly
    streamlit run src\\c3_dashboard.py

Single page, no menu. It only DISPLAYS what your scripts produced in the results folder
(nodes.csv, edges.csv, and optionally model_comparison.csv, mcnemar_results.csv, sensitivity_table.csv,
explanations.csv, cleaning_report.txt) and runs the real cascade code from c3_cascade.py live.
Every number on screen is computed from those files - nothing is hard-coded.

Needs the updated c3_cascade.py (simulate() now takes a severity argument).
Optional: set the environment variable C3_RESULTS to read a different results folder.
"""
import html
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c3_cascade import load_graph, simulate  # noqa: E402

RESULTS = Path(os.environ.get("C3_RESULTS", SRC.parent / "results"))
st.set_page_config(page_title="Component 3 - Supply Chain Risk Propagation", layout="wide",
                   initial_sidebar_state="collapsed")

MIN_P = 0.10                                              # below this chance an entity is not called "likely" to be reached
SEVERITY = {"Low": 0.4, "Medium": 0.65, "High": 0.9}      # inside the 0.3-1.0 range used for the labels
DTYPES = ["Supplier Delay", "Factory Disruption", "Logistics Delay", "Demand Shock"]
ROW_Y = {"Category": 1.0, "Country": 0.5, "Region": 0.0}
TAG = {"Category": "CAT", "Country": "CTY", "Region": "REG"}
TIER_NOTE = {"Category": "Category (supply-source proxy)", "Country": "Country (distribution-point proxy)",
             "Region": "Region (regional-hub proxy)"}
LEVEL = {"Low": ("#2e9e6b", "#eaf7f0"), "Medium": ("#d99a1b", "#fff6e3"), "High": ("#c0392b", "#fdeceb")}

st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display:none;}
.badge {background:#111827;color:#fff;font-family:monospace;font-size:12px;padding:3px 9px;border-radius:5px;font-weight:700;}
.tag {background:#fff4dd;color:#9a6a00;font-family:monospace;font-size:11px;padding:3px 9px;border-radius:12px;border:1px solid #f0d9a6;margin-left:6px;}
.comp {color:#3b5bdb;font-size:12px;font-weight:700;letter-spacing:.08em;margin-top:14px;}
.pill {display:inline-block;font-size:12px;padding:3px 10px;border-radius:12px;font-weight:600;}
.pill.on {background:#fdeceb;color:#c0392b;border:1px solid #f3c1bc;} .pill.off {background:#eaf7f0;color:#2e9e6b;border:1px solid #bfe3d0;}
.grid2 {display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.tile {background:#f7f8fa;border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;}
.tile.red {background:#fdeceb;border-color:#f3c1bc;}
.tl {font-size:10.5px;color:#6b7280;font-weight:700;letter-spacing:.04em;text-transform:uppercase;}
.tile.red .tl {color:#c0392b;}
.tv {font-size:20px;font-weight:700;color:#111827;margin-top:2px;} .tile.red .tv {color:#c0392b;}
.ts {font-size:11.5px;color:#6b7280;margin-top:2px;}
.risk {float:right;font-size:12px;font-weight:700;padding:2px 10px;border-radius:12px;}
.kv {display:flex;justify-content:space-between;border-bottom:1px solid #f0f1f3;padding:5px 0;font-size:14px;} .kv span:first-child{color:#6b7280;}
.card-sub {color:#6b7280;font-size:13px;margin-top:-6px;margin-bottom:8px;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def esc(s):
    return html.escape(str(s), quote=False)


def tier(n):
    return n.split("::", 1)[0]


def short(n, width=22):
    s = n.split("::", 1)[1]
    return s if len(s) <= width else s[: width - 1] + "..."


def pretty(n):
    return f"{tier(n)}: {n.split('::', 1)[1]}"


def level(score):
    return "Low" if score < 33 else ("Medium" if score < 67 else "High")


@st.cache_data
def read_csv(path, **kw):
    return pd.read_csv(path, **kw)


def get(name, **kw):
    p = RESULTS / name
    return read_csv(str(p), **kw) if p.exists() else None


@st.cache_resource
def get_graph(scale, stamp):
    return load_graph(str(RESULTS / "nodes.csv"), str(RESULTS / "edges.csv"), scale)


nodes, edges = get("nodes.csv", index_col=0), get("edges.csv")
if nodes is None or edges is None:
    st.error(f"nodes.csv / edges.csv not found in {RESULTS}. Run c3_pipeline.py from the results folder first.")
    st.stop()
STAMP = os.path.getmtime(RESULTS / "nodes.csv")           # refresh the cached graph if the files change


@st.cache_data
def derive(nodes, edges):
    """Per-entity figures shown in the detail panels (all computed from nodes.csv / edges.csv)."""
    m = pd.DataFrame(index=nodes.index)
    m["tier"] = nodes["ntype"]
    inc = edges.groupby("dst")
    m["n_in"] = inc.size().reindex(m.index).fillna(0).astype(int)
    m["max_dep"] = inc["dependency"].max().reindex(m.index)
    m["alt_orders"] = inc["orders"].sum().reindex(m.index) - inc["orders"].max().reindex(m.index)
    m["vol_share"] = nodes["orders"] / nodes.groupby("ntype")["orders"].transform("sum")
    m["reliability"] = 1 - nodes["late_rate"]
    m["avg_delay"] = nodes["avg_delay"]
    base = nodes["label_score"] if "label_score" in nodes else nodes["late_rate"]
    m["base_risk"] = (base.groupby(nodes["ntype"]).rank(pct=True) * 100).round()
    for col, name, low in [("max_dep", "hi_dep", False), ("vol_share", "hi_vol", False),
                           ("avg_delay", "hi_delay", False), ("alt_orders", "lo_alt", True)]:
        thr = m.groupby("tier")[col].transform(lambda s: s.quantile(0.25 if low else 0.75))
        m[name] = (m[col] <= thr) if low else (m[col] >= thr)
    return m


M = derive(nodes, edges)
ALL = list(nodes.index)

# ---------------------------------------------------------- state + callbacks
cats = nodes[nodes["ntype"] == "Category"]
for k, v in dict(origin=cats["orders"].idxmax() if len(cats) else ALL[0], sev="High", dtype=DTYPES[0],
                 scale=6.0, nsims=1000, topk=8, sim=None).items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("inspect", st.session_state["origin"])


def run_sim():
    o, label = st.session_state["origin"], st.session_state["sev"]
    scale = st.session_state["scale"]
    _, _, out = get_graph(scale, STAMP)
    prob, path = simulate(out, o, n_sims=st.session_state["nsims"], seed=0, severity=SEVERITY[label])
    st.session_state["sim"] = dict(origin=o, sev_label=label, dtype=st.session_state["dtype"], scale=scale,
                                   nsims=st.session_state["nsims"], prob=prob.astype(float), path=path)
    st.session_state["inspect"] = o


def clear_sim():
    st.session_state["sim"] = None


def on_click():
    """Clicking a node in the chart selects it for the detail panels."""
    try:
        pts = st.session_state["netchart"]["selection"]["points"]
        if pts:
            cd = pts[0].get("customdata")
            node = cd[0] if isinstance(cd, (list, tuple)) else cd
            if node in nodes.index:
                st.session_state["inspect"] = node
    except Exception:
        pass


sim = st.session_state["sim"]
origin = sim["origin"] if sim else st.session_state["origin"]


def risk_of(n):
    """Number shown on a node: % chance this disruption reaches it (active) or baseline percentile (idle)."""
    if sim:
        return 100 if n == sim["origin"] else int(round(100 * float(sim["prob"].get(n, 0.0))))
    return int(M.loc[n, "base_risk"])


def hops(n):
    d, cur = 0, n
    while cur != sim["origin"] and d < 12:
        cur, d = sim["path"].get(cur), d + 1
        if cur is None:
            return None
    return d


def focus_nodes(k):
    """Entities drawn in the focused view."""
    if sim and not sim["prob"].empty:
        prob, shown = sim["prob"], {origin}
        for t in ("Country", "Region"):                      # most-affected entities in each tier
            ids = [n for n in prob.index if tier(n) == t]
            for n in prob[ids].sort_values(ascending=False).head(max(2, k // 2)).index:
                while n != origin and n not in shown:        # add the chain back to the failed entity
                    shown.add(n)
                    n = sim["path"].get(n, origin)
        return [origin] + [n for n in shown if n != origin]
    ctx = edges[(edges["src"] == origin) | (edges["dst"] == origin)].sort_values("orders", ascending=False)
    others = [d if s == origin else s for s, d in zip(ctx["src"], ctx["dst"])]
    return [origin] + others[:k]


def build_fig(view):
    risk = {n: risk_of(n) for n in view}
    pos = {}
    for t in ROW_Y:
        lst = sorted([n for n in view if tier(n) == t], key=lambda n: -risk[n])
        for i, n in enumerate(lst):
            x = 0.5 if len(lst) == 1 else 0.1 + 0.8 * i / (len(lst) - 1)
            offs = [0.0] if len(lst) <= 4 else ([0.09, -0.09] if len(lst) <= 8 else [0.14, 0.0, -0.14])   # stagger crowded tiers
            pos[n] = (x, ROW_Y[t] + offs[i % len(offs)])
    path_edges = set()
    if sim:
        path_edges = {(u, v) for v, u in sim["path"].items() if v in pos and u in pos and risk.get(v, 0) > 0}
    fig = go.Figure()
    for e in edges[edges["src"].isin(pos) & edges["dst"].isin(pos)].itertuples():
        on = (e.src, e.dst) in path_edges
        (x0, y0), (x1, y1) = pos[e.src], pos[e.dst]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(width=min(7, 2 + 8 * e.dependency) if on else min(4, 1 + 6 * e.dependency),
                                           color="#c0392b" if on else "#c9ced6")))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"{100 * e.dependency:.0f}%", showarrow=False,
                           font=dict(size=9, color="#c0392b" if on else "#8b93a1"), bgcolor="rgba(255,255,255,0.85)")
    hover = []
    for n in view:
        lvl = level(risk[n])
        border, fill = LEVEL[lvl]
        orig = n == origin
        fig.add_annotation(
            x=pos[n][0], y=pos[n][1], showarrow=False, align="center", borderpad=5,
            text=f"<b>{esc(short(n))}</b><br><span style='font-size:10px'>{TAG[tier(n)]} - risk {risk[n]}</span>",
            bgcolor=fill, bordercolor="#2f4fd8" if orig else border, borderwidth=3 if orig else 1.5,
            font=dict(size=12, color="#1f2937"))
        hover.append(f"<b>{esc(n.split('::', 1)[1])}</b><br>{TIER_NOTE[tier(n)]}<br>risk number: {risk[n]} ({lvl})"
                     f"<br>orders: {int(nodes.loc[n, 'orders']):,}<br>late rate: {nodes.loc[n, 'late_rate']:.2f}")
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in view], y=[pos[n][1] for n in view], mode="markers",
                             marker=dict(size=48, color="rgba(0,0,0,0)"), customdata=view, text=hover,
                             hoverinfo="text", showlegend=False))
    for t, y in ROW_Y.items():
        fig.add_annotation(xref="paper", x=0.0, xanchor="left", y=y + 0.27, text=TIER_NOTE[t].upper(), showarrow=False,
                           font=dict(size=10, color="#9ca3af"))
    fig.update_layout(height=560, margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor="white", showlegend=False,
                      xaxis=dict(visible=False, range=[-0.04, 1.04]), yaxis=dict(visible=False, range=[-0.30, 1.42]))
    return fig, pos


def tile(label, value, sub="", red=False):
    return (f'<div class="tile{" red" if red else ""}"><div class="tl">{esc(label)}</div>'
            f'<div class="tv">{value}</div><div class="ts">{sub}</div></div>')


# ------------------------------------------------------------------- header
st.markdown('<span class="badge">J26-DS-318</span><span class="tag">REAL DATACO GRAPH - SIMULATED DISRUPTIONS</span>',
            unsafe_allow_html=True)
st.markdown("**Explainable AI Framework for Stochastic Supply Chain Decision Intelligence**")
st.markdown('<div class="comp">COMPONENT 3</div>', unsafe_allow_html=True)
st.title("Supply Chain Risk Propagation")
st.write("Understanding how disruptions can propagate through connected entities.")
st.caption("Research prototype. The graph is built from real DataCo orders, and the risk numbers come from "
           "simulated disruption events ")

left, right = st.columns([2, 1], gap="medium")

# ------------------------------------------------------------ network card
with left, st.container(border=True):
    h1, h2 = st.columns([3, 1])
    h1.markdown("#### Focused supply chain network")
    h1.markdown('<div class="card-sub">Category -> Country -> Region: the failed entity and its most-affected neighbours</div>',
                unsafe_allow_html=True)
    h2.markdown(f'<div style="text-align:right"><span class="pill {"on" if sim else "off"}">'
                f'{("Disruption active: " + esc(short(sim["origin"], 18)) + " - " + sim["sev_label"]) if sim else "No active disruption"}'
                f'</span></div>', unsafe_allow_html=True)
    view = focus_nodes(st.session_state["topk"])
    fig, pos = build_fig(view)
    st.plotly_chart(fig, key="netchart", on_select=on_click, selection_mode="points",
                    config={"displayModeBar": False}, width="stretch")
    st.caption("Border colour = level of the number on the node (Low < 33, Medium 33-66, High >= 67). Idle: the number is the "
               "entity's baseline simulated-risk percentile within its tier. After a simulation: the % chance the disruption "
               "reaches it. Thick red lane = most likely propagation path; lane labels = dependency (share of the destination's "
               "incoming orders). Blue border = failed entity. Click a node to inspect it.")
    

# ------------------------------------------------------- controls + detail
with right:
    with st.container(border=True):
        st.markdown("#### Disruption controls")
        st.markdown('<div class="card-sub">Configure and run an illustrative disruption</div>', unsafe_allow_html=True)
        st.selectbox("Disruption type (illustrative label only)", DTYPES, key="dtype")
        st.caption("Label only: the model has no disruption types, so this does not change the simulation.")
        st.radio("Severity", list(SEVERITY), key="sev", horizontal=True)
        st.selectbox("Affected entity", ALL, key="origin", format_func=pretty)
        b1, b2 = st.columns(2)
        b1.button("Simulate disruption", type="primary", on_click=run_sim, width="stretch")
        b2.button("Clear", on_click=clear_sim, width="stretch")
        with st.expander("Assumptions (advanced)"):
            st.slider("Spread scale", 1.0, 12.0, step=0.5, key="scale")
            st.slider("Simulations", 200, 5000, step=100, key="nsims")
            st.slider("Focused view size (higher = more entities)", 4, 12, key="topk")
            st.caption(f"Severity values: {', '.join(f'{k} = {v}' for k, v in SEVERITY.items())} (the label simulation "
                       "draws severity between 0.3 and 1.0). The spread scale is an assumption - see Model validation below.")

    with st.container(border=True):
        sel = st.session_state["inspect"]
        r = int(M.loc[sel, "base_risk"]) if not sim else (100 if sel == sim["origin"] else int(round(100 * float(sim["prob"].get(sel, 0.0)))))
        border, fill = LEVEL[level(r)]
        st.markdown(f"#### Entity detail <span class='risk' style='background:{fill};color:{border}'>{level(r)}</span>",
                    unsafe_allow_html=True)
        st.selectbox("Click a node above or pick an entity", ALL, key="inspect", format_func=pretty)
        t = tier(sel)
        geo = ("n/a (product line)" if t == "Category" else short(sel, 40) if t == "Region" else
               ", ".join(short(d, 30) for d in edges[edges["src"] == sel]["dst"] if d.startswith("Region::")) or "n/a")
        depth = "-" if not sim else ("0 hops (origin)" if sel == sim["origin"] else
                                     (f"{hops(sel)} hop(s) from origin" if hops(sel) is not None else "not reached"))
        rows = [("Entity", esc(short(sel, 40))), ("Entity type", TIER_NOTE[t]),
                ("Reliability (1 - late rate)", f"{M.loc[sel, 'reliability']:.0%}"),
                ("Order volume", f"{int(nodes.loc[sel, 'orders']):,} orders"),
                ("Dependency level", "n/a (source)" if M.loc[sel, "n_in"] == 0 else f"{M.loc[sel, 'max_dep']:.0%} of inbound flow"),
                ("Simulated risk level", f"{level(r)} ({r}/100)"), ("Geographic exposure", esc(geo)),
                ("Propagation depth", depth)]
        st.markdown("".join(f'<div class="kv"><span>{a}</span><span><b>{b}</b></span></div>' for a, b in rows),
                    unsafe_allow_html=True)

# ------------------------------------------------- impact + why-at-risk row
c_imp, c_why = st.columns(2, gap="medium")
with c_imp, st.container(border=True):
    st.markdown("#### Impact summary")
    if not sim:
        st.markdown('<div class="card-sub">Run a simulation to see the impact</div>', unsafe_allow_html=True)
        st.markdown('<div class="grid2">' + tile("Entities likely affected", "-") + tile("Propagation depth", "-")
                    + tile("Most exposed entity", "-") + tile("Estimated service impact", "-") + "</div>",
                    unsafe_allow_html=True)
    else:
        prob, o = sim["prob"], sim["origin"]
        N, expected = len(nodes), float(prob.sum())
        likely = prob[prob >= 0.5].index.tolist()
        names = ", ".join(short(n, 18) for n in likely[:5]) + (" ..." if len(likely) > 5 else "")
        reach = prob[prob >= 0.10].index.tolist()
        reach_down = [n for n in reach if n != o]
        depth_v = max([hops(n) or 0 for n in reach], default=0)
        tiers_v = len({tier(n) for n in reach} | {tier(o)})
        down = prob.drop(o, errors="ignore")
        top_n = down.idxmax() if not down.empty else None
        if tier(o) == "Region":
            exposed, base = float(nodes.loc[o, "orders"]), float(nodes.loc[o, "orders"]) / max(1e-9, M.loc[o, "vol_share"])
        else:
            cty = nodes[nodes["ntype"] == "Country"]["orders"]
            exposed = sum(float(prob.get(c, 0.0)) * v for c, v in cty.items()) + (float(cty.get(o, 0.0)) if tier(o) == "Country" else 0.0)
            base = float(cty.sum())
        st.markdown(f'<div class="card-sub">{esc(sim["dtype"])} (illustrative label) at {esc(pretty(o))}, '
                    f'{sim["sev_label"].lower()} severity</div>', unsafe_allow_html=True)
        st.markdown('<div class="grid2">'
                    + tile("Entities likely affected", f"~{expected:.1f} of {N}",
                           f"expected over {sim['nsims']} runs; &gt;=50% chance: {esc(names) if likely else 'none'}")
                    + (tile("Propagation depth", f"{depth_v} hop(s)", f"{tiers_v} tier(s) reached (entities with &gt;=10% chance)") if reach_down
                       else tile("Propagation depth", "No spread", "no other entity above a 10% chance"))
                    + tile("Most exposed entity", esc(short(top_n, 20)) if top_n else "none",
                           (f"{100 * float(down[top_n]):.0f}% chance of being reached" + ("" if float(down[top_n]) >= MIN_P else " (low)")) if top_n else "no downstream lanes",
                           red=bool(top_n) and float(down[top_n]) >= MIN_P)
                    + tile("Estimated service impact", f"{exposed / base:.1%}",
                           f"~{exposed:,.0f} delivery orders exposed (expected, proxy)") + "</div>", unsafe_allow_html=True)
        st.caption("Computed from the simulation. Service impact = expected share of delivery orders at affected countries "
                   "(a proxy, not measured lost sales).")

with c_why, st.container(border=True):
    sel = st.session_state["inspect"]
    st.markdown("#### Why is this entity at risk?")
    st.markdown(f'<div class="card-sub">{esc(pretty(sel))} - structural risk factors</div>', unsafe_allow_html=True)
    s = M.loc[sel]
    src_node = s["n_in"] == 0
    dep_txt = "n/a (source)" if src_node else f"{s['max_dep']:.0%} of inbound flow"
    red_ = "Low" if (not src_node and s["max_dep"] >= 0.35) else ("Medium" if (not src_node and s["max_dep"] >= 0.15) else "High")
    r_now = risk_of(sel)
    st.markdown('<div class="grid2">'
                + tile("Dependency level", dep_txt, "largest single inbound lane", red=bool(s["hi_dep"]) and not src_node)
                + tile("Order volume share", f"{s['vol_share']:.1%}", f"of all {tier(sel)} orders", red=bool(s["hi_vol"]))
                + tile("Delay vs schedule", f"{s['avg_delay']:+.2f} days", "average, real minus scheduled", red=bool(s["hi_delay"]))
                + tile("Redundancy", "n/a (source)" if src_node else red_, "" if src_node else f"{int(s['n_in'])} inbound lanes",
                       red=(not src_node and red_ == "Low"))
                + tile("Alternative inbound orders", "n/a (source)" if src_node else f"{s['alt_orders']:,.0f}",
                       "orders arriving via other lanes", red=bool(s["lo_alt"]) and not src_node)
                + tile("Disruption exposure", f"{r_now}/100", "chance this disruption reaches it" if sim else "baseline percentile",
                       red=r_now >= 67) + "</div>", unsafe_allow_html=True)
    st.caption("Red = highest quartile within its tier (lowest for alternative orders), Low redundancy (largest lane >= 35%), "
               "or exposure >= 67. Redundancy: High < 15%, Medium 15-35%. All values are proxies derived from DataCo orders.")
    ex = get("explanations.csv")
    if ex is not None and not ex[ex["node"] == sel].empty:
        sub = ex[ex["node"] == sel].sort_values("rank").head(3)
        st.markdown("**GNNExplainer - lanes the GAT relied on most:** " + "; ".join(
            f"{esc(short(a, 18))} -> {esc(short(b, 18))} ({v:.2f})" for a, b, v in zip(sub["src"], sub["dst"], sub["importance"])))

# ------------------------------------------------------ explainable decision
st.markdown("### Explainable decision - risk stage")
st.caption("Auto-generated from the current simulation and validation files.")
if not sim:
    st.info("Run a simulation to generate the Decision, Uncertainty and Risk cards.")
else:
    mc = get("model_comparison.csv", index_col=0)
    if mc is not None:
        top2 = mc["f1_mean"].sort_values(ascending=False)
        gap, spread = top2.iloc[0] - top2.iloc[1], float(mc.loc[top2.index[:2], "f1_std"].max())
        model_txt = (f"The model comparison shows no clear winner (best {top2.index[0]}, F1 {top2.iloc[0]:.2f}; gap to "
                     f"runner-up {gap:.3f} is within the fold spread {spread:.3f}).") if gap < spread else \
            f"In the model comparison {top2.index[0]} scored highest (F1 {top2.iloc[0]:.2f})."
    else:
        model_txt = "Validation files were not found, so model performance is not summarised here."
    strong = down[down >= MIN_P].sort_values(ascending=False) if top_n else down
    if top_n and len(strong):
        lane = edges[edges["dst"] == strong.index[0]].sort_values("dependency", ascending=False).head(1)
        lane_txt = (f" {short(strong.index[0], 24)}'s largest inbound lane comes from {short(lane['src'].iloc[0], 24)} "
                    f"({lane['dependency'].iloc[0]:.0%} of its orders)." if len(lane) else "")
        decision = (f"Prioritise buffer stock and backup lanes for {short(strong.index[0], 24)}"
                    + (f" and {short(strong.index[1], 24)}" if len(strong) > 1 else "")
                    + f", the entities a failure at {short(o, 24)} is most likely to reach "
                      f"({100 * float(strong.iloc[0]):.0f}% chance for the first).{lane_txt}")
    elif top_n:
        decision = (f"Impact is small: no downstream entity has more than a {MIN_P:.0%} chance of being reached "
                    f"(highest: {short(top_n, 24)}, {100 * float(down[top_n]):.0f}%). Try a higher-impact entity, or review "
                    f"{short(o, 24)}'s own inbound lanes.")
    else:
        decision = (f"A failure at {short(o, 24)} reaches no downstream entities in this model, so the impact stays local; "
                    "review its inbound lanes and upstream suppliers instead.")
    cards = [("Decision", "What should be done?", decision),
             ("Uncertainty", "What is uncertain?",
              f"Spread strength is an assumption (scale {sim['scale']:g}, {sim['sev_label'].lower()} severity); figures are "
              f"averages over {sim['nsims']} simulated events and the risk labels are simulated. {model_txt}"),
             ("Risk", "What could go wrong?",
              f"A failure at {short(o, 24)} is expected to reach ~{expected:.1f} of {N} entities ({100 * expected / N:.0f}% of "
              + (f"the network) across {tiers_v} tier(s)" if reach_down else f"the network), with no other entity above a {MIN_P:.0%} chance")
              + f", exposing about {exposed / base:.1%} of delivery orders . ")]
    for col, (title, q, body) in zip(st.columns(3, gap="medium"), cards):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(q)
            st.write(body)

# ------------------------------------------------ collapsed sections at the bottom
with st.expander("Model validation (optional)", expanded=False):
    mc = get("model_comparison.csv", index_col=0)
    if mc is None:
        st.info("model_comparison.csv not found. Create it with c3_pipeline.py.")
    else:
        st.markdown("**Model comparison - cross-validated F1 (mean and std over folds)**")
        figv = go.Figure(go.Bar(x=mc.index, y=mc["f1_mean"], error_y=dict(type="data", array=mc["f1_std"]),
                                marker_color=["#c0392b" if m_.startswith(("GCN", "GraphSAGE", "GAT")) else "#9ca3af" for m_ in mc.index]))
        figv.update_layout(height=340, yaxis_title="F1", margin=dict(t=10))
        st.plotly_chart(figv, width="stretch")
        st.dataframe(mc, width="stretch")
    mn = get("mcnemar_results.csv")
    if mn is not None:
        first = mn[mn["repeat"] == mn["repeat"].min()]
        st.markdown("**McNemar's test (first CV repeat)**")
        st.dataframe(first[["ref", "other", "n_nodes", "ref_only_correct", "other_only_correct", "p_value"]].round(4),
                     hide_index=True, width="stretch")
        st.caption(f"Bonferroni threshold = 0.05 / {len(first)} = {0.05 / max(len(first), 1):.4f}. A non-significant result "
                   "means no evidence of a difference, not that the models are equal.")
    ss = get("sensitivity_table.csv", index_col=0)
    if ss is not None:
        st.markdown("**Spread-scale sensitivity**")
        st.dataframe(ss, width="stretch")
    rep = RESULTS / "cleaning_report.txt"
    if rep.exists():
        st.markdown("**Data cleaning report**")
        st.text(rep.read_text())

with st.expander("About the data and its limits", expanded=False):
    st.markdown(
        "- **Graph:** built from real DataCo orders : Category (supply source) -> Country (distribution point) "
        "-> Region (hub). DataCo has no real supplier / manufacturer / distributor IDs.\n"
        "- **Risk labels are simulated.** In DataCo, a node's late rate in an early period did not predict its late rate "
        "later, so labels come from simulated disruption events on the real graph (the fallback in the TAF, as in "
        "Kayalvizhi et al. 2025).\n"
        "- **Spread scale and severity are assumptions.** Results depend on them (see the sensitivity table).\n"
        "- **Dependency** = share of a node's incoming orders that come through a lane. Reliability, redundancy and "
        "service impact are proxies derived from orders, not measured values.\n"
        "- **Disruption type** is a label only; the model does not distinguish types.\n")