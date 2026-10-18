"""
Component 3 - Function 1: graph construction + node-level disruption risk prediction
Dataset: cleaned DataCo (run c3_clean.py first -> dataco_clean.csv)

Run:  python c3_pipeline.py --csv dataco_clean.csv --no-gnn                 (simulated labels, default)
      python c3_pipeline.py --csv dataco_clean.csv --label-mode real --no-gnn (old real-label experiment)

Outputs: nodes.csv, edges.csv (used by c3_cascade.py), model_comparison.csv

Graph (a PROXY - DataCo has no real supplier/manufacturer/distributor IDs):
    Tier 1 Category (supply source) -> Tier 2 Country (distribution point) -> Tier 3 Region (hub)

Label modes
  real      : label = node late rate in a LATER time window (features from an earlier window).
              Result on DataCo: no persistent signal, models score ~chance.
  simulated : the graph + features are real; the risk label comes from simulated disruption events
              (TAF fallback, same idea as Kayalvizhi et al. 2025). See simulate_labels().
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

SPECS = [("Category", "Category Name"), ("Country", "Order Country"), ("Region", "Order Region")]
EDGE_SPECS = [("Category Name", "Order Country", "Category", "Country"),
              ("Order Country", "Order Region", "Country", "Region")]
NUM_COLS = ["orders", "late_rate", "avg_delay", "avg_qty", "avg_sales",
            "avg_discount", "avg_profit_ratio"]


# ---------------------------------------------------------------- load
def load(path, enc="utf-8"):
    df = pd.read_csv(path, encoding=enc)               # cleaned file is UTF-8 (raw DataCo needs latin-1)
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    df["delay"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
    df["late"] = df["Late_delivery_risk"].astype(int)    # 1 = late
    return df


# ------------------------------------------------------- node features
def node_features(df, col, ntype, modes):
    g = df.groupby(col)
    out = pd.DataFrame({
        "orders": g.size(),
        "late_rate": g["late"].mean(),
        "avg_delay": g["delay"].mean(),
        "avg_qty": g["Order Item Quantity"].mean(),
        "avg_sales": g["Sales"].mean(),
        "avg_discount": g["Order Item Discount Rate"].mean(),
        "avg_profit_ratio": g["Order Item Profit Ratio"].mean(),
    })
    share = pd.crosstab(df[col], df["Shipping Mode"], normalize="index")
    share = share.reindex(columns=modes, fill_value=0.0)
    share.columns = [f"mode_{m}" for m in modes]
    out = out.join(share)
    out.index = [f"{ntype}::{i}" for i in out.index]      # string ids, e.g. Country::Brazil
    out["ntype"] = ntype
    return out


# ------------------------------------------------------- graph structure
def build_structure(df, min_node_orders, min_edge_orders):
    modes = sorted(df["Shipping Mode"].unique())
    nodes = pd.concat([node_features(df, c, t, modes) for t, c in SPECS])
    nodes = nodes[nodes["orders"] >= min_node_orders]    # drop tiny nodes (too few orders to trust)

    rows = []
    for a, b, ta, tb in EDGE_SPECS:                      # shipment lanes
        g = df.groupby([a, b]).agg(orders=("late", "size"), late_rate=("late", "mean"),
                                   avg_delay=("delay", "mean")).reset_index()
        g["src"] = ta + "::" + g[a].astype(str)
        g["dst"] = tb + "::" + g[b].astype(str)
        # dependency of dst on src = share of ALL of dst's incoming orders (computed BEFORE the
        # lane filter, so a small node is not wrongly shown as 100% dependent on one lane)
        g["dependency"] = g["orders"] / g.groupby("dst")["orders"].transform("sum")
        g = g[g["orders"] >= min_edge_orders]
        rows.append(g[["src", "dst", "orders", "late_rate", "avg_delay", "dependency"]])
    edges = pd.concat(rows)
    edges = edges[edges["src"].isin(nodes.index) & edges["dst"].isin(nodes.index)]
    return nodes, edges.reset_index(drop=True)


def feature_matrix(nodes):
    X = nodes[NUM_COLS + [c for c in nodes.columns if c.startswith("mode_")]].fillna(0.0)
    X = X.join(pd.get_dummies(nodes["ntype"]).astype(float))
    return X


# ------------------------------------------------------------ labels
def real_labels(nodes, df_lab, min_label_orders, q):
    """Old experiment: label = FUTURE late rate (above the q-quantile of nodes = high risk)."""
    labs = []
    for t, c in SPECS:
        g = df_lab.groupby(c)["late"].agg(future_late_rate="mean", future_orders="size")
        g.index = [f"{t}::{i}" for i in g.index]
        labs.append(g)
    lab = nodes.join(pd.concat(labs))
    ok = lab["future_orders"].fillna(0) >= min_label_orders
    thr = lab.loc[ok, "future_late_rate"].quantile(q)
    y = np.where(ok, (lab["future_late_rate"] >= thr).astype(float), np.nan)
    return y, lab["future_late_rate"].to_numpy(), thr


def simulate_labels(nodes, edges, n_events, spread_scale, q, seed):
    """
    Simulated disruption events on the REAL graph (assumptions - state them in the report):
      * an event starts at one node; nodes with a higher late rate are more likely starting points
      * severity s ~ U(0.3, 1.0), duration t ~ U(1, 30) days  (ranges from Kayalvizhi et al. 2025)
      * the failure spreads along directed lanes (independent cascade) with probability
            p = dependency * lane_late_rate * spread_scale * severity
      * every failed node suffers impact = s * t / 30
    vulnerability = average impact per event. Within each tier, nodes at/above the q-quantile are labelled 1.
    """
    rng = np.random.default_rng(seed)
    ids = list(nodes.index)
    idx = {n: i for i, n in enumerate(ids)}
    out = [[] for _ in ids]
    for s, d, dep, lr in zip(edges["src"], edges["dst"], edges["dependency"], edges["late_rate"]):
        out[idx[s]].append((idx[d], dep * lr))
    w = nodes["late_rate"].to_numpy(dtype=float)
    w = w / w.sum()
    vuln = np.zeros(len(ids))
    sizes = []
    for _ in range(n_events):
        start = rng.choice(len(ids), p=w)
        sev, dur = rng.uniform(0.3, 1.0), rng.uniform(1, 30)
        failed, frontier = {start}, [start]
        while frontier:
            nxt = []
            for u in frontier:
                for v, p in out[u]:
                    if v not in failed and rng.random() < min(1.0, p * spread_scale * sev):
                        failed.add(v); nxt.append(v)
            frontier = nxt
        for v in failed:
            vuln[v] += sev * dur / 30.0
        sizes.append(len(failed))
    vuln /= n_events
    # label WITHIN each tier: vulnerable = at/above the q-quantile of its own node type.
    # (A global threshold made the label mostly a function of the tier: Category 0%, Region 100%.)
    y = np.zeros(len(ids))
    for t in nodes["ntype"].unique():
        m = (nodes["ntype"] == t).to_numpy()
        y[m] = (vuln[m] >= np.quantile(vuln[m], q)).astype(float)
    return y, vuln, float(np.quantile(vuln, q)), float(np.mean(sizes))


# ------------------------------------------------------------ models
def scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred), precision=p, recall=r, f1=f)


def neighbour_mean(X, edges, node_ids):
    """Network-enhanced ML baseline: each node's features + the mean features of its neighbours."""
    idx = {n: i for i, n in enumerate(node_ids)}
    A = np.zeros((len(node_ids), len(node_ids)))
    for s, d in zip(edges["src"], edges["dst"]):
        A[idx[s], idx[d]] = A[idx[d], idx[s]] = 1.0
    deg = np.maximum(A.sum(1, keepdims=True), 1.0)
    return np.hstack([X, (A @ X) / deg])


def run_sklearn(X, XN, y, tr, te, seed):
    res = {}
    for tag, M in [("", X), ("NbrMean+", XN)]:
        sc = StandardScaler().fit(M[tr])
        Ms = sc.transform(M)
        for name, m in [("LogisticRegression", LogisticRegression(max_iter=2000)),
                        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=seed))]:
            m.fit(Ms[tr], y[tr])
            res[tag + name] = scores(y[te], m.predict(Ms[te]))
    return res


def run_gnns(X, y, edges, node_ids, tr, va, te, seed, epochs=200):
    """GCN / GraphSAGE / GAT with PyTorch Geometric. Needs torch + torch_geometric (UNTESTED by the author)."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
    torch.manual_seed(seed)

    idx = {n: i for i, n in enumerate(node_ids)}
    src = [idx[s] for s in edges["src"]]
    dst = [idx[d] for d in edges["dst"]]
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)   # both directions for message passing

    sc = StandardScaler().fit(X[tr])
    x = torch.tensor(sc.transform(X), dtype=torch.float)
    yt = torch.tensor(np.nan_to_num(y, nan=0).astype(int))
    tr_t, va_t, te_t = (torch.tensor(m, dtype=torch.long) for m in (tr, va, te))

    class Net(torch.nn.Module):
        def __init__(self, kind, d_in, d_h=32):
            super().__init__()
            if kind == "GraphSAGE":
                self.c1, self.c2 = SAGEConv(d_in, d_h), SAGEConv(d_h, d_h)
            elif kind == "GCN":
                self.c1, self.c2 = GCNConv(d_in, d_h), GCNConv(d_h, d_h)
            else:  # GAT, 4 heads x 8 dims = 32
                self.c1 = GATConv(d_in, d_h // 4, heads=4)
                self.c2 = GATConv(d_h, d_h // 4, heads=4)
            self.lin = torch.nn.Linear(d_h, 2)

        def forward(self, x, ei):
            x = F.dropout(F.relu(self.c1(x, ei)), 0.3, self.training)
            x = F.relu(self.c2(x, ei))
            return self.lin(x)

    res, models = {}, {}
    for kind in ["GCN", "GraphSAGE", "GAT"]:
        model = Net(kind, x.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        best_f1, best_state = -1, None
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            loss = F.cross_entropy(model(x, ei)[tr_t], yt[tr_t]); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(x, ei).argmax(1)
            f1 = scores(yt[va_t].numpy(), pred[va_t].numpy())["f1"]
            if f1 > best_f1:                              # model selection on VALIDATION only
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pred = model(x, ei).argmax(1)
        res[kind] = scores(yt[te_t].numpy(), pred[te_t].numpy())
        models[kind] = model
    return res, models, x, ei


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataco_clean.csv from c3_clean.py")
    ap.add_argument("--encoding", default="utf-8", help="utf-8 for the cleaned file, latin-1 for the raw file")
    ap.add_argument("--label-mode", choices=["simulated", "real"], default="simulated")
    ap.add_argument("--min-node-orders", type=int, default=None, help="default 100 (simulated) / 0 (real)")
    ap.add_argument("--min-edge-orders", type=int, default=30)
    ap.add_argument("--label-quantile", type=float, default=0.5)
    ap.add_argument("--cutoff", type=float, default=0.7, help="real mode: time quantile splitting feature/label windows")
    ap.add_argument("--min-label-orders", type=int, default=30, help="real mode only")
    ap.add_argument("--events", type=int, default=3000, help="simulated mode: number of disruption events")
    ap.add_argument("--spread-scale", type=float, default=6.0, help="simulated mode: strength of cascade spread")
    ap.add_argument("--repeats", type=int, default=2, help="repeats of 5-fold CV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-gnn", action="store_true", help="skip GNNs (no torch installed)")
    a = ap.parse_args()
    if a.min_node_orders is None:
        a.min_node_orders = 100 if a.label_mode == "simulated" else 0

    df = load(a.csv, a.encoding)
    if a.label_mode == "real":
        cut = df["order_date"].quantile(a.cutoff)
        df_feat, df_lab = df[df["order_date"] <= cut], df[df["order_date"] > cut]
        nodes, edges = build_structure(df_feat, a.min_node_orders, a.min_edge_orders)
        y, score, thr = real_labels(nodes, df_lab, a.min_label_orders, a.label_quantile)
        extra = ""
    else:
        nodes, edges = build_structure(df, a.min_node_orders, a.min_edge_orders)
        y, score, thr, mean_size = simulate_labels(nodes, edges, a.events, a.spread_scale, a.label_quantile, a.seed)
        extra = f" | mean nodes failed per event: {mean_size:.2f}"
    nodes["label_score"] = score
    nodes["y"] = y

    X = feature_matrix(nodes)
    lab = np.where(~np.isnan(y))[0]
    print(f"mode: {a.label_mode} | orders used: {len(df)}")
    print(f"nodes: {len(nodes)} (labelled {len(lab)}) | edges: {len(edges)} | label threshold: {thr:.4f}{extra}")
    print("class balance (labelled):", np.bincount(y[lab].astype(int)))

    Xn = X.to_numpy()
    XN = neighbour_mean(Xn, edges, list(nodes.index))
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=a.seed)
    gnn_ok = not a.no_gnn
    for k, (tr_all, te_l) in enumerate(cv.split(lab, y[lab])):
        tr_all, te_i = lab[tr_all], lab[te_l]
        tr_i, va_i = train_test_split(tr_all, test_size=0.18, stratify=y[tr_all], random_state=a.seed + k)
        res = run_sklearn(Xn, XN, y, tr_i, te_i, a.seed)
        if gnn_ok:
            try:
                gres, _, _, _ = run_gnns(Xn, y, edges, list(nodes.index), tr_i, va_i, te_i, a.seed + k)
                res.update(gres)
            except ImportError:
                print("torch / torch_geometric not installed - GNNs skipped")
                gnn_ok = False
        for m, s in res.items():
            rows.append(dict(model=m, fold=k, **s))

    r = pd.DataFrame(rows)
    table = r.groupby("model")[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"]).round(3)
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    order = [m for m in ["LogisticRegression", "RandomForest", "NbrMean+LogisticRegression", "NbrMean+RandomForest",
                         "GCN", "GraphSAGE", "GAT"] if m in table.index]
    table = table.loc[order]
    print(f"\n5-fold CV x {a.repeats} repeats ({r['fold'].nunique()} test folds), mean and std\n", table.to_string())

    nodes.to_csv("nodes.csv"); edges.to_csv("edges.csv", index=False); table.to_csv("model_comparison.csv")
    print("\nsaved nodes.csv, edges.csv, model_comparison.csv")


if __name__ == "__main__":
    main()