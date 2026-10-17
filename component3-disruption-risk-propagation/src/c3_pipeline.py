"""
Component 3 - Function 1: Graph construction + node-level disruption risk prediction
Dataset: DataCo Smart Supply Chain (DataCoSupplyChainDataset.csv)

Run:  python c3_clean.py --csv DataCoSupplyChainDataset.csv   (first)
      python c3_pipeline.py --csv dataco_clean.csv
Outputs: nodes.csv, edges.csv (used by c3_cascade.py) and a metrics table.

Graph design (a PROXY - DataCo has no real supplier/manufacturer/distributor IDs):
    Tier 1  Category  (product line, stands in for the supply source)
    Tier 2  Country   (order country, stands in for the distribution point)
    Tier 3  Region    (order region, stands in for the regional hub)
    Edges   Category -> Country (shipment lanes), Country -> Region
Leakage control: features come from an EARLY time window, labels from a LATER one.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

SPECS = [("Category", "Category Name"), ("Country", "Order Country"), ("Region", "Order Region")]
EDGE_SPECS = [("Category Name", "Order Country", "Category", "Country"),
              ("Order Country", "Order Region", "Country", "Region")]
NUM_COLS = ["orders", "late_rate", "avg_delay", "avg_qty", "avg_sales",
            "avg_discount", "avg_profit_ratio"]


# ---------------------------------------------------------------- Step 1: load
def load(path, enc="utf-8"):
    df = pd.read_csv(path, encoding=enc)               # cleaned file is UTF-8 (raw DataCo needs latin-1)
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    df["delay"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
    df["late"] = df["Late_delivery_risk"].astype(int)    # 1 = late
    return df


# ------------------------------------------------------- Step 2: node features
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


# ------------------------------------------------- Step 3: graph + labels
def build_graph(df_feat, df_lab, min_edge_orders, min_label_orders, label_q):
    modes = sorted(df_feat["Shipping Mode"].unique())
    nodes = pd.concat([node_features(df_feat, c, t, modes) for t, c in SPECS])

    labs = []
    for t, c in SPECS:                                   # label = FUTURE late rate
        g = df_lab.groupby(c)["late"].agg(future_late_rate="mean", future_orders="size")
        g.index = [f"{t}::{i}" for i in g.index]
        labs.append(g)
    nodes = nodes.join(pd.concat(labs))

    ok = nodes["future_orders"].fillna(0) >= min_label_orders
    thr = nodes.loc[ok, "future_late_rate"].quantile(label_q)
    nodes["y"] = np.where(ok, (nodes["future_late_rate"] >= thr).astype(float), np.nan)

    rows = []
    for a, b, ta, tb in EDGE_SPECS:                      # shipment lanes
        g = df_feat.groupby([a, b]).agg(orders=("late", "size"),
                                        late_rate=("late", "mean"),
                                        avg_delay=("delay", "mean")).reset_index()
        g = g[g["orders"] >= min_edge_orders]
        g["src"] = ta + "::" + g[a].astype(str)
        g["dst"] = tb + "::" + g[b].astype(str)
        rows.append(g[["src", "dst", "orders", "late_rate", "avg_delay"]])
    edges = pd.concat(rows)
    edges = edges[edges["src"].isin(nodes.index) & edges["dst"].isin(nodes.index)]
    # dependency of dst on src = share of dst's incoming orders coming from src
    edges["dependency"] = edges["orders"] / edges.groupby("dst")["orders"].transform("sum")
    return nodes, edges.reset_index(drop=True), thr


def feature_matrix(nodes):
    X = nodes[NUM_COLS + [c for c in nodes.columns if c.startswith("mode_")]].fillna(0.0)
    X = X.join(pd.get_dummies(nodes["ntype"]).astype(float))
    return X


# ----------------------------------------------- Step 4: non-graph baselines
def scores(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred), precision=p, recall=r, f1=f)


def run_baselines(X, y, tr, va, te, seed):
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    models = {"LogisticRegression": LogisticRegression(max_iter=1000),
              "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed)}
    res = {}
    for name, m in models.items():
        m.fit(Xs[tr], y[tr])
        res[name] = scores(y[te], m.predict(Xs[te]))
    # 5-fold CV on all labelled nodes (for the validation plan)
    lab = np.where(~np.isnan(y))[0]
    cv = {}
    for name in models:
        f1s = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(lab, y[lab]):
            sc2 = StandardScaler().fit(X[lab[a]])
            mm = type(models[name])(**models[name].get_params())
            mm.fit(sc2.transform(X[lab[a]]), y[lab[a]])
            f1s.append(scores(y[lab[b]], mm.predict(sc2.transform(X[lab[b]])))["f1"])
        cv[name] = (np.mean(f1s), np.std(f1s))
    return res, cv


# ------------------------------------- Step 5: GNNs (needs torch + PyG; UNTESTED here)
def run_gnns(X, y, edges, node_ids, tr, va, te, seed, epochs=200):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
    torch.manual_seed(seed)

    idx = {n: i for i, n in enumerate(node_ids)}
    src = [idx[s] for s in edges["src"]]
    dst = [idx[d] for d in edges["dst"]]
    # message passing in both directions (cascade DIRECTION is handled in c3_cascade.py)
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)

    sc = StandardScaler().fit(X[tr])
    x = torch.tensor(sc.transform(X), dtype=torch.float)
    yt = torch.tensor(np.nan_to_num(y, nan=0).astype(int))
    tr_t, va_t, te_t = (torch.tensor(m) for m in (tr, va, te))

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
    ap.add_argument("--csv", required=True, help="use dataco_clean.csv from c3_clean.py")
    ap.add_argument("--encoding", default="utf-8", help="utf-8 for the cleaned file, latin-1 for the raw file")
    ap.add_argument("--cutoff", type=float, default=0.7, help="time quantile splitting feature/label windows")
    ap.add_argument("--min-edge-orders", type=int, default=30)
    ap.add_argument("--min-label-orders", type=int, default=30)
    ap.add_argument("--label-quantile", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-gnn", action="store_true", help="skip GNNs (no torch installed)")
    a = ap.parse_args()

    df = load(a.csv, a.encoding)
    cut = df["order_date"].quantile(a.cutoff)
    df_feat, df_lab = df[df["order_date"] <= cut], df[df["order_date"] > cut]
    print(f"orders: {len(df)} | feature window: {len(df_feat)} | label window: {len(df_lab)}")

    nodes, edges, thr = build_graph(df_feat, df_lab, a.min_edge_orders, a.min_label_orders, a.label_quantile)
    X = feature_matrix(nodes)
    y = nodes["y"].to_numpy()
    lab = np.where(~np.isnan(y))[0]
    print(f"nodes: {len(nodes)} (labelled {len(lab)}) | edges: {len(edges)} | high-risk threshold: {thr:.3f}")
    print("class balance (labelled):", np.bincount(y[lab].astype(int)))

    tr_i, tmp = train_test_split(lab, test_size=0.30, stratify=y[lab], random_state=a.seed)
    va_i, te_i = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=a.seed)
    n = len(nodes)
    tr, va, te = (np.isin(np.arange(n), s) for s in (tr_i, va_i, te_i))

    res, cv = run_baselines(X.to_numpy(), y, tr, va, te, a.seed)
    if not a.no_gnn:
        try:
            gres, _, _, _ = run_gnns(X.to_numpy(), y, edges, list(nodes.index), tr, va, te, a.seed)
            res.update(gres)
        except ImportError:
            print("torch / torch_geometric not installed - GNNs skipped")

    table = pd.DataFrame(res).T.round(4)
    print("\nTest-set results\n", table.to_string())
    print("\n5-fold CV F1 (baselines): ",
          {k: f"{m:.3f} +/- {s:.3f}" for k, (m, s) in cv.items()})

    nodes.to_csv("nodes.csv"); edges.to_csv("edges.csv", index=False)
    table.to_csv("model_comparison.csv")
    print("\nsaved nodes.csv, edges.csv, model_comparison.csv")


if __name__ == "__main__":
    main()
