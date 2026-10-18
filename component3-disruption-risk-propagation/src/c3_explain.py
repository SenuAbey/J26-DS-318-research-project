"""
Component 3 - Function 2 (part B): GNNExplainer on the trained GAT

Run (from the results folder, AFTER c3_pipeline.py has written nodes.csv / edges.csv):
    python ..\\src\\c3_explain.py                         # explains the most vulnerable node of each tier
    python ..\\src\\c3_explain.py --node "Country::Brazil"
    python ..\\src\\c3_explain.py --per-tier 2 --epochs 500

What it does
  1. trains GCN / GraphSAGE / GAT once on a stratified split (same code as the pipeline)
  2. takes the trained GAT and asks GNNExplainer, for each chosen node:
        - which shipment LANES (edges) most influenced the prediction
        - which input FEATURES mattered most
  3. prints the result and saves explanations.csv

NOTE: needs torch + torch_geometric. The post-processing was tested; the GNNExplainer call itself follows
the PyTorch Geometric >= 2.3 API from memory and has NOT been run by the author - check the PyG docs
if your installed version complains.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from c3_pipeline import feature_matrix, run_gnns


def top_lanes(edge_mask, edges, k=5):
    """edge_mask covers [forward edges, reverse edges]; add both directions back into one value per lane."""
    e = len(edges)
    imp = np.asarray(edge_mask[:e]) + np.asarray(edge_mask[e:2 * e])
    out = edges.assign(importance=imp).sort_values("importance", ascending=False).head(k)
    return out[["src", "dst", "importance", "dependency", "late_rate"]].reset_index(drop=True)


def top_features(node_mask_row, names, k=5):
    imp = np.abs(np.asarray(node_mask_row))                   # relevance of the explained node's OWN features
    order = np.argsort(imp)[::-1][:k]
    return [(names[i], float(imp[i])) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", help='node id, e.g. "Country::Brazil"')
    ap.add_argument("--per-tier", type=int, default=1, help="explain the N most vulnerable nodes of EACH tier")
    ap.add_argument("--epochs", type=int, default=300, help="GNNExplainer optimisation epochs")
    ap.add_argument("--lr", type=float, default=0.01, help="GNNExplainer learning rate (try 0.05 if importances look flat)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import torch
    from torch_geometric.explain import Explainer, GNNExplainer

    nodes = pd.read_csv("nodes.csv", index_col=0)
    edges = pd.read_csv("edges.csv")
    fm = feature_matrix(nodes)
    X, names, y = fm.to_numpy(), list(fm.columns), nodes["y"].to_numpy()
    ids = list(nodes.index)

    # stratified train / validation / test split, then train the three GNNs once
    idx_all = np.arange(len(ids))
    tr_all, te = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=a.seed)
    tr, va = train_test_split(tr_all, test_size=0.18, stratify=y[tr_all], random_state=a.seed)
    res, models, x, ei = run_gnns(X, y, edges, ids, tr, va, te, a.seed)
    print("GNN test results (single split):")
    print(pd.DataFrame(res).T.round(3).to_string(), "\n")

    model = models["GAT"]
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=a.epochs, lr=a.lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )

    if a.node:
        targets = [a.node]
    else:
        srt = nodes.sort_values("label_score", ascending=False)
        targets = [n for t in ["Category", "Country", "Region"] for n in srt[srt["ntype"] == t].index[:a.per_tier]]
    rows = []
    for name in targets:
        if name not in ids:
            print(f"{name} not found in nodes.csv (check spelling / quotes)")
            continue
        i = ids.index(name)
        with torch.no_grad():
            prob = torch.softmax(model(x, ei)[i], dim=-1).numpy()
        exp = explainer(x, ei, index=i)
        lanes = top_lanes(exp.edge_mask.detach().cpu().numpy(), edges)
        feats = top_features(exp.node_mask.detach().cpu().numpy()[i], names)

        print(f"=== {name} | true label {int(y[i])} | model P(vulnerable) = {prob[1]:.2f}")
        print("Most influential lanes:")
        print(lanes.round(3).to_string(index=False))
        spread = (lanes["importance"].max() - lanes["importance"].min()) / max(lanes["importance"].max(), 1e-9)
        if spread < 0.25:
            print(f"WARNING: top-5 lane importances differ by only {100 * spread:.0f}% - explanation is weak; "
                  "try --epochs 500 or --lr 0.05")
        print("Most influential features:", ", ".join(f"{n} ({v:.2f})" for n, v in feats), "\n")
        for r, row in lanes.iterrows():
            rows.append(dict(node=name, rank=r + 1, src=row["src"], dst=row["dst"],
                             importance=row["importance"], dependency=row["dependency"], late_rate=row["late_rate"]))

    pd.DataFrame(rows).to_csv("explanations.csv", index=False)
    print("saved explanations.csv")


if __name__ == "__main__":
    main()