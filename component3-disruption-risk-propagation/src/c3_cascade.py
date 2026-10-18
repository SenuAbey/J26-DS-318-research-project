"""
Component 3 - Function 2 (part A): disruption cascade simulation (no torch needed)

Reads nodes.csv / edges.csv written by c3_pipeline.py, fails one node and simulates
how the failure spreads along DIRECTED shipment lanes (independent-cascade model).

    p(src -> dst) = dependency(dst on src) * late_rate(lane) * spread_scale   [a proxy, document it]
    (use the SAME --spread-scale as c3_pipeline.py, default 6)

Run:  python c3_cascade.py --start "Category::Cleats"
      python c3_cascade.py --rank            # which nodes cause the biggest cascades
"""
import argparse
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


def load_graph(nodes_path="nodes.csv", edges_path="edges.csv", spread_scale=6.0):
    nodes = pd.read_csv(nodes_path, index_col=0)
    edges = pd.read_csv(edges_path)
    edges["p"] = (edges["dependency"] * edges["late_rate"] * spread_scale).clip(0, 1)
    out = defaultdict(list)
    for s, d, p in zip(edges["src"], edges["dst"], edges["p"]):
        out[s].append((d, p))
    return nodes, edges, out


def simulate(out, start, n_sims=2000, seed=0):
    rng = np.random.default_rng(seed)
    hit, parent = Counter(), defaultdict(Counter)
    for _ in range(n_sims):
        failed, frontier = {start}, [start]
        while frontier:
            nxt = []
            for u in frontier:
                for v, p in out.get(u, []):
                    if v not in failed and rng.random() < p:
                        failed.add(v); nxt.append(v); parent[v][u] += 1
            frontier = nxt
        hit.update(failed - {start})
    prob = pd.Series({k: v / n_sims for k, v in hit.items()}).sort_values(ascending=False)
    path = {v: parent[v].most_common(1)[0][0] for v in prob.index}   # most frequent infector
    return prob, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--spread-scale", type=float, default=6.0)
    a = ap.parse_args()
    nodes, edges, out = load_graph(spread_scale=a.spread_scale)
    total = len(nodes)

    if a.rank:
        rows = []
        for s in out:
            prob, _ = simulate(out, s, n_sims=500)
            rows.append((s, prob.sum(), prob.sum() / total))
        r = pd.DataFrame(rows, columns=["node", "expected_nodes_affected", "share_of_network"])
        print(r.sort_values("expected_nodes_affected", ascending=False).head(10).round(3).to_string(index=False))
        return

    prob, path = simulate(out, a.start, a.sims)
    print(f"Failure at {a.start}: expected nodes affected = {prob.sum():.2f} "
          f"({100 * prob.sum() / total:.1f}% of {total} nodes)")
    df = pd.DataFrame({"infection_prob": prob.round(3), "most_likely_via": pd.Series(path)})
    print(df.head(15).to_string())
    df.to_csv("cascade_result.csv")


if __name__ == "__main__":
    main()