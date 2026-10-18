"""
Component 3 - validation: McNemar's test on paired predictions

Run (from the results folder, AFTER c3_pipeline.py has written predictions.csv):
    python ..\\src\\c3_mcnemar.py                       # compares GAT against every other model
    python ..\\src\\c3_mcnemar.py --ref GraphSAGE
    python ..\\src\\c3_mcnemar.py --pred sens_scale3\\predictions.csv

Idea: in 5-fold cross-validation every node is tested exactly once per repeat, so two models can be compared
node by node. For a pair (A, B) count the nodes that
    b = A right, B wrong         c = A wrong, B right
McNemar's test asks whether b and c differ more than chance allows (exact two-sided binomial test, better than
the chi-square version when counts are small). A small p-value = the two models really make different errors.
It says NOTHING about which is better - look at b vs c (and the F1 table) for the direction.

Each repeat of the CV is a separate paired comparison on the same 166 nodes, so the repeats are NOT independent;
report repeat 0 as the main test and the other repeats as a stability check.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def mcnemar_exact(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum())     # reference right, other wrong
    c = int((~a_ok & b_ok).sum())     # reference wrong, other right
    n = b + c
    p = 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5).pvalue)
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions.csv")
    ap.add_argument("--ref", default="GAT", help="reference model to compare the others against")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    df = pd.read_csv(a.pred)
    models = [c[5:] for c in df.columns if c.startswith("pred_")]
    if a.ref not in models:
        raise SystemExit(f"reference model '{a.ref}' not in predictions.csv (available: {models})")
    others = [m for m in models if m != a.ref]

    rows = []
    for rep, g in df.groupby("repeat"):
        ref_ok = (g["pred_" + a.ref] == g["y_true"]).to_numpy()
        for m in others:
            ok = (g["pred_" + m] == g["y_true"]).to_numpy()
            b, c, p = mcnemar_exact(ref_ok, ok)
            rows.append(dict(ref=a.ref, other=m, repeat=int(rep), n_nodes=len(g),
                             ref_only_correct=b, other_only_correct=c, p_value=p))
    res = pd.DataFrame(rows)
    res.to_csv("mcnemar_results.csv", index=False)

    k = len(others)
    print(f"McNemar (exact) - reference model: {a.ref} | {k} comparisons -> Bonferroni threshold {a.alpha / k:.4f}\n")
    main_rep = res[res["repeat"] == res["repeat"].min()].copy()
    main_rep["verdict"] = np.where(main_rep["p_value"] < a.alpha / k, "different (significant)",
                                   np.where(main_rep["p_value"] < a.alpha, "borderline (not after correction)",
                                            "no evidence of a difference"))
    print("Main test (first CV repeat):")
    print(main_rep[["other", "n_nodes", "ref_only_correct", "other_only_correct", "p_value", "verdict"]]
          .round(4).to_string(index=False))
    stab = res.groupby("other")["p_value"].agg(["min", "median", "max"]).round(4)
    print("\nStability across repeats (p-value):")
    print(stab.to_string())
    print("\nsaved mcnemar_results.csv")


if __name__ == "__main__":
    main()
