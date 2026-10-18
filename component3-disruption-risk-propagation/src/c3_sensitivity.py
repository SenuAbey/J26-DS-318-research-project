"""
Component 3 - validation: spread-scale sensitivity table

Run (from the results folder, AFTER running c3_pipeline.py once per scale with --out-dir sens_scaleN):
    python ..\\src\\c3_sensitivity.py

Reads sens_scale*\\model_comparison.csv and prints one table: mean F1 (+/- std over the CV folds) for every model
at every spread scale. Shows WHERE the graph helps and where it does not, i.e. how much your conclusion
depends on the simulation assumption. Saves sensitivity_table.csv.
"""
import glob
import re
import pandas as pd

ORDER = ["LogisticRegression", "RandomForest", "NbrMean+LogisticRegression", "NbrMean+RandomForest",
         "GCN", "GraphSAGE", "GAT"]


def main():
    files = glob.glob("sens_scale*/model_comparison.csv")
    if not files:
        raise SystemExit("no sens_scale*/model_comparison.csv found - run the pipeline with --out-dir sens_scale3 etc. first")
    cols, raw = {}, {}
    for f in files:
        scale = float(re.search(r"sens_scale([0-9.]+)", f.replace("\\", "/")).group(1))
        t = pd.read_csv(f, index_col=0)
        cols[scale] = t.apply(lambda r: f"{r['f1_mean']:.3f} +/- {r['f1_std']:.3f}", axis=1)
        raw[scale] = t["f1_mean"]
    scales = sorted(cols)
    table = pd.DataFrame({f"scale {s:g}": cols[s] for s in scales})
    table = table.reindex([m for m in ORDER if m in table.index])
    print("F1 (mean +/- std over CV folds) by spread scale\n")
    print(table.to_string())

    num = pd.DataFrame({f"scale {s:g}": raw[s] for s in scales}).reindex(table.index)
    if "RandomForest" in num.index and "NbrMean+RandomForest" in num.index:
        gain = num.loc["NbrMean+RandomForest"] - num.loc["RandomForest"]
        print("\nGain from neighbour information (NbrMean+RandomForest minus RandomForest):")
        print(gain.round(3).to_string())
    table.to_csv("sensitivity_table.csv")
    print("\nsaved sensitivity_table.csv")


if __name__ == "__main__":
    main()
