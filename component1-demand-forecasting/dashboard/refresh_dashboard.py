"""
refresh_dashboard.py - build the working dashboard (dashboard.html) from the template and the latest results.

  dashboard_template.html  (design, contains the placeholder __DATA__)
+ results/dashboard_data.json   (written by src/09_export_dashboard_data.py)
= dashboard.html            (one self-contained file: double-click to open, no server needed)

Run from the component1-demand-forecasting folder:
    python dashboard/refresh_dashboard.py            # build
    python dashboard/refresh_dashboard.py --open     # build and open in the browser
"""
import json
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # .../component1-demand-forecasting/dashboard
ROOT = HERE.parent
TEMPLATE = HERE / "dashboard_template.html"
DATA = ROOT / "results" / "dashboard_data.json"
OUT = HERE / "dashboard.html"
PLACEHOLDER = "__DATA__"


def build():
    for p, hint in ((TEMPLATE, "keep dashboard_template.html in the dashboard folder"),
                    (DATA, "run python src/09_export_dashboard_data.py first")):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found ({hint}).")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        sys.exit(f"ERROR: the template must contain {PLACEHOLDER} exactly once (found {template.count(PLACEHOLDER)}).")
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for key in ("weeks", "products", "conformal_summary", "decision_main", "shap_groups", "model_comparison"):
        if not data.get(key):
            sys.exit(f"ERROR: '{key}' is missing or empty in dashboard_data.json. Re-run the earlier scripts (02-08), then 09.")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")      # safe inside a <script> tag
    OUT.write_text(template.replace(PLACEHOLDER, payload), encoding="utf-8")
    print(f"dashboard.html built | {len(data['products'])} products | {OUT.stat().st_size / 1e6:.2f} MB | data generated {data.get('generated')}")
    return OUT


if __name__ == "__main__":
    out = build()
    if "--open" in sys.argv:
        webbrowser.open(out.as_uri())
