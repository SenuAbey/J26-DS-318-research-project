"""
01_baselines.py - baseline forecasting models (Component 1)
Models: Naive, Seasonal naive, Moving average, ARIMA, LightGBM quantile regression,
        LSTM (point) and MC-Dropout LSTM (intervals)
Task:   forecast weekly sales 1-4 weeks ahead (rolling origin) on the calibration and test weeks.
Run from the component1-demand-forecasting folder:  python src/01_baselines.py
Needs:  pip install statsmodels lightgbm torch matplotlib pyarrow
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------- SETTINGS ----------------
HORIZON = 4                    # forecast 1..4 weeks ahead
TRAIN_END, CAL_END = 193, 235  # same split as 01_preprocess.py
QUICK = False                  # True = 20 series and few epochs (fast test of the code)
RUN_ARIMA, RUN_LGBM, RUN_NEURAL = True, True, True
NEURAL_MODELS = ("LSTM", "TCN")     # deep-learning baselines (needs torch)
QUANTILES = (0.025, 0.5, 0.975)    # 95% interval + median
LSTM_WINDOW = 26               # weeks of history given to the LSTM
MC_SAMPLES = 100               # MC-Dropout samples
SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# ---------------- LOAD DATA ----------------
w = pd.read_parquet(ROOT / "data" / "processed" / "m5_weekly_ca1.parquet")
series = np.sort(w.series_id.unique())
if QUICK:
    series = series[:20]
w = w[w.series_id.isin(series)].sort_values(["series_id", "time_idx"])
S, T = len(series), int(w.time_idx.max()) + 1

def mat(col):
    """(series x week) matrix of a column"""
    return w.pivot(index="series_id", columns="time_idx", values=col).loc[series].to_numpy(float)

Y, PR, PM, SN, EV = mat("sales"), mat("price_ratio"), mat("promo_flag"), mat("snap_days"), mat("has_event")
cat_name = w.groupby("series_id").cat_id.first().loc[series].to_numpy()
cats = np.sort(np.unique(cat_name))
cat_code = np.searchsorted(cats, cat_name)
t_vec = w.drop_duplicates("time_idx").sort_values("time_idx")
WOY = t_vec.week_of_year.to_numpy(float)
MONTH = pd.to_datetime(t_vec.week_start).dt.month.to_numpy(float)

mu = Y[:, :TRAIN_END + 1].mean(1)            # series level from TRAINING weeks only
mu[mu == 0] = 1.0
print(f"Series: {S} | weeks: {T} | train <= {TRAIN_END}, calib <= {CAL_END}, test after")

# ---------------- EVALUATION GRID (rolling origin) ----------------
# an origin o = last week we know; we forecast weeks o+1 ... o+4
O = np.arange(TRAIN_END, T - 1)               # origins 193 ... T-2
NO = len(O)
TGT = O[:, None] + np.arange(1, HORIZON + 1)[None, :]      # (NO, 4) target week index
VALID = TGT <= T - 1
Yt = np.full((S, NO, HORIZON), np.nan)
for i in range(NO):
    for h in range(HORIZON):
        if VALID[i, h]:
            Yt[:, i, h] = Y[:, TGT[i, h]]
SPLIT = np.where(TGT <= CAL_END, "calib", "test")          # split by TARGET week (no overlap in time)
MASK = np.broadcast_to(VALID[None], Yt.shape)

results = {}      # name -> (P, lo, hi) cubes of shape (S, NO, 4); lo/hi None for point models

def new_cube():
    return np.full((S, NO, HORIZON), np.nan)

def timeit(name):
    print(f"\n--- {name} ---")
    return time.time()

# ---------------- 1. SIMPLE BASELINES ----------------
t0 = timeit("Naive / Seasonal naive / Moving average")
P_naive, P_snaive, P_ma = new_cube(), new_cube(), new_cube()
for i, o in enumerate(O):
    for h in range(HORIZON):
        t = TGT[i, h]
        P_naive[:, i, h] = Y[:, o]                      # same as last week
        if t - 52 >= 0:
            P_snaive[:, i, h] = Y[:, t - 52]            # same week last year
        P_ma[:, i, h] = Y[:, o - 3:o + 1].mean(1)       # mean of last 4 weeks
results["Naive"] = (P_naive, None, None)
results["Seasonal naive (52w)"] = (P_snaive, None, None)
results["Moving average (4w)"] = (P_ma, None, None)
print(f"done in {time.time() - t0:.1f}s")

# ---------------- 2. ARIMA (fit on training weeks, then roll forward without refitting) ----------------
if RUN_ARIMA:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        t0 = timeit("ARIMA(1,1,1) per series")
        P_ar = new_cube()
        n_fail = 0
        for s in range(S):
            try:
                res = ARIMA(Y[s, :TRAIN_END + 1], order=(1, 1, 1)).fit()
                for i, o in enumerate(O):
                    P_ar[s, i, :] = res.forecast(HORIZON)
                    if i < NO - 1:
                        res = res.extend(Y[s, o + 1:o + 2])      # add the new week, parameters stay fixed
            except Exception:
                n_fail += 1
                P_ar[s, :, :] = P_naive[s]                       # fall back to naive for this series
        P_ar = np.clip(P_ar, 0, None)
        results["ARIMA(1,1,1)"] = (P_ar, None, None)
        print(f"done in {time.time() - t0:.1f}s | fallbacks to naive: {n_fail}")
    except ImportError:
        print("statsmodels not installed -> ARIMA skipped (pip install statsmodels)")

# ---------------- 3. LIGHTGBM QUANTILE REGRESSION (one global model, all series) ----------------
# Note: it predicts sales as a ratio of the series' recent level, then multiplies back.
def take(M, idx):
    """M[:, idx] with NaN where idx is outside the matrix"""
    out = np.full((M.shape[0], len(idx)), np.nan)
    ok = (idx >= 0) & (idx < M.shape[1])
    out[:, ok] = M[:, idx[ok]]
    return out

LAGS = [0, 1, 2, 3, 4, 7, 12, 25, 51]

def make_features(origins, h):
    """Features for forecasting week o+h from information known at week o. Rows are series-major.
    Sales are divided by the series' recent level (mean of the last 13 weeks) so the model also works
    when the overall demand level drifts down or up (a tree cannot extrapolate beyond what it saw)."""
    origins = np.asarray(origins)
    t = origins + h
    n = len(origins)
    level = np.nanmean(np.stack([take(Y, origins - j) for j in range(13)]), axis=0)      # (S, n)
    level = np.maximum(level, 0.1 * mu[:, None])
    f = [take(Y, origins - k) / level for k in LAGS]                                      # lags
    for win in (4, 26):
        f.append(np.nanmean(np.stack([take(Y, origins - j) for j in range(win)]), axis=0) / level)
    f.append(take(Y, t - 52) / level)                                                     # same week last year
    f.append(np.nanmean(np.stack([take(Y, t - 52 + j) for j in (-1, 0, 1)]), axis=0) / level)
    f.append(take(PR, t))                                                                 # known-in-advance covariates
    f.append(take(PM, t))
    f.append(take(SN, t))
    f.append(take(EV, t))
    f.append(take(PR, t) / take(PR, origins))                                             # planned price vs current
    f.append(np.broadcast_to(np.where((t >= 0) & (t < T), WOY[np.clip(t, 0, T - 1)], np.nan), (S, n)))
    f.append(np.broadcast_to(np.where((t >= 0) & (t < T), MONTH[np.clip(t, 0, T - 1)], np.nan), (S, n)))
    f.append(np.full((S, n), float(h)))
    f.append(np.broadcast_to(cat_code[:, None].astype(float), (S, n)))                    # static
    f.append(np.broadcast_to(np.log(mu)[:, None], (S, n)))
    X = np.stack([a.reshape(-1) for a in f], axis=1)                                      # (S*n, features)
    y = (take(Y, t) / level).reshape(-1)                                                  # target relative to recent level
    return X, y, level

if RUN_LGBM:
    try:
        import lightgbm as lgb
        t0 = timeit("LightGBM quantile regression")
        VAL_WEEKS = 20                              # last 20 training weeks = validation (early stopping)
        Xtr, ytr, Xva, yva = [], [], [], []
        for h in range(1, HORIZON + 1):
            o_all = np.arange(12, TRAIN_END - h + 1)        # target week stays inside the training period
            for part, X_, y_ in (("tr", Xtr, ytr), ("va", Xva, yva)):
                sel = o_all[(o_all + h <= TRAIN_END - VAL_WEEKS)] if part == "tr" else \
                      o_all[(o_all + h > TRAIN_END - VAL_WEEKS)]
                X, y, _ = make_features(sel, h)
                X_.append(X); y_.append(y)
        Xtr, ytr, Xva, yva = map(np.concatenate, (Xtr, ytr, Xva, yva))
        ok_tr, ok_va = ~np.isnan(ytr), ~np.isnan(yva)
        print(f"training rows: {ok_tr.sum()} | validation rows: {ok_va.sum()}")

        # features of the evaluation rows
        Xev = {}
        LEV = {}
        for h in range(1, HORIZON + 1):
            Xev[h], _, LEV[h] = make_features(O, h)

        qpred = {}
        for q in QUANTILES:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=60 if QUICK else 600,
                                      learning_rate=0.05, num_leaves=31, min_child_samples=40,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                      random_state=SEED, verbose=-1)
            model.fit(Xtr[ok_tr], ytr[ok_tr], eval_set=[(Xva[ok_va], yva[ok_va])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            cube = new_cube()
            for h in range(1, HORIZON + 1):
                cube[:, :, h - 1] = model.predict(Xev[h]).reshape(S, NO) * LEV[h]
            qpred[q] = np.clip(cube, 0, None)
            print(f"  quantile {q}: best iteration {model.best_iteration_}")
        lo = np.minimum(qpred[QUANTILES[0]], qpred[0.5])         # avoid crossing quantiles
        hi = np.maximum(qpred[QUANTILES[-1]], qpred[0.5])
        results["LightGBM quantile reg."] = (qpred[0.5], lo, hi)
        print(f"done in {time.time() - t0:.1f}s")
    except ImportError:
        print("lightgbm not installed -> skipped (pip install lightgbm)")

# ---------------- 4. LSTM and TCN, each used two ways: normal (point) and MC Dropout (intervals) ----------------
if RUN_NEURAL:
    try:
        import torch
        import torch.nn as nn
        t0 = timeit("Neural baselines: " + ", ".join(NEURAL_MODELS))
        L, H = LSTM_WINDOW, HORIZON

        def pad(M):                                   # extend covariates by H weeks (only used for masked targets)
            return np.concatenate([M, np.repeat(M[:, -1:], H, axis=1)], axis=1)
        woy_p = np.concatenate([WOY, np.repeat(WOY[-1:], H)])
        SIN = np.broadcast_to(np.sin(2 * np.pi * woy_p / 52)[None], (S, T + H))
        COS = np.broadcast_to(np.cos(2 * np.pi * woy_p / 52)[None], (S, T + H))
        cov_p = [pad(np.clip(PR, 0.5, 1.5)), pad(PM), pad(SN / 7.0), pad(EV), SIN, COS]
        Y_p = np.concatenate([Y, np.full((S, H), np.nan)], axis=1)
        onehot = np.eye(len(cats))[cat_code]

        def windows(origins):
            """Past 26 weeks + known future covariates for every (series, origin). Sales are divided by the
            mean of the window, so the network sees relative changes and copes with a drifting level."""
            origins = np.asarray(origins)
            n = len(origins)
            idx = origins[:, None] + np.arange(-L + 1, 1)[None, :]          # (n, L) past weeks
            fidx = origins[:, None] + np.arange(1, H + 1)[None, :]          # (n, H) future weeks
            scale = np.maximum(Y[:, idx].mean(axis=2), 0.1 * mu[:, None])   # (S, n)
            seq = [np.clip(Y_p[:, idx] / scale[:, :, None], 0, 8)] + [c[:, idx] for c in cov_p]
            Xs = np.stack(seq, axis=-1).reshape(S * n, L, -1)
            Xf = np.stack([c[:, fidx] for c in cov_p], axis=-1).reshape(S * n, -1)
            Xst = np.repeat(onehot, n, axis=0)
            Yf = (Y_p[:, fidx] / scale[:, :, None]).reshape(S * n, H)
            return Xs, Xf, Xst, Yf, scale.reshape(S * n)

        def to_t(arrs):
            return [torch.tensor(a, dtype=torch.float32) for a in arrs[:4]]

        class LSTMNet(nn.Module):
            def __init__(self, f_in, f_fut, f_st, hid=64, p=0.2):
                super().__init__()
                self.lstm = nn.LSTM(f_in, hid, batch_first=True)
                self.drop = nn.Dropout(p)
                self.head = nn.Sequential(nn.Linear(hid + f_fut + f_st, 64), nn.ReLU(),
                                          nn.Dropout(p), nn.Linear(64, H))
            def forward(self, xs, xf, xst):
                _, (hn, _) = self.lstm(xs)
                return self.head(torch.cat([self.drop(hn[-1]), xf, xst], dim=1))

        class CausalBlock(nn.Module):
            """Two dilated causal convolutions + residual connection (the building block of a TCN)."""
            def __init__(self, c_in, c_out, dilation, p, k=3):
                super().__init__()
                self.pad = (k - 1) * dilation                    # pad on the left only = no peeking at the future
                self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
                self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
                self.drop = nn.Dropout(p)
                self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
            def forward(self, x):
                y = self.drop(torch.relu(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
                y = self.drop(torch.relu(self.conv2(nn.functional.pad(y, (self.pad, 0)))))
                return torch.relu(y + self.skip(x))

        class TCNNet(nn.Module):
            """Dilations 1,2,4,8 with kernel 3 (two convolutions per block) -> receptive field of 61 weeks, more than the 26-week window."""
            def __init__(self, f_in, f_fut, f_st, ch=32, p=0.2):
                super().__init__()
                blocks, c = [], f_in
                for d in (1, 2, 4, 8):
                    blocks.append(CausalBlock(c, ch, d, p))
                    c = ch
                self.tcn = nn.Sequential(*blocks)
                self.drop = nn.Dropout(p)
                self.head = nn.Sequential(nn.Linear(ch + f_fut + f_st, 64), nn.ReLU(),
                                          nn.Dropout(p), nn.Linear(64, H))
            def forward(self, xs, xf, xst):
                z = self.tcn(xs.transpose(1, 2))[:, :, -1]       # output at the last week
                return self.head(torch.cat([self.drop(z), xf, xst], dim=1))

        ARCH = {"LSTM": LSTMNet, "TCN": TCNNet}

        VAL_WEEKS = 20
        o_tr = np.arange(L - 1, TRAIN_END - H - VAL_WEEKS + 1)          # all 4 targets inside train (minus validation)
        o_va = np.arange(TRAIN_END - H - VAL_WEEKS + 1, TRAIN_END - H + 1)
        tr, va = to_t(windows(o_tr)), to_t(windows(o_va))
        ev_np = windows(O)                                               # evaluation windows, series-major
        ev = to_t(ev_np)
        ev_scale = ev_np[4].reshape(S, NO, 1)
        n_tr = tr[0].shape[0]
        print(f"training windows: {n_tr} | validation windows: {va[0].shape[0]}")
        loss_fn = nn.SmoothL1Loss()
        max_epochs, patience, bs = (4 if QUICK else 40), 6, 256

        def run(net, mc):
            net.eval()
            if mc:                                                       # keep dropout ON at prediction time
                for m in net.modules():
                    if isinstance(m, nn.Dropout):
                        m.train()
            with torch.no_grad():
                return torch.cat([net(ev[0][b:b + 4096], ev[1][b:b + 4096], ev[2][b:b + 4096])
                                  for b in range(0, ev[0].shape[0], 4096)]).numpy()

        for arch in NEURAL_MODELS:
            t1 = time.time()
            torch.manual_seed(SEED)
            net = ARCH[arch](tr[0].shape[2], tr[1].shape[1], tr[2].shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            best, best_state, wait = np.inf, None, 0
            for ep in range(max_epochs):
                net.train()
                perm = torch.randperm(n_tr)
                for b in range(0, n_tr, bs):
                    ib = perm[b:b + bs]
                    opt.zero_grad()
                    loss = loss_fn(net(tr[0][ib], tr[1][ib], tr[2][ib]), tr[3][ib])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vloss = loss_fn(net(va[0], va[1], va[2]), va[3]).item()
                if vloss < best - 1e-5:
                    best, wait = vloss, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break
            net.load_state_dict(best_state)
            print(f"{arch}: stopped after epoch {ep + 1} | best validation loss {best:.4f} | "
                  f"parameters {sum(p.numel() for p in net.parameters())}")

            P_point = np.clip(run(net, False).reshape(S, NO, H) * ev_scale, 0, None)
            results[arch] = (P_point, None, None)
            samples = np.stack([run(net, True) for _ in range(MC_SAMPLES)]).reshape(MC_SAMPLES, S, NO, H)
            samples = np.clip(samples * ev_scale[None], 0, None)
            results[f"MC-Dropout {arch}"] = (samples.mean(0),
                                             np.quantile(samples, QUANTILES[0], axis=0),
                                             np.quantile(samples, QUANTILES[-1], axis=0))
            print(f"{arch} done in {time.time() - t1:.1f}s")
    except ImportError:
        print("torch not installed -> LSTM/TCN skipped (pip install torch)")

# ---------------- EVALUATION ----------------
def point_metrics(y, p, m):
    e = p - y
    denom = (np.abs(y) + np.abs(p)) / 2
    smape = np.where(denom == 0, 0.0, np.abs(e) / np.where(denom == 0, 1, denom)).mean() * 100
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                WAPE=np.abs(e).sum() / np.abs(y).sum() * 100, sMAPE=smape,
                rel_MAE=(np.abs(e) / m).mean(), n=len(y))

mu_cube = np.broadcast_to(mu[:, None, None], Yt.shape)
rows, rows_h, long = [], [], []
for name, (P, lo, hi) in results.items():
    for split in ("calib", "test"):
        sel = MASK & (np.broadcast_to(SPLIT[None], Yt.shape) == split) & ~np.isnan(P)
        r = dict(model=name, split=split, **point_metrics(Yt[sel], P[sel], mu_cube[sel]))
        if lo is not None:
            r["coverage_%"] = ((Yt[sel] >= lo[sel]) & (Yt[sel] <= hi[sel])).mean() * 100
            r["width"] = (hi[sel] - lo[sel]).mean()
            r["rel_width"] = ((hi[sel] - lo[sel]) / mu_cube[sel]).mean()
        rows.append(r)
    for h in range(HORIZON):
        sel = MASK & (np.broadcast_to(SPLIT[None], Yt.shape) == "test") & ~np.isnan(P)
        sel_h = np.zeros_like(sel); sel_h[:, :, h] = sel[:, :, h]
        rows_h.append(dict(model=name, horizon=h + 1, MAE=np.abs(P[sel_h] - Yt[sel_h]).mean()))
    s_idx, i_idx, h_idx = np.where(MASK & ~np.isnan(P))
    long.append(pd.DataFrame(dict(
        model=name, series_id=series[s_idx], origin=O[i_idx], horizon=h_idx + 1,
        target_idx=TGT[i_idx, h_idx], split=SPLIT[i_idx, h_idx], y=Yt[s_idx, i_idx, h_idx],
        yhat=P[s_idx, i_idx, h_idx],
        lo=lo[s_idx, i_idx, h_idx] if lo is not None else np.nan,
        hi=hi[s_idx, i_idx, h_idx] if hi is not None else np.nan)))

metrics = pd.DataFrame(rows)
by_h = pd.DataFrame(rows_h)
pd.concat(long, ignore_index=True).to_parquet(RES / "baseline_forecasts.parquet", index=False)
metrics.round(3).to_csv(RES / "baseline_metrics.csv", index=False)
by_h.round(3).to_csv(RES / "baseline_metrics_by_horizon.csv", index=False)

pd.set_option("display.width", 200)
print("\n===== TEST SET (weeks after the calibration period) =====")
print(metrics[metrics.split == "test"].drop(columns="split").sort_values("MAE").round(2).to_string(index=False))
print("\n===== CALIBRATION SET =====")
print(metrics[metrics.split == "calib"].drop(columns="split").sort_values("MAE").round(2).to_string(index=False))
print("\n===== TEST MAE BY FORECAST HORIZON =====")
print(by_h.pivot(index="model", columns="horizon", values="MAE").round(2).to_string())

# ---------------- PLOTS ----------------
test = metrics[metrics.split == "test"].sort_values("WAPE")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].barh(test.model, test.WAPE, color="steelblue")
axes[0].invert_yaxis()
axes[0].set(title="Test error (WAPE %, lower is better)", xlabel="WAPE %")
for name, g in by_h.groupby("model"):
    axes[1].plot(g.horizon, g.MAE, marker="o", label=name)
axes[1].set(title="Test MAE by forecast horizon", xlabel="Weeks ahead", ylabel="MAE (units)")
axes[1].set_xticks(range(1, HORIZON + 1))
axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(RES / "baseline_comparison.png", dpi=150); plt.close(fig)

ints = metrics[(metrics.split == "test") & metrics["coverage_%"].notna()] if "coverage_%" in metrics else pd.DataFrame()
if len(ints):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(ints.model, ints["coverage_%"], color="darkorange")
    ax.axhline(95, color="grey", ls="--"); ax.text(0, 96, "95% target", fontsize=8)
    ax.set(title="Interval coverage on the test set", ylabel="Coverage %", ylim=(0, 100))
    fig.tight_layout(); fig.savefig(RES / "baseline_coverage.png", dpi=150); plt.close(fig)

print("\nSaved: results/baseline_metrics.csv, baseline_metrics_by_horizon.csv, baseline_forecasts.parquet, baseline_comparison.png")