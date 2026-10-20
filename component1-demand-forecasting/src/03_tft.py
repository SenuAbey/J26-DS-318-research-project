"""
01_tft.py - Temporal Fusion Transformer (Component 1)
Trains a TFT on the weekly M5 data and forecasts 1-4 weeks ahead for EXACTLY the same rolling-origin
rows as 01_baselines.py, so 04_compare_models.py can compare TFT with the baselines fairly.
Run from the component1-demand-forecasting folder:  python src/01_tft.py
Needs:  pip install torch lightning pytorch-forecasting pyarrow matplotlib
Tip:    on a laptop CPU this can take 30-90 min. A free GPU (Colab / Kaggle) is much faster; the script
        is unchanged there (accelerator="auto").
"""
from pathlib import Path
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

try:                                    # newer installs
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:                     # older installs
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")

# ---------------- SETTINGS ----------------
HORIZON = 4                          # forecast 1..4 weeks ahead
ENCODER = 26                         # weeks of history (same as the LSTM / TCN baselines)
TRAIN_END, CAL_END = 193, 235        # same split as 01_preprocess.py
VAL_WEEKS = 20                       # last 20 training weeks = validation (early stopping)
QUICK = False                        # True = 20 series, 2 epochs (fast test of the code)
MAX_EPOCHS, PATIENCE, BATCH = 30, 5, 128
QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]       # 0.5 = point forecast, 0.025 / 0.975 = 95% interval
NORMALIZER = "encoder"               # "encoder" = scale each window with its own recent history (copes with level drift)
                                     # "group"   = scale each series with its training-period statistics
HP = dict(hidden_size=16, attention_head_size=2, dropout=0.1, hidden_continuous_size=8, learning_rate=0.03)
TUNE_TRIALS = 0                      # >0 = random hyperparameter search first (e.g. 6); each trial = TUNE_EPOCHS epochs
TUNE_EPOCHS = 8
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RES, MODELS = ROOT / "results", ROOT / "models"
RES.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
pl.seed_everything(SEED, workers=True)

# ---------------- DATA ----------------
df = pd.read_parquet(ROOT / "data" / "processed" / "m5_weekly_ca1.parquet")
if QUICK:
    keep = np.sort(df.series_id.unique())[:20]
    df = df[df.series_id.isin(keep)]
    MAX_EPOCHS = 2
df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)
T = int(df.time_idx.max()) + 1
df["woy_sin"] = np.sin(2 * np.pi * df.week_of_year / 52)
df["woy_cos"] = np.cos(2 * np.pi * df.week_of_year / 52)
df["price_ratio"] = df.price_ratio.clip(0.5, 1.5)
df["month"] = df.month.astype(str)
for c in ("series_id", "item_id", "cat_id", "dept_id", "event_type", "month"):
    df[c] = df[c].astype(str)
for c in ("promo_flag", "has_event", "snap_days", "sales"):
    df[c] = df[c].astype(float)
print(f"Series: {df.series_id.nunique()} | weeks: {T}")

if NORMALIZER == "encoder":
    normalizer = EncoderNormalizer(method="standard", transformation="softplus")
else:
    normalizer = GroupNormalizer(groups=["series_id"], transformation="softplus")

training = TimeSeriesDataSet(
    df[df.time_idx <= TRAIN_END - VAL_WEEKS],            # all training targets stay before the validation weeks
    time_idx="time_idx", target="sales", group_ids=["series_id"],
    min_encoder_length=ENCODER, max_encoder_length=ENCODER,
    min_prediction_length=HORIZON, max_prediction_length=HORIZON,
    static_categoricals=["cat_id", "dept_id", "item_id"],                      # product identifiers
    time_varying_known_categoricals=["event_type", "month"],                   # known in advance (calendar)
    time_varying_known_reals=["price_ratio", "promo_flag", "snap_days", "has_event", "woy_sin", "woy_cos"],
    time_varying_unknown_reals=["sales"],                                      # only known up to the forecast origin
    target_normalizer=normalizer,
    add_relative_time_idx=True, add_target_scales=True, add_encoder_length=False,
)
validation = TimeSeriesDataSet.from_dataset(
    training, df[df.time_idx <= TRAIN_END], predict=False, stop_randomization=True,
    min_prediction_idx=TRAIN_END - HORIZON - VAL_WEEKS + 2)                    # same validation origins as the LSTM
evalset = TimeSeriesDataSet.from_dataset(
    training, df, predict=False, stop_randomization=True,
    min_prediction_idx=TRAIN_END + 1)                                          # decoder starts at week 194 or later
train_dl = training.to_dataloader(train=True, batch_size=BATCH, num_workers=0)
val_dl = validation.to_dataloader(train=False, batch_size=BATCH * 4, num_workers=0)
eval_dl = evalset.to_dataloader(train=False, batch_size=BATCH * 4, num_workers=0)
print(f"training windows: {len(training)} | validation windows: {len(validation)} | evaluation windows: {len(evalset)}")

# ---------------- TRAINING ----------------
def fit(hp, max_epochs, tag):
    """Train one TFT; return (path of best checkpoint, best validation loss, epochs used)."""
    pl.seed_everything(SEED, workers=True)
    model = TemporalFusionTransformer.from_dataset(
        training, **hp, loss=QuantileLoss(QUANTILES), log_interval=-1,
        optimizer="adam", reduce_on_plateau_patience=3)
    ckpt = ModelCheckpoint(dirpath=MODELS / tag, filename="tft", monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices=1, gradient_clip_val=0.1,
                         callbacks=[EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=PATIENCE, mode="min"), ckpt],
                         logger=False, enable_model_summary=False, enable_progress_bar=True)
    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] best val_loss {float(ckpt.best_model_score):.4f} | epochs {trainer.current_epoch} | "
          f"parameters {n_params} | {time.time() - t0:.0f}s")
    return ckpt.best_model_path, float(ckpt.best_model_score), n_params

tuning_log = []
best_hp = dict(HP)
if TUNE_TRIALS > 0:
    rng = np.random.default_rng(SEED)
    space = dict(hidden_size=[16, 32, 64], attention_head_size=[1, 2, 4], dropout=[0.1, 0.2, 0.3],
                 hidden_continuous_size=[8, 16], learning_rate=[0.003, 0.01, 0.03])
    trials = [dict(HP)] + [{k: v[rng.integers(len(v))].item() if hasattr(v[0], "item") else v[rng.integers(len(v))]
                            for k, v in space.items()} for _ in range(TUNE_TRIALS - 1)]
    best_loss = np.inf
    for i, hp in enumerate(trials):
        _, loss, _ = fit(hp, TUNE_EPOCHS, f"tune_{i}")
        tuning_log.append(dict(trial=i, val_loss=loss, **hp))
        if loss < best_loss:
            best_loss, best_hp = loss, hp
    pd.DataFrame(tuning_log).to_csv(RES / "tft_tuning.csv", index=False)
    print("best hyperparameters:", best_hp)

ckpt_path, val_loss, n_params = fit(best_hp, MAX_EPOCHS, "tft_final")
best = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
json.dump(dict(hyperparameters=best_hp, val_loss=val_loss, parameters=n_params, normalizer=NORMALIZER,
               quantiles=QUANTILES), open(RES / "tft_config.json", "w"), indent=2)

# ---------------- FORECASTS ON THE EVALUATION WEEKS ----------------
out = best.predict(eval_dl, mode="quantiles", return_index=True,
                   trainer_kwargs=dict(accelerator="auto", logger=False, enable_progress_bar=False))
pred = out.output if hasattr(out, "output") else out[0]
index = out.index if hasattr(out, "index") else out[1]
pred = pred.cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)           # (windows, 4, 5)
q = {v: pred[:, :, QUANTILES.index(v)] for v in (0.025, 0.5, 0.975)}
Ymat = df.pivot(index="series_id", columns="time_idx", values="sales")

rows = []
for h in range(HORIZON):
    target = index.time_idx.to_numpy() + h                                        # decoder starts at index.time_idx
    sid = index.series_id.to_numpy()
    rows.append(pd.DataFrame(dict(
        model="TFT", series_id=sid, origin=index.time_idx.to_numpy() - 1, horizon=h + 1, target_idx=target,
        split=np.where(target <= CAL_END, "calib", "test"),
        y=Ymat.to_numpy()[Ymat.index.get_indexer(sid), target],
        yhat=np.clip(q[0.5][:, h], 0, None),
        lo=np.clip(np.minimum(q[0.025][:, h], q[0.5][:, h]), 0, None),
        hi=np.clip(np.maximum(q[0.975][:, h], q[0.5][:, h]), 0, None))))
fc = pd.concat(rows, ignore_index=True)
fc.to_parquet(RES / "tft_forecasts.parquet", index=False)

# quick look (the full comparison with the baselines is in 04_compare_models.py)
for split, g in fc.groupby("split"):
    e = g.yhat - g.y
    print(f"TFT {split:5s} | MAE {np.abs(e).mean():.2f} | RMSE {np.sqrt((e ** 2).mean()):.2f} | "
          f"WAPE {np.abs(e).sum() / g.y.abs().sum() * 100:.1f}% | 95% interval coverage "
          f"{((g.y >= g.lo) & (g.y <= g.hi)).mean() * 100:.1f}% | rows {len(g)}")

# ---------------- INTERPRETATION (attention + variable importance) ----------------
try:
    raw = best.predict(eval_dl, mode="raw", return_x=True,
                       trainer_kwargs=dict(accelerator="auto", logger=False, enable_progress_bar=False))
    raw_out = raw.output if hasattr(raw, "output") else raw[0]
    interp = best.interpret_output(raw_out, reduction="sum")
    figs = best.plot_interpretation(interp)
    for name, fig in figs.items():
        fig.savefig(RES / f"tft_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    imp = {}
    for key, names in (("static_variables", best.static_variables), ("encoder_variables", best.encoder_variables),
                       ("decoder_variables", best.decoder_variables)):
        w = (interp[key] / interp[key].sum()).cpu().numpy()
        imp.update({f"{key.split('_')[0]}: {n}": float(v) for n, v in zip(names, w)})
    pd.Series(imp, name="importance").sort_values(ascending=False).round(4).to_csv(RES / "tft_variable_importance.csv")
    print("saved attention and variable importance plots (results/tft_*.png, tft_variable_importance.csv)")
except Exception as e:                              # interpretation must never break the forecasts
    print("interpretation skipped:", repr(e)[:200])

print("\nSaved: results/tft_forecasts.parquet, tft_config.json, models/tft_final/tft.ckpt")
print("Next: python src/04_compare_models.py")