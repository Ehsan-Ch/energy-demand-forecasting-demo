"""One-hour-ahead forecasting on explicitly synthetic electricity demand.

Run: python forecast.py
All features for target time t use calendar information or observations before t.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def generate_data(days: int = 365, seed: int = SEED) -> pd.DataFrame:
    """Invent a load series for a fictional small grid; never real market data."""
    if days < 60:
        raise ValueError("Use at least 60 days so chronological splits are meaningful.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    hour = dates.hour.to_numpy()
    day = np.arange(len(dates)) / 24
    weather_noise = np.zeros(len(dates))
    demand_noise = np.zeros(len(dates))
    for t in range(1, len(dates)):
        weather_noise[t] = 0.92 * weather_noise[t - 1] + rng.normal(0, 0.7)
        demand_noise[t] = 0.6 * demand_noise[t - 1] + rng.normal(0, 2.5)
    temperature = (10 + 9 * np.sin(2 * np.pi * (day - 90) / 365)
                   + 2 * np.sin(2 * np.pi * (hour - 8) / 24) + weather_noise)
    evening = 18 * np.exp(-0.5 * ((hour - 18) / 2.5) ** 2)
    morning = 9 * np.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
    weekday = (dates.dayofweek.to_numpy() < 5).astype(float)
    demand = (65 + evening + morning + 10 * weekday
              + 2.2 * np.maximum(15 - temperature, 0)
              + 0.012 * day + demand_noise)
    return pd.DataFrame({"demand_mw": demand, "temperature_c": temperature}, index=dates)


def make_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """At t, assume demand/temperature through t-1 have arrived without delay."""
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError("Timestamps must be sorted and unique.")
    if not (data.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Expected regular hourly timestamps without gaps.")
    if data[["demand_mw", "temperature_c"]].isna().any().any():
        raise ValueError("Missing observations require an explicit data-quality policy.")
    x = pd.DataFrame(index=data.index)
    for lag in (1, 2, 24, 48, 168):
        x[f"load_lag_{lag}"] = data.demand_mw.shift(lag)
    x["load_mean_24"] = data.demand_mw.shift(1).rolling(24).mean()
    x["temperature_lag_1"] = data.temperature_c.shift(1)
    x["heating_lag_1"] = (15 - x.temperature_lag_1).clip(lower=0)
    for k in (1, 2, 3):
        x[f"hour_sin_{k}"] = np.sin(k * 2 * np.pi * data.index.hour / 24)
        x[f"hour_cos_{k}"] = np.cos(k * 2 * np.pi * data.index.hour / 24)
    x["weekend"] = (data.index.dayofweek >= 5).astype(int)
    x["trend_days"] = (data.index - data.index[0]).total_seconds() / 86400
    x = x.dropna()
    return x, data.demand_mw.loc[x.index].copy()


def split_bounds(n: int) -> tuple[int, int]:
    return int(n * 0.6), int(n * 0.8)


def errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {"mae_mw": float(mean_absolute_error(actual, predicted)),
            "rmse_mw": float(np.sqrt(mean_squared_error(actual, predicted)))}


def build_model(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def run(output: Path, days: int = 365, seed: int = SEED) -> dict:
    data = generate_data(days, seed)
    x, y = make_features(data)
    train_end, val_end = split_bounds(len(x))
    train_x, val_x, test_x = x.iloc[:train_end], x.iloc[train_end:val_end], x.iloc[val_end:]
    train_y, val_y, test_y = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
    tuning = []
    for alpha in ALPHAS:
        model = build_model(alpha)
        model.fit(train_x, train_y)
        tuning.append({"alpha": alpha, **errors(val_y, model.predict(val_x))})
    selected_alpha = min(tuning, key=lambda row: row["mae_mw"])["alpha"]
    final_model = build_model(selected_alpha)
    final_model.fit(x.iloc[:val_end], y.iloc[:val_end])
    predictions = pd.DataFrame({"actual_mw": test_y,
                                "ridge_mw": final_model.predict(test_x),
                                "last_hour_mw": test_x.load_lag_1,
                                "daily_naive_mw": test_x.load_lag_24,
                                "weekly_naive_mw": test_x.load_lag_168})
    metrics = {name: errors(test_y, predictions[column]) for name, column in
               [("Ridge", "ridge_mw"), ("Last hour", "last_hour_mw"),
                ("Daily naive", "daily_naive_mw"), ("Weekly naive", "weekly_naive_mw")]}
    def period(frame):
        return {"start": frame.index[0].isoformat(), "end": frame.index[-1].isoformat(),
                "rows": len(frame)}
    result = {"data_kind": "synthetic; fictional grid; not Danish market observations",
              "seed": seed, "days": days, "horizon_hours": 1,
              "evaluation": "fixed model, sequential one-step predictions with observed history",
              "split": {"train": period(train_x), "validation": period(val_x),
                        "test": period(test_x)},
              "selected_alpha": selected_alpha, "validation_candidates": tuning,
              "test_metrics": metrics, "features": list(x.columns),
              "data_sha256": hashlib.sha256(data.to_csv(float_format="%.10f").encode()).hexdigest(),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                              "matplotlib": matplotlib.__version__}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    predictions.to_csv(output / "test_predictions.csv", index_label="timestamp", float_format="%.6f")
    data.head(48).to_csv(output / "synthetic_data_sample.csv", index_label="timestamp", float_format="%.6f")
    plot_results(predictions, metrics, output / "figures")
    write_report(result, output)
    return result


def plot_results(predictions: pd.DataFrame, metrics: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "energy-demand-demo"})
    colors = {"actual_mw": "#172b4d", "ridge_mw": "#168a81", "daily_naive_mw": "#b47525"}
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    tail = predictions.tail(168)
    for col, label in [("actual_mw", "Synthetic demand"), ("ridge_mw", "Ridge"),
                       ("daily_naive_mw", "Daily naive")]:
        ax.plot(tail.index, tail[col], label=label, color=colors[col], linewidth=1.4)
    ax.set(title="Synthetic electricity demand | final test week", ylabel="Demand (MW)", xlabel="Target timestamp (UTC)")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.15)
    fig.savefig(output / "forecast.svg", metadata={"Date": None})
    fig.savefig(output / "forecast.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    names = list(metrics)
    for ax, metric, label in zip(axes, ("mae_mw", "rmse_mw"), ("MAE", "RMSE")):
        vals = [metrics[name][metric] for name in names]
        bars = ax.barh(names, vals, color=["#168a81", "#91a3b0", "#b47525", "#717da7"])
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set(xlabel=f"{label} (MW) — lower is better", xlim=(0, max(vals) * 1.18))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Measured errors on the synthetic holdout set")
    fig.savefig(output / "errors.svg", metadata={"Date": None})
    fig.savefig(output / "errors.png", dpi=160)
    plt.close(fig)


def write_report(result: dict, output: Path) -> None:
    rows = "\n".join(f"| {name} | {m['mae_mw']:.3f} | {m['rmse_mw']:.3f} |"
                     for name, m in result["test_metrics"].items())
    splits = "\n".join(f"| {name.title()} | {p['start']} | {p['end']} | {p['rows']} |"
                       for name, p in result["split"].items())
    report = f"""# Model evaluation — synthetic demonstration

These are computed regression errors from the included experiment, not mock accuracy
percentages or evidence of performance on real electricity markets. Seed: {result['seed']}.
The data cover {result['days']} fictional days. All timestamps are UTC.

## Evaluation design

Predict demand one hour ahead. At the start of hour t, observations through t-1 are
assumed available. Earlier test observations are therefore used as lags for later
test predictions. This is sequential one-step evaluation, not a day-ahead or a
multi-step forecast. The fitted model stays fixed throughout the test period.

| Split | First target | Last target | Rows |
| --- | --- | --- | ---: |
{splits}

The first 168 observations supply lag history and are excluded from target rows.
Ridge alpha is chosen using validation MAE from {ALPHAS}. Selected alpha:
**{result['selected_alpha']}**. Each candidate scaler is fitted only to the training
split. The final scaler and model are refitted on train plus validation, then
evaluated on the final test period. No random split, full-series scaling or
backfilling is used. The test period is not used to choose alpha.

## Measured test results

| Model | MAE (MW) | RMSE (MW) |
| --- | ---: | ---: |
{rows}

MAE is the mean absolute error. RMSE is the square root of mean squared error
and gives larger errors more weight. Both use the same target rows. A regression
forecast has no universal accuracy percentage, so none is reported.

![Measured errors](figures/errors.svg)

![Final test week](figures/forecast.svg)

## Limits

- The generator is intentionally simple and known to the author. Results measure
  performance on that generator, not generalisation to an independent real system.
- One seed and one chronological holdout are insufficient for a production claim.
- Demand and temperature are assumed to arrive without operational latency.
- No price forecast, trading return, prediction interval or deployment is claimed.
- Calendar effects omit local holidays and daylight-saving transitions.
- Real data would require availability-aware weather inputs, missing-data policies,
  multiple rolling-origin evaluations, drift checks and appropriate data licensing.

## Reproduce and inspect

Run `python forecast.py`, then `python -m unittest discover -s tests -v`.
`metrics.json` stores exact scores, split boundaries, environment versions and a
synthetic-data hash. `test_predictions.csv` contains every holdout prediction.
`synthetic_data_sample.csv` contains the first 48 invented observations.
"""
    (output / "evaluation.md").write_text(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run(args.output, args.days, args.seed)
    print(json.dumps(result["test_metrics"], indent=2))
