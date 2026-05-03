import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

from crashrisk.config import Outputs


def poisson_nll(y_true, mu_pred, eps=1e-9):
    mu = np.clip(mu_pred, eps, None)
    return float(np.mean(mu - y_true * np.log(mu)))


def top_k_capture(df, pred_col, k_frac):
    k = max(1, int(len(df) * k_frac))
    top = df.sort_values(pred_col, ascending=False).head(k)
    total_crashes = df["y"].sum()

    if total_crashes == 0:
        return np.nan

    return top["y"].sum() / total_crashes


def compute_metrics(df, pred_col="mu_pred", rate_col="rate_pred"):
    y = df["y"].to_numpy(dtype=float)
    mu = df[pred_col].to_numpy(dtype=float)

    exposure = df["exposure"].to_numpy(dtype=float)
    true_rate = y / np.clip(exposure, 1e-9, None)
    pred_rate = df[rate_col].to_numpy(dtype=float)

    spearman_count = spearmanr(y, mu).correlation
    spearman_rate = spearmanr(true_rate, pred_rate).correlation

    return {
        "MAE Count": mean_absolute_error(y, mu),
        "RMSE Count": np.sqrt(mean_squared_error(y, mu)),
        "Poisson NLL": poisson_nll(y, mu),
        "MAE Rate": mean_absolute_error(true_rate, pred_rate),
        "Spearman Count": spearman_count,
        "Spearman Rate": spearman_rate,
        "Top 5% Crash Capture": top_k_capture(df, pred_col, 0.05),
        "Top 10% Crash Capture": top_k_capture(df, pred_col, 0.10),
        "Top 20% Crash Capture": top_k_capture(df, pred_col, 0.20),
    }


def add_baselines(df):
    df = df.copy()

    global_mean = df["y"].mean()
    df["mu_constant"] = global_mean
    df["rate_constant"] = df["mu_constant"] / np.clip(df["exposure"], 1e-9, None)

    global_rate = df["y"].sum() / np.clip(df["exposure"].sum(), 1e-9, None)
    df["mu_exposure_only"] = global_rate * df["exposure"]
    df["rate_exposure_only"] = global_rate

    return df


def plot_pred_vs_actual(df, out_dir):
    plt.figure(figsize=(7, 5))
    plt.scatter(df["y"], df["mu_pred"], alpha=0.4)
    plt.xlabel("Actual Crash Count")
    plt.ylabel("Predicted Crash Count")
    plt.title("Predicted vs Actual Crash Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_vs_actual_count.png"), dpi=200)
    plt.close()


def plot_prediction_distribution(df, out_dir):
    plt.figure(figsize=(7, 5))
    plt.hist(df["mu_pred"], bins=50)
    plt.xlabel("Predicted Crash Count")
    plt.ylabel("Number of Edges")
    plt.title("Distribution of Predicted Crash Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mu_pred_distribution.png"), dpi=200)
    plt.close()


def plot_topk_capture(df, out_dir):
    fracs = np.array([0.01, 0.05, 0.10, 0.20, 0.30])
    captures = [top_k_capture(df, "mu_pred", f) for f in fracs]

    plt.figure(figsize=(7, 5))
    plt.plot(fracs * 100, np.array(captures) * 100, marker="o")
    plt.xlabel("Top Predicted Edges Included (%)")
    plt.ylabel("Observed Crashes Captured (%)")
    plt.title("Crash Capture by Top Predicted Risk Segments")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "topk_crash_capture.png"), dpi=200)
    plt.close()


def plot_risk_bin_calibration(df, out_dir, n_bins=5):
    temp = df.copy()
    temp["risk_bin"] = pd.qcut(
        temp["mu_pred"].rank(method="first"),
        q=n_bins,
        labels=[f"Bin {i+1}" for i in range(n_bins)]
    )

    calib = temp.groupby("risk_bin").agg(
        actual_crashes=("y", "mean"),
        predicted_crashes=("mu_pred", "mean"),
        edge_count=("edge_id", "count"),
    ).reset_index()

    x = np.arange(len(calib))

    plt.figure(figsize=(8, 5))
    plt.plot(x, calib["actual_crashes"], marker="o", label="Actual Mean Crashes")
    plt.plot(x, calib["predicted_crashes"], marker="o", label="Predicted Mean Crashes")
    plt.xticks(x, calib["risk_bin"])
    plt.xlabel("Predicted Risk Bin")
    plt.ylabel("Mean Crash Count per Edge")
    plt.title("Risk Bin Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "risk_bin_calibration.png"), dpi=200)
    plt.close()

    return calib


def main():
    out = Outputs()
    out_dir = out.evaluation_dir
    os.makedirs(out_dir, exist_ok=True)

    pred_path = out.gnn_edge_predictions_file
    df = pd.read_parquet(pred_path)

    required = {"edge_id", "y", "mu_pred", "exposure", "rate_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")

    df = add_baselines(df)

    rows = []

    rows.append({
        "Model": "GraphSAGE",
        **compute_metrics(df, pred_col="mu_pred", rate_col="rate_pred")
    })

    rows.append({
        "Model": "Constant Mean",
        **compute_metrics(df, pred_col="mu_constant", rate_col="rate_constant")
    })

    rows.append({
        "Model": "Exposure Only",
        **compute_metrics(df, pred_col="mu_exposure_only", rate_col="rate_exposure_only")
    })

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(out_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    calib = plot_risk_bin_calibration(df, out_dir)
    calib.to_csv(os.path.join(out_dir, "risk_bin_calibration.csv"), index=False)

    plot_pred_vs_actual(df, out_dir)
    plot_prediction_distribution(df, out_dir)
    plot_topk_capture(df, out_dir)

    print("\nEvaluation complete.")
    print(metrics_df)
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()