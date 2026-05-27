# data_analysis.py
# Usage:
#   python data_analysis.py
#
# Purpose:
#   F1 PitNextLap prediction-focused EDA.
#   This is NOT just random plotting.
#   It focuses on finding what makes PitNextLap=1 more likely.
#
# Required:
#   train.csv
#   test.csv
#
# Output:
#   analysis_output/
#
# Main outputs:
#   1_target_overview.png
#   2_pit_rate_by_tyre_life.png
#   3_pit_rate_by_compound.png
#   4_pit_rate_by_race_progress.png
#   5_pit_rate_by_pitstop.png
#   6_pit_rate_by_position.png
#   7_pit_rate_by_laptime_delta.png
#   8_pit_rate_by_degradation.png
#   9_pit_rate_by_race_top20.png
#   10_train_test_shift_key_features.png
#   summary_tables.xlsx
#
# Notes:
#   Driver is intentionally ignored.

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

TARGET = "PitNextLap"
ID_COL = "id"
OUT_DIR = Path("analysis_output")


KEY_NUMERIC_FEATURES = [
    "TyreLife",
    "LapNumber",
    "RaceProgress",
    "PitStop",
    "Stint",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "Position_Change",
]

COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def ensure_output_dir():
    OUT_DIR.mkdir(exist_ok=True)


def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=160)
    plt.close()


def bar_with_values(series, title, xlabel, ylabel, filename, rotation=0):
    plt.figure(figsize=(9, 5))
    ax = series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")

    for p in ax.patches:
        h = p.get_height()
        if pd.notna(h):
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    save_plot(filename)


def target_overview(train):
    counts = train[TARGET].value_counts().sort_index()
    rate = train[TARGET].mean()

    plt.figure(figsize=(7, 5))
    ax = counts.plot(kind="bar")
    plt.title(f"Target Distribution: PitNextLap, Positive Rate = {rate:.4f}")
    plt.xlabel("PitNextLap")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{int(h)}",
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_plot("1_target_overview.png")

    return pd.DataFrame({
        "PitNextLap": counts.index,
        "count": counts.values,
        "ratio": counts.values / counts.sum(),
    })


def pit_rate_by_category(train, col, min_count=100):
    tmp = (
        train.groupby(col)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"mean": "pit_rate"})
        .reset_index()
    )
    tmp = tmp[tmp["count"] >= min_count].copy()
    return tmp.sort_values("pit_rate", ascending=False)


def compound_analysis(train):
    if "Compound" not in train.columns:
        return pd.DataFrame()

    tmp = train.copy()
    tmp["Compound"] = tmp["Compound"].astype(str).str.upper()

    stat = (
        tmp.groupby("Compound")[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"mean": "pit_rate"})
    )

    ordered = [c for c in COMPOUND_ORDER if c in stat.index]
    stat = stat.loc[ordered]

    bar_with_values(
        stat["pit_rate"],
        "PitNextLap Rate by Compound",
        "Compound",
        "PitNextLap Rate",
        "3_pit_rate_by_compound.png",
        rotation=0,
    )

    # Also show compound usage count
    plt.figure(figsize=(8, 5))
    ax = stat["count"].plot(kind="bar")
    plt.title("Compound Count")
    plt.xlabel("Compound")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=8)
    save_plot("3b_compound_count.png")

    return stat.reset_index()


def bin_numeric_by_quantile(train, col, bins=12):
    data = train[[col, TARGET]].dropna().copy()

    if data[col].nunique() <= bins:
        data["bin"] = data[col].astype(str)
    else:
        data["bin"] = pd.qcut(data[col], q=bins, duplicates="drop")

    stat = (
        data.groupby("bin", observed=True)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"mean": "pit_rate"})
        .reset_index()
    )

    stat["bin"] = stat["bin"].astype(str)
    return stat


def plot_pit_rate_by_numeric_bin(train, col, filename, title=None, bins=12):
    if col not in train.columns:
        return pd.DataFrame()

    stat = bin_numeric_by_quantile(train, col, bins=bins)

    plt.figure(figsize=(11, 5))
    plt.plot(range(len(stat)), stat["pit_rate"], marker="o")
    plt.title(title or f"PitNextLap Rate by {col}")
    plt.xlabel(col + " bins")
    plt.ylabel("PitNextLap Rate")
    plt.xticks(range(len(stat)), stat["bin"], rotation=60, ha="right")
    plt.grid(True, alpha=0.3)
    save_plot(filename)

    return stat


def tyre_life_analysis(train):
    if "TyreLife" not in train.columns:
        return pd.DataFrame()

    # Exact tyre life may be too noisy, so use manual bins.
    bins = [-1, 0, 3, 6, 9, 12, 15, 18, 22, 26, 30, 35, 40, 50, 1000]
    labels = [
        "0", "1-3", "4-6", "7-9", "10-12", "13-15", "16-18",
        "19-22", "23-26", "27-30", "31-35", "36-40", "41-50", "50+"
    ]

    data = train[["TyreLife", TARGET]].dropna().copy()
    data["TyreLife_bin"] = pd.cut(data["TyreLife"], bins=bins, labels=labels)

    stat = (
        data.groupby("TyreLife_bin", observed=True)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"mean": "pit_rate"})
        .reset_index()
    )

    plt.figure(figsize=(11, 5))
    plt.plot(range(len(stat)), stat["pit_rate"], marker="o")
    plt.title("PitNextLap Rate by TyreLife")
    plt.xlabel("TyreLife bin")
    plt.ylabel("PitNextLap Rate")
    plt.xticks(range(len(stat)), stat["TyreLife_bin"].astype(str), rotation=45)
    plt.grid(True, alpha=0.3)
    save_plot("2_pit_rate_by_tyre_life.png")

    return stat


def race_progress_analysis(train):
    if "RaceProgress" not in train.columns:
        return pd.DataFrame()

    bins = np.linspace(0, 1, 11)
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]

    data = train[["RaceProgress", TARGET]].dropna().copy()
    data["RaceProgress_bin"] = pd.cut(data["RaceProgress"], bins=bins, labels=labels, include_lowest=True)

    stat = (
        data.groupby("RaceProgress_bin", observed=True)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"mean": "pit_rate"})
        .reset_index()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(stat)), stat["pit_rate"], marker="o")
    plt.title("PitNextLap Rate by Race Progress")
    plt.xlabel("Race progress")
    plt.ylabel("PitNextLap Rate")
    plt.xticks(range(len(stat)), stat["RaceProgress_bin"].astype(str), rotation=45)
    plt.grid(True, alpha=0.3)
    save_plot("4_pit_rate_by_race_progress.png")

    return stat


def pitstop_analysis(train):
    if "PitStop" not in train.columns:
        return pd.DataFrame()

    stat = (
        train.groupby("PitStop")[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"mean": "pit_rate"})
    )

    bar_with_values(
        stat["pit_rate"],
        "PitNextLap Rate by Current PitStop Count",
        "PitStop",
        "PitNextLap Rate",
        "5_pit_rate_by_pitstop.png",
    )

    return stat.reset_index()


def position_analysis(train):
    if "Position" not in train.columns:
        return pd.DataFrame()

    bins = [0, 3, 6, 10, 15, 20, 30]
    labels = ["P1-P3", "P4-P6", "P7-P10", "P11-P15", "P16-P20", "P21+"]

    data = train[["Position", TARGET]].dropna().copy()
    data["Position_group"] = pd.cut(data["Position"], bins=bins, labels=labels, include_lowest=True)

    stat = (
        data.groupby("Position_group", observed=True)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={"mean": "pit_rate"})
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    ax = plt.bar(stat["Position_group"].astype(str), stat["pit_rate"])
    plt.title("PitNextLap Rate by Position Group")
    plt.xlabel("Position group")
    plt.ylabel("PitNextLap Rate")
    plt.xticks(rotation=0)

    for i, v in enumerate(stat["pit_rate"]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    save_plot("6_pit_rate_by_position.png")
    return stat


def race_analysis(train):
    if "Race" not in train.columns:
        return pd.DataFrame()

    stat = pit_rate_by_category(train, "Race", min_count=100)
    stat.to_csv(OUT_DIR / "pit_rate_by_race.csv", index=False)

    top = stat.head(20).copy()
    plt.figure(figsize=(12, 6))
    plt.bar(top["Race"].astype(str), top["pit_rate"])
    plt.title("Top 20 Races by PitNextLap Rate")
    plt.xlabel("Race")
    plt.ylabel("PitNextLap Rate")
    plt.xticks(rotation=75, ha="right")
    save_plot("9_pit_rate_by_race_top20.png")

    return stat


def interaction_compound_tyre_life(train):
    if "Compound" not in train.columns or "TyreLife" not in train.columns:
        return pd.DataFrame()

    bins = [-1, 5, 10, 15, 20, 25, 30, 40, 1000]
    labels = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-40", "40+"]

    data = train[["Compound", "TyreLife", TARGET]].dropna().copy()
    data["Compound"] = data["Compound"].astype(str).str.upper()
    data["TyreLife_bin"] = pd.cut(data["TyreLife"], bins=bins, labels=labels)

    pivot = data.pivot_table(
        index="Compound",
        columns="TyreLife_bin",
        values=TARGET,
        aggfunc="mean",
        observed=True,
    )

    ordered = [c for c in COMPOUND_ORDER if c in pivot.index]
    pivot = pivot.loc[ordered]

    plt.figure(figsize=(11, 5))
    mat = pivot.values
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="PitNextLap Rate")
    plt.title("Pit Rate Heatmap: Compound × TyreLife")
    plt.xlabel("TyreLife bin")
    plt.ylabel("Compound")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    save_plot("11_heatmap_compound_tyre_life.png")

    return pivot.reset_index()


def interaction_progress_tyre_life(train):
    if "RaceProgress" not in train.columns or "TyreLife" not in train.columns:
        return pd.DataFrame()

    data = train[["RaceProgress", "TyreLife", TARGET]].dropna().copy()

    data["RaceProgress_bin"] = pd.cut(
        data["RaceProgress"],
        bins=np.linspace(0, 1, 6),
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        include_lowest=True,
    )

    data["TyreLife_bin"] = pd.cut(
        data["TyreLife"],
        bins=[-1, 5, 10, 15, 20, 25, 30, 40, 1000],
        labels=["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-40", "40+"],
    )

    pivot = data.pivot_table(
        index="RaceProgress_bin",
        columns="TyreLife_bin",
        values=TARGET,
        aggfunc="mean",
        observed=True,
    )

    plt.figure(figsize=(11, 5))
    mat = pivot.values
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="PitNextLap Rate")
    plt.title("Pit Rate Heatmap: RaceProgress × TyreLife")
    plt.xlabel("TyreLife bin")
    plt.ylabel("RaceProgress")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    save_plot("12_heatmap_progress_tyre_life.png")

    return pivot.reset_index()


def train_test_shift(train, test):
    rows = []

    for col in KEY_NUMERIC_FEATURES:
        if col not in train.columns or col not in test.columns:
            continue

        tr = train[col].dropna()
        te = test[col].dropna()

        rows.append({
            "feature": col,
            "train_mean": tr.mean(),
            "test_mean": te.mean(),
            "mean_diff": te.mean() - tr.mean(),
            "train_std": tr.std(),
            "test_std": te.std(),
            "std_diff": te.std() - tr.std(),
        })

    stat = pd.DataFrame(rows)

    if stat.empty:
        return stat

    plot_df = stat.copy()
    plot_df["abs_mean_diff"] = plot_df["mean_diff"].abs()
    plot_df = plot_df.sort_values("abs_mean_diff", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["feature"], plot_df["mean_diff"])
    plt.title("Train/Test Shift: Mean Difference on Key Features")
    plt.xlabel("Feature")
    plt.ylabel("Test Mean - Train Mean")
    plt.xticks(rotation=60, ha="right")
    save_plot("10_train_test_shift_key_features.png")

    return stat


def target_mean_difference(train):
    rows = []

    for col in KEY_NUMERIC_FEATURES:
        if col not in train.columns:
            continue

        g = train.groupby(TARGET)[col].mean()

        rows.append({
            "feature": col,
            "mean_when_no_pit": g.get(0, np.nan),
            "mean_when_pit_next_lap": g.get(1, np.nan),
            "pit_minus_no_pit": g.get(1, np.nan) - g.get(0, np.nan),
        })

    stat = pd.DataFrame(rows)
    stat = stat.sort_values("pit_minus_no_pit", key=lambda x: x.abs(), ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(stat["feature"], stat["pit_minus_no_pit"])
    plt.title("Feature Mean Difference: PitNextLap=1 minus PitNextLap=0")
    plt.xlabel("Feature")
    plt.ylabel("Mean Difference")
    plt.xticks(rotation=60, ha="right")
    save_plot("13_feature_mean_difference_by_target.png")

    return stat


def laptime_delta_analysis(train):
    return plot_pit_rate_by_numeric_bin(
        train,
        "LapTime_Delta",
        "7_pit_rate_by_laptime_delta.png",
        title="PitNextLap Rate by LapTime_Delta",
        bins=12,
    )


def degradation_analysis(train):
    return plot_pit_rate_by_numeric_bin(
        train,
        "Cumulative_Degradation",
        "8_pit_rate_by_degradation.png",
        title="PitNextLap Rate by Cumulative Degradation",
        bins=12,
    )


def write_summary_excel(tables):
    out_path = OUT_DIR / "summary_tables.xlsx"

    with pd.ExcelWriter(out_path) as writer:
        for name, df in tables.items():
            if df is None or len(df) == 0:
                continue

            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

    print("Saved:", out_path)


def write_report_md(train, tables):
    positive_rate = train[TARGET].mean()

    lines = []
    lines.append("# F1 PitNextLap Data Analysis Summary")
    lines.append("")
    lines.append(f"- Train rows: {len(train)}")
    lines.append(f"- Positive rate: {positive_rate:.4f}")
    lines.append("- Driver is ignored intentionally.")
    lines.append("")
    lines.append("## Main things to check")
    lines.append("")
    lines.append("1. Is `PitNextLap=1` strongly related to `TyreLife`?")
    lines.append("2. Does compound type change pit probability?")
    lines.append("3. Are pit stops concentrated in certain race progress ranges?")
    lines.append("4. Does lap time degradation increase pit probability?")
    lines.append("5. Is test distribution close to train distribution?")
    lines.append("")

    if "target_mean_diff" in tables and not tables["target_mean_diff"].empty:
        lines.append("## Top numeric mean differences")
        top = tables["target_mean_diff"].head(8)
        for _, r in top.iterrows():
            lines.append(
                f"- {r['feature']}: pit mean - no-pit mean = {r['pit_minus_no_pit']:.4f}"
            )
        lines.append("")

    if "compound" in tables and not tables["compound"].empty:
        lines.append("## Compound pit rate")
        for _, r in tables["compound"].iterrows():
            lines.append(f"- {r['Compound']}: pit_rate={r['pit_rate']:.4f}, count={int(r['count'])}")
        lines.append("")

    (OUT_DIR / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", OUT_DIR / "analysis_report.md")


def main():
    ensure_output_dir()

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Do not use Driver in analysis
    if "Driver" in train.columns:
        train = train.drop(columns=["Driver"])
    if "Driver" in test.columns:
        test = test.drop(columns=["Driver"])

    print("Train shape:", train.shape)
    print("Test shape :", test.shape)
    print("Positive rate:", train[TARGET].mean())

    tables = {}

    tables["target_overview"] = target_overview(train)
    tables["compound"] = compound_analysis(train)
    tables["tyre_life"] = tyre_life_analysis(train)
    tables["race_progress"] = race_progress_analysis(train)
    tables["pitstop"] = pitstop_analysis(train)
    tables["position"] = position_analysis(train)
    tables["laptime_delta"] = laptime_delta_analysis(train)
    tables["degradation"] = degradation_analysis(train)
    tables["race"] = race_analysis(train)
    tables["compound_x_tyre"] = interaction_compound_tyre_life(train)
    tables["progress_x_tyre"] = interaction_progress_tyre_life(train)
    tables["train_test_shift"] = train_test_shift(train, test)
    tables["target_mean_diff"] = target_mean_difference(train)

    write_summary_excel(tables)
    write_report_md(train, tables)

    print()
    print("Done.")
    print("Open this folder:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
