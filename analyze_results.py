"""Explain the lane detector's results: train the random forest and draw the charts.

This needs only lane_results.csv (one row per image, written by the notebook).
It does not run the lane detector and does not need the CULane images.

    python analyze_results.py                      # reads lane_results.csv, writes to images/
    python analyze_results.py my_results.csv out/  # other file, other folder
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else "lane_results.csv"
CHART_FOLDER = Path(sys.argv[2] if len(sys.argv) > 2 else "images")

SEED = 0
PASS_PX = 25           # an image passed if its worse line was within this many pixels
ERROR_CAP = 100        # errors in the table are capped at this
VISIBLE_CONTRAST = 15  # a true line this much brighter than the road counts as visible

# Scene features only. Nothing from the detector itself (such as "missed") goes in,
# or the model would just learn "it failed when it found nothing".
FEATURES = ["brightness", "contrast", "edge_density", "curve_px", "line_contrast"]

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})


def save(name):
    plt.tight_layout()
    plt.savefig(CHART_FOLDER / name, dpi=100)
    plt.close()
    print("  saved", CHART_FOLDER / name)


def summarise(table):
    summary = table.groupby("category").agg(
        images=("passed", "size"),
        pass_rate=("passed", "mean"),
        median_error=("error", "median"),
        missed_a_line=("missed", "mean"),
    ).sort_values("pass_rate", ascending=False)
    print(f"{len(table)} images, overall pass rate {table['passed'].mean():.1%}\n")
    print(summary.round(2).to_string())

    if "line_contrast" in table:
        visible = table["line_contrast"] >= VISIBLE_CONTRAST
        print(f"\nBoth true lines clearly visible in {visible.mean():.0%} of images.")
        print(f"  pass rate when they are:     {table.loc[visible, 'passed'].mean():.1%}")
        print(f"  pass rate when they are not: {table.loc[~visible, 'passed'].mean():.1%}")
    return summary


def train_forest(table, features):
    """Predict pass/fail from the scene features; return each feature's importance."""
    X = table[features].fillna(table[features].median())
    y = table["passed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    forest = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    ).fit(X_train, y_train)

    predicted = forest.predict(X_test)
    always_common = max(y_test.mean(), 1 - y_test.mean())
    print(f"\nAccuracy:          {accuracy_score(y_test, predicted):.1%}"
          f"  (always guessing the commoner outcome: {always_common:.1%})")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predicted):.1%}"
          "  (chance: 50%)")

    # Importance on held-out images: how much the score drops when one
    # feature's values are shuffled.
    shuffled = permutation_importance(
        forest, X_test, y_test, scoring="balanced_accuracy",
        n_repeats=20, random_state=SEED,
    )
    importance = pd.DataFrame({
        "feature": features,
        "importance": shuffled.importances_mean,
        "spread": shuffled.importances_std,
    }).sort_values("importance")
    print("\n" + importance.sort_values("importance", ascending=False)
          .round(3).to_string(index=False))
    return importance


def pass_rate_chart(summary):
    rates = summary.sort_values("pass_rate")
    figure, axis = plt.subplots(figsize=(7, 0.45 * len(rates) + 1.2))
    bars = axis.barh(
        [f"{name}  (n={count})" for name, count in zip(rates.index, rates["images"])],
        rates["pass_rate"] * 100, color=BLUE, height=0.6,
    )
    axis.bar_label(bars, fmt="%.0f%%", padding=4, color=INK)
    axis.set_xlim(0, 108)
    axis.set_xlabel(f"Images passing (worse line within {PASS_PX} px), %")
    axis.set_title("Pass rate per category")
    axis.grid(axis="y", visible=False)
    save("pass_rate_per_category.png")


def error_chart(table, feature, label, file_name, x_max=None):
    """Top: pass rate in ten equal-sized groups. Bottom: every image's error."""
    name = label.split(" (")[0].lower()
    figure, (rate_axis, error_axis) = plt.subplots(
        2, 1, figsize=(7, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]}
    )
    groups = pd.qcut(table[feature], 10, duplicates="drop")
    grouped = table.groupby(groups, observed=True).agg(
        x=(feature, "median"), rate=("passed", "mean")
    )
    rate_axis.plot(grouped["x"], grouped["rate"] * 100, color=BLUE, linewidth=2,
                   marker="o", markersize=6)
    rate_axis.set_ylim(0, max(50, grouped["rate"].max() * 110))
    rate_axis.set_ylabel("Images passing, %")
    rate_axis.set_title(f"Pass rate and error vs {name}")

    error_axis.scatter(table[feature], table["error"], s=8, color=BLUE, alpha=0.25,
                       linewidths=0)
    error_axis.axhline(PASS_PX, color=MUTED, linewidth=1, linestyle="--")
    error_axis.annotate(f"pass limit, {PASS_PX} px", (1, PASS_PX),
                        xycoords=("axes fraction", "data"), xytext=(-4, 4),
                        textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    error_axis.set_ylim(0, ERROR_CAP * 1.05)
    error_axis.set_ylabel(f"Error, px (capped at {ERROR_CAP})")
    if x_max is not None:
        beyond = (table[feature] > x_max).mean()
        error_axis.set_xlim(left=min(0, table[feature].min()) - x_max * 0.02, right=x_max)
        label += f"; {beyond:.0%} of images are beyond {x_max} and not shown"
    error_axis.set_xlabel(label)
    save(file_name)


def importance_chart(importance):
    figure, axis = plt.subplots(figsize=(7, 0.5 * len(importance) + 1.2))
    axis.barh(importance["feature"], importance["importance"], xerr=importance["spread"],
              color=BLUE, height=0.6, ecolor=MUTED, capsize=3)
    axis.axvline(0, color=MUTED, linewidth=1)
    axis.set_xlabel("Drop in balanced accuracy when the feature is shuffled")
    axis.set_title("What the forest relies on to predict pass/fail")
    axis.grid(axis="y", visible=False)
    save("feature_importance.png")


def main():
    table = pd.read_csv(RESULTS_FILE)
    features = [feature for feature in FEATURES if feature in table.columns]
    CHART_FOLDER.mkdir(parents=True, exist_ok=True)

    summary = summarise(table)
    importance = train_forest(table, features)

    print("\nCharts:")
    pass_rate_chart(summary)
    error_chart(table, "brightness", "Brightness (mean grey level in the box, 0-255)",
                "error_vs_brightness.png")
    error_chart(table, "curve_px", "Curvature (true line's bow from straight, px)",
                "error_vs_curvature.png", x_max=60)
    if "line_contrast" in table:
        error_chart(table, "line_contrast",
                    "Line contrast (true line vs road beside it, grey levels)",
                    "error_vs_line_contrast.png")
    importance_chart(importance)


if __name__ == "__main__":
    main()
