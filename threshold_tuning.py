from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class ThresholdTuningResult:
    """Immutable result of a threshold sweep.

    Attributes:
        best_threshold: The threshold with the highest weighted_score (rank 1).
        best_weighted_score: The weighted_score achieved at best_threshold.
        results: The full per-threshold results DataFrame (see evaluate_thresholds).
        csv_path: Path where the results CSV was written.
        html_path: Path where the interactive HTML chart was written.
    """
    best_threshold: float
    best_weighted_score: float
    results: pd.DataFrame
    csv_path: Path
    html_path: Path


def build_threshold_grid(start: float = 0.05, end: float = 0.55, step: float = 0.01) -> list[float]:
    """Generate the list of candidate decision thresholds to sweep over.

    Args:
        start: First threshold value in the grid (inclusive).
        end: Last threshold value in the grid (inclusive).
        step: Spacing between consecutive threshold values.

    Returns:
        A list of threshold floats, rounded to 4 decimal places, from start to end.

    Raises:
        ValueError: If step is not positive, or end is less than start.
    """
    if step <= 0:
        raise ValueError("Threshold step must be positive.")
    if end < start:
        raise ValueError("Threshold end must be greater than or equal to start.")

    threshold_values = np.arange(start, end + (step / 2.0), step)
    return [float(round(value, 4)) for value in threshold_values]


def _detect_inflection_points(results: pd.DataFrame) -> pd.Series:
    """Flag threshold rows that are interesting turning points on the weighted-score curve.

    A row is flagged if the weighted_score's slope changes sign between the previous
    and next threshold (a local min/max), or if its curvature (the absolute change in
    slope) is in the top 10% of all rows. Used purely to annotate the threshold-sweep
    chart with diamond markers.

    Args:
        results: The per-threshold results DataFrame, sorted by threshold ascending,
            with a 'weighted_score' column.

    Returns:
        A boolean Series aligned to results.index; all False if fewer than 3 rows.
    """
    if len(results) < 3:
        return pd.Series([False] * len(results), index=results.index)

    weighted_score = results["weighted_score"]
    prev_slope = weighted_score.diff()
    next_slope = weighted_score.shift(-1) - weighted_score
    sign_change = (
        prev_slope.notna()
        & next_slope.notna()
        & ((prev_slope > 0) != (next_slope > 0))
    )

    curvature = (next_slope - prev_slope).abs().fillna(0.0)
    curvature_threshold = float(curvature.iloc[1:-1].quantile(0.9)) if len(results) > 3 else 0.0
    high_curvature = curvature >= curvature_threshold if curvature_threshold > 0 else pd.Series(False, index=results.index)
    return sign_change | high_curvature


def evaluate_thresholds(
    y_true: Iterable[int],
    y_prob_pos: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate classification metrics at every candidate threshold and rank them.

    Computes overall ROC-AUC and PR-AUC once (they don't depend on the threshold), then
    for each threshold in `thresholds` (or build_threshold_grid()'s default grid if not
    given) applies `y_prob_pos >= threshold` to get binary predictions and computes
    accuracy/precision/recall/f1_score, plus a composite weighted_score defined as
    `0.40*recall + 0.35*f1_score + 0.25*roc_auc` (roc_auc treated as 0 if undefined).
    Rows are then ranked by (weighted_score, recall, f1_score, threshold) descending/
    ascending tie-break, with rank 1 flagged as is_best_threshold, and
    is_inflection_point computed via _detect_inflection_points.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_prob_pos: Predicted positive-class probabilities, aligned with y_true.
        thresholds: Optional explicit list of thresholds to evaluate; defaults to
            build_threshold_grid()'s 0.05-0.55 step-0.01 grid.

    Returns:
        A DataFrame with one row per threshold, sorted by threshold ascending, with
        columns: threshold, accuracy, precision, recall, f1_score, roc_auc, pr_auc,
        weighted_score, positive_predictions, positive_prediction_rate,
        is_inflection_point, rank, and is_best_threshold.
    """
    y_true_array = np.asarray(list(y_true), dtype=int)
    y_prob_array = np.asarray(list(y_prob_pos), dtype=float)
    threshold_values = list(thresholds) if thresholds is not None else build_threshold_grid()

    roc_auc = float(roc_auc_score(y_true_array, y_prob_array)) if np.unique(y_true_array).size > 1 else float("nan")
    pr_auc = float(average_precision_score(y_true_array, y_prob_array)) if np.unique(y_true_array).size > 1 else float("nan")
    roc_auc_component = 0.0 if np.isnan(roc_auc) else roc_auc

    rows: list[dict[str, float | int]] = []
    for threshold in threshold_values:
        y_pred = (y_prob_array >= threshold).astype(int)
        recall = float(recall_score(y_true_array, y_pred, pos_label=1, zero_division=0))
        precision = float(precision_score(y_true_array, y_pred, pos_label=1, zero_division=0))
        f1_value = float(f1_score(y_true_array, y_pred, pos_label=1, zero_division=0))
        accuracy = float(accuracy_score(y_true_array, y_pred))
        weighted_score = (recall * 0.40) + (f1_value * 0.35) + (roc_auc_component * 0.25)

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_value,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "weighted_score": float(weighted_score),
                "positive_predictions": int(y_pred.sum()),
                "positive_prediction_rate": float(y_pred.mean()),
            }
        )

    results = pd.DataFrame(rows).sort_values("threshold", ascending=True).reset_index(drop=True)
    results["is_inflection_point"] = _detect_inflection_points(results)

    ranked = results.sort_values(
        ["weighted_score", "recall", "f1_score", "threshold"],
        ascending=[False, False, False, True],
    ).reset_index()
    ranked["rank"] = ranked.index + 1
    results = results.merge(ranked[["index", "rank"]], left_index=True, right_on="index", how="left")
    results = results.drop(columns=["index"])

    best_index = int(results["rank"].idxmin())
    results["is_best_threshold"] = False
    results.loc[best_index, "is_best_threshold"] = True
    return results


def write_threshold_report(results: pd.DataFrame, csv_path: Path, html_path: Path, title: str) -> None:
    """Persist threshold-sweep results as a CSV and an interactive Plotly HTML report.

    Writes the full `results` DataFrame to csv_path. Builds a 2-row HTML figure: the
    top row is a line chart of weighted_score/recall/f1_score/roc_auc/pr_auc vs.
    threshold (with inflection-point markers and a vertical dashed line at the best
    threshold); the bottom row is a table of the top-10 ranked thresholds. Saves the
    figure to html_path using the Plotly CDN for the JS bundle.

    Args:
        results: The per-threshold results DataFrame from evaluate_thresholds.
        csv_path: Destination path for the CSV; parent directories are created.
        html_path: Destination path for the HTML chart; parent directories are created.
        title: Title displayed at the top of the HTML chart.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(csv_path, index=False)

    best_row = results.loc[results["is_best_threshold"]].iloc[0]
    inflection_points = results.loc[results["is_inflection_point"]]
    top_ranked = results.sort_values("rank", ascending=True).head(10)

    figure = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        row_heights=[0.72, 0.28],
        vertical_spacing=0.12,
        subplot_titles=("Threshold sweep", "Top ranked thresholds"),
    )

    for column_name, display_name in [
        ("weighted_score", "Weighted score"),
        ("recall", "Recall"),
        ("f1_score", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=results["threshold"],
                y=results[column_name],
                mode="lines+markers",
                name=display_name,
            ),
            row=1,
            col=1,
        )

    if not inflection_points.empty:
        figure.add_trace(
            go.Scatter(
                x=inflection_points["threshold"],
                y=inflection_points["weighted_score"],
                mode="markers",
                name="Inflection points",
                marker={"size": 11, "symbol": "diamond", "color": "#d97706"},
            ),
            row=1,
            col=1,
        )

    figure.add_vline(
        x=float(best_row["threshold"]),
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Best threshold = {best_row['threshold']:.2f}",
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Table(
            header={"values": ["Rank", "Threshold", "Weighted score", "Recall", "F1", "ROC-AUC", "PR-AUC"]},
            cells={
                "values": [
                    top_ranked["rank"],
                    top_ranked["threshold"].map(lambda value: f"{value:.2f}"),
                    top_ranked["weighted_score"].map(lambda value: f"{value:.6f}"),
                    top_ranked["recall"].map(lambda value: f"{value:.6f}"),
                    top_ranked["f1_score"].map(lambda value: f"{value:.6f}"),
                    top_ranked["roc_auc"].map(lambda value: f"{value:.6f}" if pd.notna(value) else "nan"),
                    top_ranked["pr_auc"].map(lambda value: f"{value:.6f}" if pd.notna(value) else "nan"),
                ]
            },
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        title=title,
        template="plotly_white",
        height=900,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    figure.update_xaxes(title_text="Threshold", row=1, col=1)
    figure.update_yaxes(title_text="Metric value", row=1, col=1)
    figure.write_html(str(html_path), include_plotlyjs="cdn")


def tune_thresholds_from_probabilities(
    y_true: Iterable[int],
    y_prob_pos: Iterable[float],
    csv_path: str | Path,
    html_path: str | Path,
    title: str = "Threshold tuning report",
    thresholds: Iterable[float] | None = None,
) -> ThresholdTuningResult:
    """Convenience wrapper: sweep thresholds, write the CSV/HTML report, and return the best one.

    This is the function train.py calls right after training to pick the production
    decision threshold from the test split's predicted probabilities.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_prob_pos: Predicted positive-class probabilities, aligned with y_true.
        csv_path: Destination path for the results CSV.
        html_path: Destination path for the interactive HTML chart.
        title: Title displayed at the top of the HTML chart.
        thresholds: Optional explicit list of thresholds to evaluate; defaults to
            build_threshold_grid()'s default grid.

    Returns:
        A ThresholdTuningResult with the best threshold, its weighted score, the full
        results table, and the output file paths.
    """
    csv_output_path = Path(csv_path)
    html_output_path = Path(html_path)
    results = evaluate_thresholds(y_true=y_true, y_prob_pos=y_prob_pos, thresholds=thresholds)
    write_threshold_report(results=results, csv_path=csv_output_path, html_path=html_output_path, title=title)
    best_row = results.loc[results["is_best_threshold"]].iloc[0]
    return ThresholdTuningResult(
        best_threshold=float(best_row["threshold"]),
        best_weighted_score=float(best_row["weighted_score"]),
        results=results,
        csv_path=csv_output_path,
        html_path=html_output_path,
    )


def main() -> None:
    """CLI entry point for standalone threshold tuning.

    Reads a predictions CSV (target + positive-class probability columns), builds a
    threshold grid from --start/--end/--step, runs tune_thresholds_from_probabilities,
    and prints the chosen threshold plus the output CSV/HTML paths.
    """
    parser = argparse.ArgumentParser(description="Sweep classification thresholds and write CSV/HTML reports.")
    parser.add_argument("--predictions-csv", required=True, help="CSV containing target and positive-class probability columns.")
    parser.add_argument("--target-column", default="y_true", help="Binary target column name.")
    parser.add_argument("--probability-column", default="y_prob_pos", help="Positive-class probability column name.")
    parser.add_argument("--output-csv", default="artifacts/threshold_tuning.csv", help="Output CSV path.")
    parser.add_argument("--output-html", default="artifacts/threshold_tuning.html", help="Output HTML path.")
    parser.add_argument("--start", type=float, default=0.05, help="Threshold grid start.")
    parser.add_argument("--end", type=float, default=0.55, help="Threshold grid end.")
    parser.add_argument("--step", type=float, default=0.01, help="Threshold grid step.")
    args = parser.parse_args()

    predictions_df = pd.read_csv(args.predictions_csv)
    threshold_grid = build_threshold_grid(start=args.start, end=args.end, step=args.step)
    result = tune_thresholds_from_probabilities(
        y_true=predictions_df[args.target_column].astype(int),
        y_prob_pos=predictions_df[args.probability_column].astype(float),
        csv_path=args.output_csv,
        html_path=args.output_html,
        thresholds=threshold_grid,
    )

    print(f"Chosen decision threshold: {result.best_threshold:.2f}")
    print(f"Threshold tuning CSV: {result.csv_path}")
    print(f"Threshold tuning HTML: {result.html_path}")


if __name__ == "__main__":
    main()