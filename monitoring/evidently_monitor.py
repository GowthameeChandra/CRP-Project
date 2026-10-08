from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    f1_score,
    log_loss as sklearn_log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_INPUT_DIR = Path("monitoring/input")
DEFAULT_OUTPUT_DIR = Path("monitoring/output")
DEFAULT_TRAIN_PREDICTION_FILE = "development_prediction.csv"
DEFAULT_TEST_PREDICTION_FILE = "production_current_prediction.csv"
DEFAULT_REPORT_HTML = "evidently_report.html"
DEFAULT_REPORT_JSON = "evidently_report.json"
DEFAULT_REPORT_CSV = "evidently_monitoring_report.csv"
DEFAULT_REPORT_EXCEL = "evidently_monitoring_report.xlsx"
DEFAULT_MONITORING_CONFIG_PATH = Path("monitoring/monitoring_config.json")

# Statistical method behind each classification metric. These formulas/implementations
# are fixed regardless of dataset or batch, so a single reference column/value is used
# everywhere the metric is reported (as opposed to drift methods, which Evidently can
# select differently per column/batch and are therefore reported per-item).
CLASSIFICATION_METRIC_METHODS: dict[str, str] = {
    "accuracy": "Accuracy = (TP+TN)/Total (sklearn.metrics.accuracy_score)",
    "precision": "Precision = TP/(TP+FP) (sklearn.metrics.precision_score)",
    "recall": "Recall = TP/(TP+FN) (sklearn.metrics.recall_score)",
    "f1": "F1 = 2*(Precision*Recall)/(Precision+Recall) (sklearn.metrics.f1_score)",
    "roc_auc": "Area under ROC curve (sklearn.metrics.roc_auc_score)",
    "pr_auc": "Average Precision, area under Precision-Recall curve (sklearn.metrics.average_precision_score)",
    "log_loss": "Negative log-likelihood of predicted probabilities (sklearn.metrics.log_loss)",
    "lift_5pct": "Lift at top 5% = precision among highest-scored 5% / overall positive rate",
    "lift_10pct": "Lift at top 10% = precision among highest-scored 10% / overall positive rate",
    "lift_25pct": "Lift at top 25% = precision among highest-scored 25% / overall positive rate",
}

LIFT_METRIC_LABELS = [
    ("lift_5pct", "Lift@5%"),
    ("lift_10pct", "Lift@10%"),
    ("lift_25pct", "Lift@25%"),
]


def _drift_unavailable_reason(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    column: str,
) -> str:
    """Explain why Evidently produced no ValueDrift metric for a given column.

    Evidently silently skips drift computation for columns it can't meaningfully
    compare (absent from one dataset, entirely missing values, or constant/single-
    valued). This returns a human-readable reason so reports show an explanation
    instead of a bare "N/A".

    Args:
        reference_df: The reference/development dataset.
        current_df: The current/production (or batch) dataset.
        column: Name of the column to check.

    Returns:
        A short string explaining why drift wasn't computed for this column.
    """
    if column not in reference_df.columns or column not in current_df.columns:
        return "Not computed (column absent from one dataset)"

    reference_valid = int(reference_df[column].notna().sum())
    current_valid = int(current_df[column].notna().sum())

    if reference_valid == 0 and current_valid == 0:
        return "Not computed (column is 100% missing in both datasets)"
    if reference_valid == 0:
        return "Not computed (column is 100% missing in reference dataset)"
    if current_valid == 0:
        return "Not computed (column is 100% missing in current dataset)"
    if reference_df[column].nunique(dropna=True) <= 1 and current_df[column].nunique(dropna=True) <= 1:
        return "Not computed (constant column, no distribution to compare)"
    return "Not computed (Evidently skipped this column)"


def load_training_config(config_path: Path) -> dict[str, Any]:
    """Load and parse the training configuration JSON file.

    Args:
        config_path: Path to train_config.json. Opened with 'utf-8-sig' encoding so a
            leading byte-order-mark does not break parsing.

    Returns:
        The parsed config as a nested dict.
    """
    with config_path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_io_paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve the input prediction CSV paths and output report paths from config.

    Reads 'monitoring.input_dir'/'output_dir' plus file-name overrides, falling back to
    module-level DEFAULT_* constants. Supports both the current key names
    ('reference_prediction_file'/'current_prediction_file') and the legacy names
    ('train_prediction_file'/'test_prediction_file').

    Args:
        config: The full parsed training config dict.

    Returns:
        A tuple of (reference_prediction_path, current_prediction_path,
        report_html_path, report_json_path, report_csv_path).
    """
    monitoring_config = config.get("monitoring", {})

    input_dir = Path(monitoring_config.get("input_dir", str(DEFAULT_INPUT_DIR)))
    output_dir = Path(monitoring_config.get("output_dir", str(DEFAULT_OUTPUT_DIR)))

    train_name = str(monitoring_config.get(
        "reference_prediction_file",
        monitoring_config.get("train_prediction_file", DEFAULT_TRAIN_PREDICTION_FILE),
    ))
    test_name = str(monitoring_config.get(
        "current_prediction_file",
        monitoring_config.get("test_prediction_file", DEFAULT_TEST_PREDICTION_FILE),
    ))
    report_html_name = str(monitoring_config.get("report_html_file", DEFAULT_REPORT_HTML))
    report_json_name = str(monitoring_config.get("report_json_file", DEFAULT_REPORT_JSON))
    report_csv_name = str(monitoring_config.get("report_csv_file", DEFAULT_REPORT_CSV))

    return (
        input_dir / train_name,
        input_dir / test_name,
        output_dir / report_html_name,
        output_dir / report_json_name,
        output_dir / report_csv_name,
    )


def load_monitoring_runtime_config() -> dict[str, Any]:
    """Load the monitoring runtime config written by train.py after training.

    Reads monitoring/monitoring_config.json, which records which prediction CSVs to
    compare, their date ranges, and the batching mode to use. Never raises; any
    missing file or parse error results in an empty dict, so callers can safely fall
    back to config.json defaults.

    Returns:
        The parsed monitoring_config.json contents as a dict, or {} if the file is
        missing, unreadable, or not a JSON object.
    """
    if not DEFAULT_MONITORING_CONFIG_PATH.exists():
        return {}
    try:
        with DEFAULT_MONITORING_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def read_prediction_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    """Load a prediction CSV produced by train.py, validating it exists and has rows.

    Args:
        path: Path to the prediction CSV.
        dataset_name: Human-readable name used in error messages (e.g. "reference").

    Returns:
        The loaded prediction data as a DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file exists but contains no rows.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_name} prediction file at '{path}'. Run training first to generate prediction artifacts."
        )

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"{dataset_name} prediction file '{path}' is empty.")

    return dataframe


def detect_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    configured_target_column: str,
    configured_date_column: str | None = None,
) -> tuple[str, str | None, str | None, list[str]]:
    """Identify the target, prediction-label, prediction-probability, and feature columns.

    Since prediction CSVs can come from different pipeline versions, this matches
    column names against ordered candidate lists rather than assuming fixed names: the
    target is matched against the configured target column name (as-is and
    lowercased) plus common fallbacks ("target", "y_true", "actual", "label",
    "change_failure_flag"); the prediction label against
    ("prediction_label", "predicted_label", "prediction", "y_pred", "pred_label"); and
    the prediction probability against ("prediction_probability",
    "predicted_probability", "prediction_score", "score", "probability",
    "y_prob_pos"). Whichever candidate appears first in reference_df/current_df's
    common columns wins. Any remaining common column not identified as target/
    prediction/probability/date is treated as a feature column.

    Args:
        reference_df: The reference/development prediction dataset.
        current_df: The current/production prediction dataset.
        configured_target_column: The target column name from train_config.json,
            checked first (and case-insensitively) before the generic fallbacks.
        configured_date_column: Optional date column name to exclude from the
            feature list (it's metadata, not a model feature).

    Returns:
        A tuple of (target_column, prediction_column, probability_column,
        feature_columns). prediction_column and/or probability_column may be None if
        not found, but at least one of them must be present.

    Raises:
        ValueError: If the two datasets share no columns, no target column can be
            identified, neither a prediction label nor probability column can be
            identified, or no feature columns remain after exclusions.
    """
    common_columns = set(reference_df.columns).intersection(current_df.columns)
    if not common_columns:
        raise ValueError("Reference and current datasets do not share any common columns.")

    target_candidates = [
        configured_target_column,
        configured_target_column.strip().lower(),
        "target",
        "y_true",
        "actual",
        "label",
        "change_failure_flag",
    ]
    prediction_candidates = [
        "prediction_label",
        "predicted_label",
        "prediction",
        "y_pred",
        "pred_label",
    ]
    probability_candidates = [
        "prediction_probability",
        "predicted_probability",
        "prediction_score",
        "score",
        "probability",
        "y_prob_pos",
    ]

    target_column = next((name for name in target_candidates if name in common_columns), None)
    if target_column is None:
        raise ValueError(
            "Could not identify target column in prediction datasets. "
            f"Checked candidates: {target_candidates}"
        )

    prediction_column = next((name for name in prediction_candidates if name in common_columns), None)
    probability_column = next((name for name in probability_candidates if name in common_columns), None)

    if prediction_column is None and probability_column is None:
        raise ValueError(
            "Could not identify prediction label or prediction probability column in prediction datasets. "
            f"Checked label candidates: {prediction_candidates}; probability candidates: {probability_candidates}"
        )

    excluded = {target_column}
    if configured_date_column:
        date_match = next(
            (column for column in common_columns
             if str(column).strip().lower() == configured_date_column.strip().lower()),
            None,
        )
        if date_match is not None:
            excluded.add(date_match)
    if prediction_column is not None:
        excluded.add(prediction_column)
    if probability_column is not None:
        excluded.add(probability_column)

    feature_columns = [
        column
        for column in reference_df.columns
        if column in common_columns and column not in excluded
    ]

    if not feature_columns:
        raise ValueError("No feature columns available after excluding target/prediction columns.")

    return target_column, prediction_column, probability_column, feature_columns


def normalize_binary_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str,
    prediction_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Normalize the target (and prediction label, if present) to consistent 0/1 integers.

    Collects every distinct target value across both reference_df and current_df,
    verifies there are exactly two, sorts them lexicographically, and maps the first
    to 0 and the second to 1 (applied identically to the prediction label column when
    present). This guarantees Evidently and the sklearn metric functions see a
    consistent binary encoding regardless of the original label strings (e.g. "N"/"Y").

    Args:
        reference_df: The reference/development prediction dataset.
        current_df: The current/production (or batch) prediction dataset.
        target_column: Name of the target/actual-label column.
        prediction_column: Name of the predicted-label column, or None if absent.

    Returns:
        A tuple of (reference_copy, current_copy, pos_label) where pos_label is always
        1 (the second sorted label value).

    Raises:
        ValueError: If the combined target values across both datasets aren't exactly
            two distinct values, or if a column contains a value outside the two
            detected labels.
    """
    all_target_values = pd.concat([reference_df[target_column], current_df[target_column]], axis=0)
    unique_target_values = pd.Series(all_target_values).dropna().astype(str).str.strip().unique().tolist()
    sorted_values = sorted(unique_target_values)

    if len(sorted_values) != 2:
        raise ValueError(
            f"Expected binary target column '{target_column}', found values: {sorted_values}"
        )

    label_mapping = {sorted_values[0]: 0, sorted_values[1]: 1}

    def _map_binary(series: pd.Series) -> pd.Series:
        normalized = series.astype(str).str.strip()
        unknown_values = sorted(set(normalized.unique()) - set(label_mapping.keys()))
        if unknown_values:
            raise ValueError(f"Unexpected labels in binary mapping for '{series.name}': {unknown_values}")
        return normalized.map(label_mapping).astype(int)

    reference_copy = reference_df.copy()
    current_copy = current_df.copy()

    reference_copy[target_column] = _map_binary(reference_copy[target_column])
    current_copy[target_column] = _map_binary(current_copy[target_column])

    if prediction_column is not None:
        reference_copy[prediction_column] = _map_binary(reference_copy[prediction_column])
        current_copy[prediction_column] = _map_binary(current_copy[prediction_column])

    return reference_copy, current_copy, 1


def build_data_definition(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    prediction_column: str | None,
    probability_column: str | None,
    pos_label: int,
) -> DataDefinition:
    """Build the Evidently DataDefinition describing column roles/types for the Report.

    Classifies each feature column as numerical or categorical based on its dtype in
    the combined reference+current data, adds the probability column (if present) as
    numerical and the prediction-label column (if present) as categorical alongside
    the target, and wraps everything in a BinaryClassification spec so Evidently's
    ClassificationPreset/DataDriftPreset know which columns are target/prediction/
    probability.

    Args:
        reference_df: The (already 0/1-normalized) reference dataset.
        current_df: The (already 0/1-normalized) current dataset.
        feature_columns: List of model feature column names.
        target_column: Name of the target column.
        prediction_column: Name of the predicted-label column, or None.
        probability_column: Name of the predicted-probability column, or None.
        pos_label: The integer label value (always 1) considered the positive class.

    Returns:
        An evidently.DataDefinition ready to pass to Dataset.from_pandas.
    """
    combined = pd.concat([reference_df[feature_columns], current_df[feature_columns]], axis=0)

    numerical_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(combined[column]):
            numerical_columns.append(column)
        else:
            categorical_columns.append(column)

    if probability_column is not None:
        numerical_columns.append(probability_column)

    if prediction_column is not None:
        categorical_columns.append(prediction_column)

    categorical_columns.append(target_column)

    classification = [
        BinaryClassification(
            target=target_column,
            prediction_labels=prediction_column,
            prediction_probas=probability_column,
            pos_label=pos_label,
        )
    ]

    return DataDefinition(
        numerical_columns=sorted(set(numerical_columns)),
        categorical_columns=sorted(set(categorical_columns)),
        classification=classification,
    )


def _metric_type(metric: dict[str, Any]) -> str:
    """Return the metric's type name, stripped of its constructor arguments."""
    metric_name = str(metric.get("metric_name", ""))
    return metric_name.split("(", 1)[0]


def _extract_current_value(value: Any) -> Any:
    """Pull the 'current' value out of an Evidently metric value dict, if present."""
    if isinstance(value, dict) and "current" in value:
        return value["current"]
    return value


def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None instead of raising on failure."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_default(value: Any) -> Any:
    """JSON serializer fallback for Enums, Paths, and numpy scalars found in the Evidently snapshot."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def extract_summary(
    snapshot_dict: dict[str, Any],
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    prediction_column: str | None,
    probability_column: str | None,
    reference_identifier: str,
    current_identifier: str,
) -> dict[str, Any]:
    """Parse an Evidently Report snapshot into a consolidated drift/performance summary.

    Iterates snapshot_dict["metrics"] (the raw output of report.run().dict()) looking
    for three kinds of entries: (1) "DriftedColumnsCount", from which the overall
    drifted-feature count/share/threshold are read; (2) "ValueDrift" entries for each
    feature/target/prediction/probability column, from which per-column drift score,
    threshold, method, and a computed drift_detected flag (score > threshold) are
    extracted; and (3) classification metric types (Accuracy, Precision, Recall,
    F1Score, RocAuc, PrecisionRecallCurve, LogLoss), whose values are collected into a
    classification_metrics dict. The classification metrics from Evidently are then
    supplemented with metrics computed directly via sklearn
    (_compute_classification_metrics) on current_df, since Evidently may not always
    expose every metric depending on the current batch's class balance.

    Args:
        snapshot_dict: The dict returned by evidently's Report snapshot.dict().
        reference_df: The (normalized) reference/development dataset.
        current_df: The (normalized) current/production (or batch) dataset.
        feature_columns: List of model feature column names.
        target_column: Name of the target column.
        prediction_column: Name of the predicted-label column, or None.
        probability_column: Name of the predicted-probability column, or None.
        reference_identifier: A label (usually the file path) identifying the
            reference dataset, stored in the summary for traceability.
        current_identifier: A label (usually the file path) identifying the current
            dataset, stored in the summary for traceability.

    Returns:
        A dict with keys including: monitoring_run_utc, reference/current dataset
        identifiers and row counts, data_quality_status, overall_data_drift_status,
        number_of_monitored_features, number_of_drifted_features, drift_share,
        drift_share_threshold, target_drift_status, prediction_drift_status,
        classification_performance_metrics, feature_drift_details,
        target_drift_detail, and prediction_drift_details. This dict is the primary
        input to every report writer in this module (CSV, Excel, HTML).
    """
    metrics = snapshot_dict.get("metrics", [])

    drifted_columns_count = None
    drifted_columns_share = None
    drift_threshold = None
    feature_drift_details: list[dict[str, Any]] = []

    classification_metric_map = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1Score": "f1",
        "RocAuc": "roc_auc",
        "PrecisionRecallCurve": "pr_auc",
        "LogLoss": "log_loss",
    }
    classification_metrics: dict[str, float] = {}

    for metric in metrics:
        metric_type = _metric_type(metric)
        config = metric.get("config", {})
        metric_value = metric.get("value")

        if metric_type == "DriftedColumnsCount" and isinstance(metric_value, dict):
            drifted_columns_count = metric_value.get("count")
            drifted_columns_share = metric_value.get("share")
            drift_threshold = config.get("drift_share")

        if metric_type == "ValueDrift":
            column_name = config.get("column")
            if column_name in feature_columns or column_name in {target_column, prediction_column, probability_column}:
                score = _safe_float(_extract_current_value(metric_value))
                threshold = _safe_float(config.get("threshold"))
                method = config.get("method")
                drift_detected = None
                if score is not None and threshold is not None:
                    drift_detected = bool(score > threshold)

                feature_drift_details.append(
                    {
                        "feature_name": column_name,
                        "drift_detected": drift_detected,
                        "drift_score": score,
                        "threshold": threshold,
                        "method": method,
                    }
                )

        if metric_type in classification_metric_map:
            metric_key = classification_metric_map[metric_type]
            metric_score = _safe_float(_extract_current_value(metric_value))
            if metric_score is not None:
                classification_metrics[metric_key] = metric_score

    feature_only_drift_details = [item for item in feature_drift_details if item["feature_name"] in feature_columns]

    classification_metrics.update(
        _compute_classification_metrics(
            current_df,
            target_column,
            prediction_column,
            probability_column,
        )
    )

    target_drift_item = next((item for item in feature_drift_details if item["feature_name"] == target_column), None)
    prediction_drift_items = [
        item
        for item in feature_drift_details
        if item["feature_name"] in {prediction_column, probability_column}
    ]

    missing_cells = int(current_df.isna().sum().sum())
    data_quality_status = "ok" if missing_cells == 0 else "missing_values_detected"

    summary = {
        "monitoring_run_utc": datetime.now(timezone.utc).isoformat(),
        "reference_dataset_identifier": reference_identifier,
        "current_dataset_identifier": current_identifier,
        "total_reference_records": int(len(reference_df)),
        "total_current_records": int(len(current_df)),
        "data_quality_status": data_quality_status,
        "overall_data_drift_status": "drift_detected" if (drifted_columns_count or 0) > 0 else "no_drift_detected",
        "number_of_monitored_features": len(feature_columns),
        "number_of_drifted_features": int(drifted_columns_count) if drifted_columns_count is not None else None,
        "drift_share": drifted_columns_share,
        "drift_share_threshold": drift_threshold,
        "target_drift_status": None if target_drift_item is None else target_drift_item["drift_detected"],
        "prediction_drift_status": None
        if not prediction_drift_items
        else any(item["drift_detected"] is True for item in prediction_drift_items),
        "classification_performance_metrics": classification_metrics,
        "feature_drift_details": feature_only_drift_details,
        "target_drift_detail": target_drift_item,
        "prediction_drift_details": prediction_drift_items,
    }

    return summary


def _fmt(value: Any) -> str:
    """Format an arbitrary value for display in a report cell.

    Renders None as "N/A", bools as-is, whole-number floats without a decimal point,
    small floats to 6 decimal places, and large floats to 2 decimal places. Any other
    type is passed through str().

    Args:
        value: The value to format (typically a metric score, count, or share).

    Returns:
        A display-ready string.
    """
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6f}" if abs(value) < 1000 else f"{value:.2f}"
    return str(value)


def _find_metric_value(metrics: list[dict[str, Any]], name_prefix: str) -> Any:
    """Find a metric's value by its name prefix in the raw snapshot metrics list."""
    for m in metrics:
        if m.get("metric_name", "").startswith(name_prefix):
            return m.get("value")
    return None


def _compute_reference_classification_metrics(
    reference_df: pd.DataFrame,
    target_column: str,
    prediction_column: str | None,
    probability_column: str | None,
) -> dict[str, float]:
    """Compute classification metrics on the reference/development dataset via sklearn.

    Used as the fixed "before" baseline that every current/batch dataset's performance
    is compared against in reports, since Evidently's own metrics are computed per
    current-vs-reference run and this gives a stable reference-side value to reuse
    across many batches.

    Args:
        reference_df: The (normalized) reference/development prediction dataset.
        target_column: Name of the target column.
        prediction_column: Name of the predicted-label column, or None to skip
            accuracy/precision/recall/f1.
        probability_column: Name of the predicted-probability column, or None to skip
            roc_auc/pr_auc/log_loss/lift.

    Returns:
        Dict of metric name -> value. Metrics that fail to compute (e.g. roc_auc with
        only one class present) are silently omitted.
    """
    ref_metrics: dict[str, float] = {}
    ref_target = reference_df[target_column]
    if prediction_column is not None:
        ref_pred = reference_df[prediction_column]
        ref_metrics["accuracy"] = accuracy_score(ref_target, ref_pred)
        ref_metrics["precision"] = precision_score(ref_target, ref_pred, zero_division=0)
        ref_metrics["recall"] = recall_score(ref_target, ref_pred, zero_division=0)
        ref_metrics["f1"] = f1_score(ref_target, ref_pred, zero_division=0)
    if probability_column is not None:
        ref_prob = reference_df[probability_column]
        try:
            ref_metrics["roc_auc"] = roc_auc_score(ref_target, ref_prob)
        except (ValueError, TypeError):
            pass
        try:
            ref_metrics["pr_auc"] = average_precision_score(ref_target, ref_prob)
        except (ValueError, TypeError):
            pass
        try:
            ref_metrics["log_loss"] = sklearn_log_loss(ref_target, ref_prob)
        except (ValueError, TypeError):
            pass
        ref_metrics.update(_compute_lift_metrics(ref_target, ref_prob))
    return ref_metrics


def _compute_lift_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float]:
    """Compute lift-at-top-k% metrics for a target/probability pair.

    Lift@k% is the precision among the top-k%-highest-scored rows divided by the
    overall positive rate; a value > 1 means the model concentrates positives more
    effectively than random ranking would. Computed for k = 5%, 10%, and 25%. Uses the
    same formula as train.py's evaluate_split_metrics.

    Args:
        target: True binary labels (0/1) as a Series.
        probability: Predicted positive-class probabilities, aligned with target.

    Returns:
        Dict with keys "lift_5pct", "lift_10pct", "lift_25pct", or {} if target is empty.
    """
    y_true = target.astype(int).to_numpy()
    y_prob_pos = probability.astype(float).to_numpy()
    if len(y_true) == 0:
        return {}

    overall_positive_rate = float(y_true.mean()) if y_true.mean() > 0 else 1.0
    sorted_indices = y_prob_pos.argsort()[::-1]
    metrics: dict[str, float] = {}
    for pct in (0.05, 0.10, 0.25):
        k = max(1, int(len(y_true) * pct))
        top_k_positives = y_true[sorted_indices[:k]].sum()
        precision_at_k = top_k_positives / k
        metrics[f"lift_{int(pct * 100)}pct"] = float(precision_at_k / overall_positive_rate)
    return metrics


def _compute_classification_metrics(
    dataframe: pd.DataFrame,
    target_column: str,
    prediction_column: str | None,
    probability_column: str | None,
) -> dict[str, float]:
    """Compute classification metrics directly from saved predictions via sklearn.

    Same metric set and logic as _compute_reference_classification_metrics, but for any
    given dataframe (used for the current/production side, or an individual monitoring
    batch). Returns an empty dict if the dataframe's target column doesn't contain
    exactly two distinct classes, since metrics like ROC-AUC are undefined in that case.

    Args:
        dataframe: The (normalized) dataset to score (current or a single batch).
        target_column: Name of the target column.
        prediction_column: Name of the predicted-label column, or None to skip
            accuracy/precision/recall/f1.
        probability_column: Name of the predicted-probability column, or None to skip
            roc_auc/pr_auc/log_loss/lift.

    Returns:
        Dict of metric name -> value, or {} if the target isn't binary in this slice.
    """
    if dataframe[target_column].nunique(dropna=True) != 2:
        return {}

    metrics: dict[str, float] = {}
    target = dataframe[target_column]
    if prediction_column is not None:
        prediction = dataframe[prediction_column]
        metrics["accuracy"] = float(accuracy_score(target, prediction))
        metrics["precision"] = float(precision_score(target, prediction, zero_division=0))
        metrics["recall"] = float(recall_score(target, prediction, zero_division=0))
        metrics["f1"] = float(f1_score(target, prediction, zero_division=0))
    if probability_column is not None:
        probability = dataframe[probability_column]
        try:
            metrics["roc_auc"] = float(roc_auc_score(target, probability))
            metrics["pr_auc"] = float(average_precision_score(target, probability))
            metrics["log_loss"] = float(sklearn_log_loss(target, probability))
        except (ValueError, TypeError):
            pass
        metrics.update(_compute_lift_metrics(target, probability))
    return metrics


def generate_monitoring_csv(
    csv_path: Path,
    snapshot_dict: dict[str, Any],
    summary: dict[str, Any],
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str,
    prediction_column: str | None,
    probability_column: str | None,
    feature_columns: list[str],
) -> Path:
    """Write the full single-run monitoring report as a flat, multi-section CSV.

    Builds a row-accumulator (list of row lists) covering nine sections: (1) run info
    (dates, dataset identifiers, row/column counts, drift summary); (2) dataset column
    configuration (target/prediction/probability roles and dtypes); (3) drift detection
    methodology (which statistical method Evidently used per drift group, with a
    description); (4) monitoring metrics (data quality, drift, and classification
    performance, each reference-vs-current with a Status column); (5) per-feature
    drift details (reference/current statistics, method, score, threshold, status);
    (6) target class distribution; (7) prediction label/probability distribution; (8)
    a classification-performance-only recap; and (9) an overall status summary
    (data quality, data drift, target drift, prediction drift, model performance). The
    accumulated rows are written with the stdlib csv module.

    Args:
        csv_path: Destination path; parent directories are created if needed.
        snapshot_dict: The raw Evidently snapshot dict (used for column-count and
            data-quality metric lookups via _find_metric_value).
        summary: The consolidated summary dict from extract_summary.
        reference_df: The (normalized) reference/development dataset.
        current_df: The (normalized) current/production dataset.
        target_column: Name of the target column.
        prediction_column: Name of the predicted-label column, or None.
        probability_column: Name of the predicted-probability column, or None.
        feature_columns: List of model feature column names.

    Returns:
        The csv_path that was written to.
    """
    metrics = snapshot_dict.get("metrics", [])
    rows: list[list[str]] = []

    def blank_rows(n: int = 3) -> None:
        for _ in range(n):
            rows.append([])

    # ── SECTION 1: Monitoring Run Information ──
    try:
        import evidently as _ev
        ev_version = _ev.__version__
    except Exception:
        ev_version = "N/A"

    ref_missing = int(reference_df.isna().sum().sum())
    cur_missing = int(current_df.isna().sum().sum())
    num_cols = _find_metric_value(metrics, "ColumnCount(column_type=ColumnType.Numerical)")
    cat_cols = _find_metric_value(metrics, "ColumnCount(column_type=ColumnType.Categorical)")
    dt_cols = _find_metric_value(metrics, "ColumnCount(column_type=ColumnType.Datetime)")
    text_cols = _find_metric_value(metrics, "ColumnCount(column_type=ColumnType.Text)")
    total_cols = _find_metric_value(metrics, "ColumnCount()")

    rows.append(["MONITORING RUN INFORMATION"])
    rows.append(["Metric", "Value"])
    rows.append(["Monitoring Run Date", summary.get("monitoring_run_utc", "N/A")])
    rows.append(["Evidently Version", ev_version])
    rows.append(["Reference Dataset", summary.get("reference_dataset_identifier", "N/A")])
    rows.append(["Current Dataset", summary.get("current_dataset_identifier", "N/A")])
    rows.append(["Reference Row Count", str(summary.get("total_reference_records", "N/A"))])
    rows.append(["Current Row Count", str(summary.get("total_current_records", "N/A"))])
    rows.append(["Number of Columns", _fmt(total_cols)])
    rows.append(["Number of Monitored Features", str(summary.get("number_of_monitored_features", "N/A"))])
    rows.append(["Number of Numerical Columns", _fmt(num_cols)])
    rows.append(["Number of Categorical Columns", _fmt(cat_cols)])
    rows.append(["Number of Text Columns", _fmt(text_cols)])
    rows.append(["Number of Datetime Columns", _fmt(dt_cols)])
    rows.append(["Missing Values in Reference", str(ref_missing)])
    rows.append(["Missing Values in Current", str(cur_missing)])
    rows.append(["Overall Dataset Drift Status", summary.get("overall_data_drift_status", "N/A")])
    rows.append(["Number of Drifted Features", str(summary.get("number_of_drifted_features", "N/A"))])
    rows.append(["Drifted Feature Share", _fmt(summary.get("drift_share"))])
    rows.append(["Dataset Drift Threshold", _fmt(summary.get("drift_share_threshold"))])
    blank_rows()

    # ── SECTION 2: Dataset Column Configuration ──
    rows.append(["DATASET COLUMN CONFIGURATION"])
    rows.append(["Column Type", "Column Name", "Data Type"])
    rows.append(["ID", "None", "N/A"])
    rows.append(["Target", target_column, str(reference_df[target_column].dtype)])
    rows.append([
        "Prediction Label",
        prediction_column or "None",
        str(reference_df[prediction_column].dtype) if prediction_column else "N/A",
    ])
    rows.append([
        "Prediction Probability",
        probability_column or "None",
        str(reference_df[probability_column].dtype) if probability_column else "N/A",
    ])
    rows.append(["Date", "None", "N/A"])
    rows.append(["Number of Feature Columns", str(len(feature_columns)), ""])
    blank_rows()

    # ── SECTION 3: Drift Detection Methodology ──
    all_drift_items = list(summary.get("feature_drift_details", []))
    target_drift_detail = summary.get("target_drift_detail")
    if target_drift_detail:
        all_drift_items.append(target_drift_detail)
    for pd_item in summary.get("prediction_drift_details", []):
        all_drift_items.append(pd_item)

    method_info: dict[str, dict[str, Any]] = {}
    for item in all_drift_items:
        method = item.get("method") or "Unknown"
        fname = item.get("feature_name", "")
        if method not in method_info:
            method_info[method] = {"features": [], "threshold": item.get("threshold")}
        method_info[method]["features"].append(fname)

    method_descriptions = {
        "Jensen-Shannon distance": "Measures divergence between two probability distributions",
        "Wasserstein distance (normed)": "Measures the normalized distance between two distributions",
        "Chi-square test": "Tests independence between categorical distributions",
        "Kolmogorov-Smirnov test": "Compares cumulative distribution functions",
        "PSI": "Population Stability Index for distribution comparison",
    }

    rows.append(["DRIFT DETECTION METHODOLOGY"])
    rows.append(["Method Type", "Statistical Method", "Applied To", "Threshold", "Feature Count", "Description"])

    for method, info in method_info.items():
        feature_types: set[str] = set()
        for f in info["features"]:
            if f == target_column:
                feature_types.add("Target")
            elif f == prediction_column:
                feature_types.add("Prediction label")
            elif f == probability_column:
                feature_types.add("Prediction probability")
            elif f in feature_columns:
                if pd.api.types.is_numeric_dtype(reference_df[f]):
                    feature_types.add("Numerical features")
                else:
                    feature_types.add("Categorical features")
        applied_to = ", ".join(sorted(feature_types))
        desc = method_descriptions.get(method, "Evidently-selected statistical method")
        rows.append([
            "Data Drift", method, applied_to, _fmt(info["threshold"]),
            str(len(info["features"])), desc,
        ])

    rows.append([])
    rows.append(["Note", "Automatic Evidently method selection was used. Methods listed above are the actual methods applied by Evidently in this run."])
    blank_rows()

    # ── SECTION 4: Monitoring Metrics ──
    rows.append(["MONITORING METRICS"])

    # Data Quality
    rows.append(["--- Data Quality ---"])
    rows.append(["Metric", "Reference", "Current", "Change", "Status"])

    ref_row_count = summary.get("total_reference_records", 0)
    cur_row_count = summary.get("total_current_records", 0)
    empty_cols_val = _find_metric_value(metrics, "EmptyColumnsCount()")
    constant_cols_val = _find_metric_value(metrics, "ConstantColumnsCount()")
    almost_constant_val = _find_metric_value(metrics, "AlmostConstantColumnsCount()")
    cur_missing_metric = _find_metric_value(metrics, "DatasetMissingValueCount()")
    cur_missing_count = cur_missing_metric.get("count", 0) if isinstance(cur_missing_metric, dict) else 0
    cur_missing_share = cur_missing_metric.get("share", 0) if isinstance(cur_missing_metric, dict) else 0
    ref_total_cells = ref_row_count * len(reference_df.columns)
    ref_missing_share = ref_missing / ref_total_cells if ref_total_cells > 0 else 0

    rows.append(["Reference Rows", str(ref_row_count), "", "", ""])
    rows.append(["Current Rows", "", str(cur_row_count), "", ""])
    rows.append(["Reference Missing Values", str(ref_missing), "", "", ""])
    rows.append(["Current Missing Values", "", str(int(cur_missing_count)), "", ""])
    rows.append(["Reference Missing Value Rate", _fmt(ref_missing_share), "", "", ""])
    rows.append(["Current Missing Value Rate", "", _fmt(cur_missing_share), "", ""])
    rows.append(["Empty Columns", "", _fmt(empty_cols_val), "", "Healthy" if (empty_cols_val or 0) == 0 else "Issue"])
    rows.append(["Constant Columns", "", _fmt(constant_cols_val), "", "Healthy" if (constant_cols_val or 0) == 0 else "Issue"])
    rows.append(["Almost Constant Columns", "", _fmt(almost_constant_val), "", "Healthy" if (almost_constant_val or 0) == 0 else "Warning"])
    rows.append([])

    # Drift
    rows.append(["--- Drift ---"])
    rows.append(["Metric", "Value", "", "", "Status"])

    n_monitored = summary.get("number_of_monitored_features", 0)
    n_drifted = summary.get("number_of_drifted_features", 0)
    drift_share = summary.get("drift_share")
    drift_status = summary.get("overall_data_drift_status", "N/A")
    drift_threshold = summary.get("drift_share_threshold")

    rows.append(["Number of Monitored Features", str(n_monitored), "", "", ""])
    rows.append(["Number of Drifted Features", str(n_drifted), "", "", ""])
    rows.append(["Drifted Feature Share", _fmt(drift_share), "", "", ""])
    rows.append(["Dataset Drift Status", drift_status, "", "", drift_status])
    rows.append(["Dataset Drift Threshold", _fmt(drift_threshold), "", "", ""])

    if target_drift_detail:
        rows.append(["Target Drift Method", target_drift_detail.get("method", "N/A"), "", "", ""])
        rows.append(["Target Drift Score", _fmt(target_drift_detail.get("drift_score")), "", "", "Drift" if target_drift_detail.get("drift_detected") else "No Drift"])
        rows.append(["Target Drift Status", str(target_drift_detail.get("drift_detected", "N/A")), "", "", ""])
    prediction_drift_details = summary.get("prediction_drift_details", [])
    for pd_item in prediction_drift_details:
        col_name = pd_item.get("feature_name", "prediction")
        rows.append([f"{col_name} Drift Method", pd_item.get("method", "N/A"), "", "", ""])
        rows.append([f"{col_name} Drift Score", _fmt(pd_item.get("drift_score")), "", "", "Drift" if pd_item.get("drift_detected") else "No Drift"])
        rows.append([f"{col_name} Drift Status", str(pd_item.get("drift_detected", "N/A")), "", "", ""])
    rows.append([])

    # Classification Performance
    rows.append(["--- Classification Performance ---"])
    rows.append(["Metric", "Reference", "Current", "Change", "Status", "Statistical Method"])

    cur_class_metrics = summary.get("classification_performance_metrics", {})
    ref_class_metrics = _compute_reference_classification_metrics(
        reference_df, target_column, prediction_column, probability_column,
    )

    perf_labels = [
        ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
        ("f1", "F1"), ("roc_auc", "ROC AUC"), ("log_loss", "LogLoss"),
    ]
    for key, label in perf_labels:
        cur_val = cur_class_metrics.get(key)
        ref_val = ref_class_metrics.get(key)
        if cur_val is None and ref_val is None:
            continue
        change = (cur_val - ref_val) if cur_val is not None and ref_val is not None else None
        status = ""
        if change is not None:
            if key == "log_loss":
                status = "Improved" if change < -1e-9 else ("Degraded" if change > 1e-9 else "Unchanged")
            else:
                status = "Improved" if change > 1e-9 else ("Degraded" if change < -1e-9 else "Unchanged")
        rows.append([label, _fmt(ref_val), _fmt(cur_val), _fmt(change), status, CLASSIFICATION_METRIC_METHODS.get(key, "N/A")])
    blank_rows()

    # ── SECTION 5: Feature Drift Details ──
    rows.append(["FEATURE DRIFT DETAILS"])
    rows.append([
        "Feature", "Feature Type", "Reference Statistic", "Current Statistic",
        "Statistical Method", "Drift Score", "Drift Threshold", "Drift Detected", "Status",
    ])

    feature_drift_map = {d["feature_name"]: d for d in summary.get("feature_drift_details", [])}

    for feat in feature_columns:
        is_numeric = pd.api.types.is_numeric_dtype(reference_df[feat])
        feat_type = "numerical" if is_numeric else "categorical"
        if is_numeric:
            ref_stat = f"mean={reference_df[feat].mean():.6f}, std={reference_df[feat].std():.6f}"
            cur_stat = f"mean={current_df[feat].mean():.6f}, std={current_df[feat].std():.6f}"
        else:
            ref_mode = reference_df[feat].mode()
            cur_mode = current_df[feat].mode()
            ref_stat = f"mode={ref_mode.iloc[0]}" if len(ref_mode) > 0 else "N/A"
            cur_stat = f"mode={cur_mode.iloc[0]}" if len(cur_mode) > 0 else "N/A"

        drift_info = feature_drift_map.get(feat, {})
        method = drift_info.get("method") or _drift_unavailable_reason(reference_df, current_df, feat)
        score = drift_info.get("drift_score")
        threshold = drift_info.get("threshold")
        detected = drift_info.get("drift_detected")
        status = "No Drift" if detected is False else ("Drift" if detected is True else "Not Evaluated")
        rows.append([
            feat, feat_type, ref_stat, cur_stat, method,
            _fmt(score), _fmt(threshold), str(detected) if detected is not None else "N/A", status,
        ])
    blank_rows()

    # ── SECTION 6: Target Distribution ──
    rows.append(["TARGET DISTRIBUTION"])
    rows.append(["Metric", "Value"])
    rows.append(["Target Column", target_column])
    rows.append(["Target Data Type", str(reference_df[target_column].dtype)])

    ref_target_counts = reference_df[target_column].value_counts()
    cur_target_counts = current_df[target_column].value_counts()
    ref_total = len(reference_df)
    cur_total = len(current_df)

    if target_drift_detail:
        rows.append(["Target Drift Method", target_drift_detail.get("method", "N/A")])
        rows.append(["Target Drift Score", _fmt(target_drift_detail.get("drift_score"))])
        rows.append(["Target Drift Threshold", _fmt(target_drift_detail.get("threshold"))])
        rows.append(["Target Drift Detected", str(target_drift_detail.get("drift_detected", "N/A"))])

    rows.append([])
    rows.append(["Target", "Class", "Reference Count", "Reference %", "Current Count", "Current %"])
    all_classes = sorted(set(ref_target_counts.index.tolist() + cur_target_counts.index.tolist()), key=str)
    for cls in all_classes:
        ref_cnt = int(ref_target_counts.get(cls, 0))
        cur_cnt = int(cur_target_counts.get(cls, 0))
        ref_pct = ref_cnt / ref_total * 100 if ref_total > 0 else 0
        cur_pct = cur_cnt / cur_total * 100 if cur_total > 0 else 0
        rows.append([target_column, str(cls), str(ref_cnt), f"{ref_pct:.4f}%", str(cur_cnt), f"{cur_pct:.4f}%"])
    blank_rows()

    # ── SECTION 7: Prediction Distribution ──
    rows.append(["PREDICTION DISTRIBUTION"])
    rows.append(["Metric", "Value"])
    rows.append(["Prediction Label Column", prediction_column or "None"])
    rows.append(["Prediction Probability Column", probability_column or "None"])

    if prediction_column:
        ref_pred_counts = reference_df[prediction_column].value_counts()
        cur_pred_counts = current_df[prediction_column].value_counts()
        pred_drift_item = next(
            (p for p in prediction_drift_details if p.get("feature_name") == prediction_column), None,
        )
        if pred_drift_item:
            rows.append(["Prediction Drift Method", pred_drift_item.get("method", "N/A")])
            rows.append(["Prediction Drift Score", _fmt(pred_drift_item.get("drift_score"))])
            rows.append(["Prediction Drift Status", "Drift" if pred_drift_item.get("drift_detected") else "No Drift"])

        rows.append([])
        rows.append(["Column", "Class", "Reference Count", "Reference %", "Current Count", "Current %"])
        pred_classes = sorted(set(ref_pred_counts.index.tolist() + cur_pred_counts.index.tolist()), key=str)
        for cls in pred_classes:
            ref_cnt = int(ref_pred_counts.get(cls, 0))
            cur_cnt = int(cur_pred_counts.get(cls, 0))
            ref_pct = ref_cnt / ref_total * 100 if ref_total > 0 else 0
            cur_pct = cur_cnt / cur_total * 100 if cur_total > 0 else 0
            rows.append([prediction_column, str(cls), str(ref_cnt), f"{ref_pct:.4f}%", str(cur_cnt), f"{cur_pct:.4f}%"])

    if probability_column:
        ref_avg_prob = reference_df[probability_column].mean()
        cur_avg_prob = current_df[probability_column].mean()
        rows.append([])
        rows.append(["Probability Statistic", "Reference", "Current"])
        rows.append(["Average Probability", _fmt(ref_avg_prob), _fmt(cur_avg_prob)])
        rows.append(["Std Probability", _fmt(reference_df[probability_column].std()), _fmt(current_df[probability_column].std())])

        prob_drift_item = next(
            (p for p in prediction_drift_details if p.get("feature_name") == probability_column), None,
        )
        if prob_drift_item:
            rows.append(["Probability Drift Method", prob_drift_item.get("method", "N/A"), ""])
            rows.append(["Probability Drift Score", _fmt(prob_drift_item.get("drift_score")), ""])
            rows.append(["Probability Drift Status", "Drift" if prob_drift_item.get("drift_detected") else "No Drift", ""])
    blank_rows()

    # ── SECTION 8: Classification Performance ──
    rows.append(["CLASSIFICATION PERFORMANCE"])
    rows.append(["Metric", "Value"])
    rows.append(["Target Column", target_column])
    rows.append(["Prediction Column", prediction_column or "None"])
    rows.append(["Prediction Probability Column", probability_column or "None"])
    rows.append([])
    rows.append(["Metric", "Reference", "Current", "Difference", "Status", "Statistical Method"])

    for key, label in perf_labels:
        cur_val = cur_class_metrics.get(key)
        ref_val = ref_class_metrics.get(key)
        if cur_val is None and ref_val is None:
            continue
        diff = (cur_val - ref_val) if cur_val is not None and ref_val is not None else None
        status = ""
        if diff is not None:
            if key == "log_loss":
                status = "Improved" if diff < -1e-9 else ("Degraded" if diff > 1e-9 else "Unchanged")
            else:
                status = "Improved" if diff > 1e-9 else ("Degraded" if diff < -1e-9 else "Unchanged")
        rows.append([label, _fmt(ref_val), _fmt(cur_val), _fmt(diff), status, CLASSIFICATION_METRIC_METHODS.get(key, "N/A")])
    blank_rows()

    # ── SECTION 9: Monitoring Status Summary ──
    rows.append(["MONITORING STATUS SUMMARY"])
    rows.append(["Area", "Status", "Explanation"])

    # Data Quality
    dq_status = summary.get("data_quality_status", "ok")
    if dq_status == "ok":
        rows.append(["Data Quality", "Healthy", "No major data-quality issue detected"])
    else:
        rows.append(["Data Quality", "Warning", f"Missing values detected (reference: {ref_missing}, current: {int(cur_missing_count)})"])

    # Data Drift
    if n_drifted == 0:
        rows.append(["Data Drift", "No Drift", f"0 of {n_monitored} monitored features drifted"])
    else:
        rows.append(["Data Drift", "Drift Detected", f"{n_drifted} of {n_monitored} monitored features drifted"])

    # Target Drift
    if target_drift_detail:
        if target_drift_detail.get("drift_detected"):
            rows.append(["Target Drift", "Drift Detected", f"Target distribution changed (score: {_fmt(target_drift_detail.get('drift_score'))})"])
        else:
            rows.append(["Target Drift", "No Drift", "Target distribution remained stable"])
    else:
        rows.append(["Target Drift", "N/A", "Target drift not computed"])

    # Prediction Drift
    any_pred_drift = any(p.get("drift_detected") is True for p in prediction_drift_details)
    if prediction_drift_details:
        if any_pred_drift:
            rows.append(["Prediction Drift", "Drift Detected", "Prediction distribution changed"])
        else:
            rows.append(["Prediction Drift", "No Drift", "Prediction distribution remained stable"])
    else:
        rows.append(["Prediction Drift", "N/A", "Prediction drift not computed"])

    # Model Performance
    perf_changes: list[str] = []
    for key, label in perf_labels:
        cur_val = cur_class_metrics.get(key)
        ref_val = ref_class_metrics.get(key)
        if cur_val is not None and ref_val is not None:
            diff = cur_val - ref_val
            better = diff < 0 if key == "log_loss" else diff > 0
            worse = diff > 0 if key == "log_loss" else diff < 0
            if worse and abs(diff) > 1e-4:
                perf_changes.append(f"{label} decreased by {abs(diff):.6f}")
    if not cur_class_metrics:
        rows.append(["Model Performance", "N/A", "Classification metrics not available"])
    elif perf_changes:
        rows.append(["Model Performance", "Degraded", "; ".join(perf_changes)])
    else:
        rows.append(["Model Performance", "Stable", "Current performance is comparable to or better than reference"])

    # Write CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return csv_path


def _run_single_monitoring(
    config_path: Path,
    reference_df_override: pd.DataFrame | None = None,
    current_df_override: pd.DataFrame | None = None,
    report_html_override: Path | None = None,
    report_json_override: Path | None = None,
    report_csv_override: Path | None = None,
    write_outputs: bool = True,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, str | list[str]], dict[str, Any]]:
    """Run a single reference-vs-current Evidently drift/performance comparison.

    This is the core comparison routine reused both for the whole current dataset
    (single monitoring run) and for each individual batch window (called by
    run_monitoring with write_outputs=False and df overrides). Steps: (1) load config
    and resolve I/O paths; (2) read (or accept overrides for) the reference/current
    prediction dataframes; (3) detect target/prediction/probability/feature columns
    via detect_columns; (4) normalize the target/prediction labels to 0/1 via
    normalize_binary_columns; (5) build an Evidently DataDefinition; (6) skip any
    feature column that is 100% missing on either side (drift can't be computed for
    it); (7) assemble a Report with DataSummaryPreset, DataDriftPreset (on the
    non-empty features), ValueDrift for target/prediction/probability, and
    ClassificationPreset (only if both datasets contain both binary classes); (8) run
    the report and extract a JSON-able snapshot; (9) call extract_summary to build the
    consolidated summary dict; and (10) if write_outputs is True, save the Evidently
    HTML report, a JSON payload (summary + column_mapping + full snapshot), and the
    flat CSV report via generate_monitoring_csv.

    Args:
        config_path: Path to train_config.json.
        reference_df_override: If given, used instead of reading the reference CSV
            from disk (used when batching, since the reference never changes per batch).
        current_df_override: If given, used instead of reading the current CSV from
            disk (used to pass a single batch's rows).
        report_html_override: If given, overrides the resolved HTML report path.
        report_json_override: If given, overrides the resolved JSON report path.
        report_csv_override: If given, overrides the resolved CSV report path.
        write_outputs: If False, skips writing any files and only returns the computed
            summary/snapshot (used for per-batch runs, where outputs are aggregated
            and written once at the end by run_monitoring).

    Returns:
        A tuple of (report_html_path, report_json_path, report_csv_path, summary,
        resolved_columns, snapshot_dict), where resolved_columns is a dict with
        target_column/prediction_column/prediction_probability_column/feature_columns.
    """
    config = load_training_config(config_path)
    configured_target_column = str(config.get("model", {}).get("target", "change_failure_flag"))
    configured_date_column = config.get("dataset", {}).get("date_column")

    train_path, test_path, report_html_path, report_json_path, report_csv_path = resolve_io_paths(config)
    if report_html_override is not None:
        report_html_path = report_html_override
    if report_json_override is not None:
        report_json_path = report_json_override
    if report_csv_override is not None:
        report_csv_path = report_csv_override

    reference_df = reference_df_override if reference_df_override is not None else read_prediction_dataset(train_path, "train")
    current_df = current_df_override if current_df_override is not None else read_prediction_dataset(test_path, "test")

    target_column, prediction_column, probability_column, feature_columns = detect_columns(
        reference_df=reference_df,
        current_df=current_df,
        configured_target_column=configured_target_column,
        configured_date_column=str(configured_date_column) if configured_date_column else None,
    )

    normalized_reference_df, normalized_current_df, pos_label = normalize_binary_columns(
        reference_df=reference_df,
        current_df=current_df,
        target_column=target_column,
        prediction_column=prediction_column,
    )

    data_definition = build_data_definition(
        reference_df=normalized_reference_df,
        current_df=normalized_current_df,
        feature_columns=feature_columns,
        target_column=target_column,
        prediction_column=prediction_column,
        probability_column=probability_column,
        pos_label=pos_label,
    )

    drift_feature_columns = [
        column
        for column in feature_columns
        if normalized_reference_df[column].notna().any()
        and normalized_current_df[column].notna().any()
    ]
    skipped_drift_columns = [
        column for column in feature_columns if column not in drift_feature_columns
    ]
    if skipped_drift_columns:
        print(
            "Skipping empty features for drift calculation: "
            f"{skipped_drift_columns}"
        )

    report_metrics: list[Any] = [
        DataSummaryPreset(),
        DataDriftPreset(columns=drift_feature_columns),
        ValueDrift(column=target_column),
    ]

    if prediction_column is not None:
        report_metrics.append(ValueDrift(column=prediction_column))
    if probability_column is not None:
        report_metrics.append(ValueDrift(column=probability_column))

    classification_ready = all(
        set(dataframe[target_column].dropna().unique()) == {0, 1}
        for dataframe in (normalized_reference_df, normalized_current_df)
    )
    if classification_ready:
        report_metrics.append(ClassificationPreset())
    else:
        print(
            "Classification metrics skipped: reference/current data does not "
            "contain both binary classes in the target column."
        )

    current_dataset = Dataset.from_pandas(
        normalized_current_df,
        data_definition=data_definition,
    )
    reference_dataset = Dataset.from_pandas(
        normalized_reference_df,
        data_definition=data_definition,
    )

    report = Report(report_metrics)
    snapshot = report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )

    snapshot_dict = snapshot.dict()
    summary = extract_summary(
        snapshot_dict=snapshot_dict,
        reference_df=normalized_reference_df,
        current_df=normalized_current_df,
        feature_columns=feature_columns,
        target_column=target_column,
        prediction_column=prediction_column,
        probability_column=probability_column,
        reference_identifier=str(train_path),
        current_identifier=str(test_path),
    )

    output_payload = {
        "summary": summary,
        "column_mapping": {
            "feature_columns": feature_columns,
            "target_column": target_column,
            "prediction_column": prediction_column,
            "prediction_probability_column": probability_column,
        },
        "evidently_snapshot": snapshot_dict,
    }

    if write_outputs:
        report_html_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(report_html_path))
        with report_json_path.open("w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2, default=_json_default)
        report_csv_path = generate_monitoring_csv(
            csv_path=report_csv_path,
            snapshot_dict=snapshot_dict,
            summary=summary,
            reference_df=normalized_reference_df,
            current_df=normalized_current_df,
            target_column=target_column,
            prediction_column=prediction_column,
            probability_column=probability_column,
            feature_columns=feature_columns,
        )

    resolved_columns = {
        "target_column": target_column,
        "prediction_column": prediction_column,
        "prediction_probability_column": probability_column,
        "feature_columns": feature_columns,
    }

    return report_html_path, report_json_path, report_csv_path, summary, resolved_columns, snapshot_dict


def _batch_metric_value(snapshot_dict: dict[str, Any], prefix: str) -> Any:
    """Look up a metric's value by name prefix within a single batch's Evidently snapshot.

    Args:
        snapshot_dict: One batch's raw Evidently snapshot dict (item["snapshot"] from
            a batch_results entry).
        prefix: The metric_name prefix to match (e.g. "EmptyColumnsCount()").

    Returns:
        The matching metric's value, or None if not found.
    """
    return _find_metric_value(snapshot_dict.get("metrics", []), prefix)


def _write_batch_csv(
    csv_path: Path,
    reference_df: pd.DataFrame,
    batch_results: list[dict[str, Any]],
    feature_columns: list[str],
    target_column: str,
    prediction_column: str | None,
    probability_column: str | None,
    evidently_version: str,
    batch_size_label: str = "14 days",
    batch_mode: str = "fixed_2_week",
) -> Path:
    """Write a flat, multi-section CSV summarizing every monitoring batch's results.

    Builds a row-accumulator covering: run info (date, Evidently version, reference
    dataset size, batch size label); a batch summary table (one row per batch: mode,
    date range, day count, row count, drift status/counts, target/prediction drift
    status, performance availability); data quality by batch (row counts, missing
    values/rates, schema validation, empty/constant columns); drift summary by batch
    (dataset/target/prediction drift scores and statuses with their statistical
    method); feature drift details by batch (per-feature score/threshold/detected/
    reason for every monitored feature); and classification performance by batch
    (reference-vs-current for accuracy/precision/recall/f1/roc_auc/pr_auc/log_loss).
    Written with utf-8-sig encoding so Excel opens it with correct character display.

    Args:
        csv_path: Destination path; parent directories are created if needed.
        reference_df: The (un-normalized) reference/development dataset, used for its
            row count, missing-value counts, and per-feature reference statistics.
        batch_results: List of per-batch result dicts as accumulated by run_monitoring,
            each containing batch_id/start/end/days/current_df/summary/snapshot (and
            optionally boundary_status for rolling-window batches).
        feature_columns: List of model feature column names.
        target_column: Name of the target column.
        prediction_column: Name of the predicted-label column, or None.
        probability_column: Name of the predicted-probability column, or None.
        evidently_version: The installed evidently package version string, for the
            run-info section.
        batch_size_label: Human-readable description of the batch size (e.g.
            "14 days", "1 month", "rolling 6 weeks (42 days, step 14 days)").
        batch_mode: "fixed_2_week" or "rolling_6_week", used to label the batch type
            in the summary table.

    Returns:
        The csv_path that was written to.
    """
    rows: list[list[Any]] = []

    def blank() -> None:
        rows.append([])

    rows.extend([
        ["MONITORING RUN INFORMATION"],
        ["Metric", "Value"],
        ["Monitoring Run Date", datetime.now(timezone.utc).isoformat()],
        ["Evidently Version", evidently_version],
        ["Reference Dataset", "Development prediction dataset"],
        ["Reference Row Count", len(reference_df)],
        ["Batch Size", batch_size_label],
    ])
    blank()

    rows.append(["BATCH SUMMARY"])
    rows.append([
        "Batch Type", "Batch Mode", "Batch ID", "Batch Start Date", "Batch End Date", "Number of Days",
        "Window Length", "Step Size", "Batch Boundary Status", "Current Row Count", "Dataset Drift Status", "Drifted Feature Count",
        "Drifted Feature Share", "Target Drift Status", "Prediction Drift Status",
        "Performance Status",
    ])
    for result in batch_results:
        summary = result["summary"]
        rows.append([
            "ROLLING" if batch_mode == "rolling_6_week" else "FIXED",
            "Rolling 6-Week Window" if batch_mode == "rolling_6_week" else "Fixed 2-Week Batches",
            result["batch_id"], result["start"], result["end"], result["days"],
            42 if batch_mode == "rolling_6_week" else 14,
            14,
            result.get("boundary_status", "FULL_WINDOW"),
            summary["total_current_records"], summary["overall_data_drift_status"],
            summary.get("number_of_drifted_features", "N/A"), _fmt(summary.get("drift_share")),
            summary.get("target_drift_status", "N/A"), summary.get("prediction_drift_status", "N/A"),
            "Available" if summary.get("classification_performance_metrics") else "Not Available",
        ])
    blank()

    rows.extend([
        ["DATA QUALITY BY BATCH"],
        ["Batch ID", "Metric", "Reference", "Current", "Status"],
    ])
    for result in batch_results:
        batch_id = result["batch_id"]
        summary = result["summary"]
        current_df = result["current_df"]
        ref_missing = int(reference_df.isna().sum().sum())
        cur_missing = int(current_df.isna().sum().sum())
        ref_cells = max(1, reference_df.shape[0] * reference_df.shape[1])
        cur_cells = max(1, current_df.shape[0] * current_df.shape[1])
        metric_rows = [
            ("Row count", len(reference_df), len(current_df), "Healthy"),
            ("Missing values", ref_missing, cur_missing, "Healthy" if cur_missing == 0 else "Issue"),
            ("Missing value rate", _fmt(ref_missing / ref_cells), _fmt(cur_missing / cur_cells), "Healthy" if cur_missing == 0 else "Issue"),
            ("Schema validation", "Valid", "Valid" if set(reference_df.columns) == set(current_df.columns) else "Invalid", "Healthy" if set(reference_df.columns) == set(current_df.columns) else "Issue"),
            ("Empty columns", "N/A", _fmt(_batch_metric_value(result["snapshot"], "EmptyColumnsCount()")), "Healthy"),
            ("Constant columns", "N/A", _fmt(_batch_metric_value(result["snapshot"], "ConstantColumnsCount()")), "Healthy"),
        ]
        for metric, reference, current, status in metric_rows:
            rows.append([batch_id, metric, reference, current, status])
    blank()

    rows.extend([
        ["DRIFT SUMMARY BY BATCH"],
        ["Batch ID", "Metric", "Value", "Status", "Statistical Method"],
    ])
    for result in batch_results:
        batch_id = result["batch_id"]
        summary = result["summary"]
        target_drift_detail = summary.get("target_drift_detail") or {}
        pred_label_detail = next((p for p in summary.get("prediction_drift_details", []) if p.get("feature_name") == prediction_column), {})
        pred_prob_detail = next((p for p in summary.get("prediction_drift_details", []) if p.get("feature_name") == probability_column), {})
        drift_rows = [
            ("Overall Dataset Drift Status", summary.get("overall_data_drift_status"), summary.get("overall_data_drift_status"), "Evidently DriftedColumnsCount() share vs. threshold"),
            ("Drifted Feature Count", summary.get("number_of_drifted_features"), "", "Evidently DriftedColumnsCount()"),
            ("Drifted Feature Share", _fmt(summary.get("drift_share")), "", "Evidently DriftedColumnsCount() share"),
            ("Threshold", _fmt(summary.get("drift_share_threshold")), "", "Configured drift_share threshold"),
            ("Target Drift Score", _fmt(target_drift_detail.get("drift_score")), "", target_drift_detail.get("method", "N/A")),
            ("Target Drift Status", summary.get("target_drift_status"), "", target_drift_detail.get("method", "N/A")),
            ("Prediction Label Drift", pred_label_detail.get("drift_detected", "N/A"), "", pred_label_detail.get("method", "N/A")),
            ("Prediction Probability Drift", pred_prob_detail.get("drift_detected", "N/A"), "", pred_prob_detail.get("method", "N/A")),
        ]
        for metric, value, status, method in drift_rows:
            rows.append([batch_id, metric, value, status, method])
    blank()

    rows.extend([
        ["FEATURE DRIFT DETAILS"],
        ["Batch ID", "Feature Name", "Feature Type", "Statistical Method", "Drift Score", "Threshold", "Drift Detected", "Reason"],
    ])
    for result in batch_results:
        summary = result["summary"]
        current_df = result["current_df"]
        details = {item.get("feature_name"): item for item in summary.get("feature_drift_details", [])}
        for feature in feature_columns:
            detail = details.get(feature, {})
            feature_type = "numerical" if pd.api.types.is_numeric_dtype(reference_df[feature]) else "categorical"
            unavailable_reason = _drift_unavailable_reason(reference_df, current_df, feature)
            drift_detected = detail.get("drift_detected")
            rows.append([
                result["batch_id"], feature, feature_type,
                detail.get("method") or unavailable_reason,
                _fmt(detail.get("drift_score")), _fmt(detail.get("threshold")),
                drift_detected if drift_detected is not None else "N/A",
                unavailable_reason if drift_detected is None else "",
            ])
    blank()

    rows.extend([
        ["CLASSIFICATION PERFORMANCE BY BATCH"],
        ["Batch ID", "Metric", "Reference", "Current", "Change", "Status", "Statistical Method"],
    ])
    performance_metrics = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("log_loss", "LogLoss")]
    reference_metrics = _compute_reference_classification_metrics(reference_df, target_column, prediction_column, probability_column)
    for result in batch_results:
        current_metrics = result["summary"].get("classification_performance_metrics", {})
        available = len(result["current_df"][target_column].dropna().unique()) == 2
        for key, label in performance_metrics:
            if not available:
                rows.append([result["batch_id"], label, "Not Available", "Not Available", "", "Current batch contains only one target class.", CLASSIFICATION_METRIC_METHODS.get(key, "N/A")])
                continue
            reference = reference_metrics.get(key)
            current = current_metrics.get(key)
            change = current - reference if current is not None and reference is not None else None
            rows.append([result["batch_id"], label, _fmt(reference), _fmt(current), _fmt(change), "Available" if current is not None else "Not Available", CLASSIFICATION_METRIC_METHODS.get(key, "N/A")])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(rows)
    return csv_path


def _write_batch_excel(
    excel_path: Path,
    reference_df: pd.DataFrame,
    batch_results: list[dict[str, Any]],
    feature_columns: list[str],
    target_column: str,
    prediction_column: str | None,
    probability_column: str | None,
    evidently_version: str,
    reference_path: Path,
    current_path: Path,
    negative_class_label: str = "N",
    positive_class_label: str = "Y",
) -> Path:
    """Write a legacy Excel workbook with one monitoring topic per sheet.

    Superseded by _write_analysis_excel (the primary Excel writer used by
    run_monitoring), but kept for direct/manual use. Produces sheets: Run Information,
    Drift Methodology, Data Quality, Drift Summary, Feature Drift, Performance, Target
    Distribution, and Prediction Distribution (the last two with column-definition
    footers explaining what class 0/1 mean). Requires the openpyxl package.

    Args:
        excel_path: Destination .xlsx path; parent directories are created if needed.
        reference_df: The (un-normalized) reference/development dataset.
        batch_results: List of per-batch result dicts (see _write_batch_csv).
        feature_columns: List of model feature column names.
        target_column: Name of the target column.
        prediction_column: Name of the predicted-label column, or None.
        probability_column: Name of the predicted-probability column, or None.
        evidently_version: The installed evidently package version string.
        reference_path: Path to the reference prediction CSV (for display only).
        current_path: Path to the current prediction CSV (for display only).
        negative_class_label: Display label for target/prediction class 0.
        positive_class_label: Display label for target/prediction class 1.

    Returns:
        The excel_path that was written to.

    Raises:
        RuntimeError: If the openpyxl package is not installed.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Excel output requires openpyxl. Install it with: py -m pip install openpyxl") from exc

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        run_rows = [
            ["Metric", "Reference / Development", "Current / Production"],
            ["Dataset", str(reference_path), str(current_path)],
            ["Row Count", len(reference_df), sum(len(item["current_df"]) for item in batch_results)],
            ["Number of Columns", len(reference_df.columns), len(reference_df.columns)],
            ["Number of Monitored Features", len(feature_columns), len(feature_columns)],
            ["Number of Numerical Columns", sum(pd.api.types.is_numeric_dtype(reference_df[col]) for col in feature_columns), sum(pd.api.types.is_numeric_dtype(reference_df[col]) for col in feature_columns)],
            ["Number of Categorical Columns", sum(not pd.api.types.is_numeric_dtype(reference_df[col]) for col in feature_columns), sum(not pd.api.types.is_numeric_dtype(reference_df[col]) for col in feature_columns)],
            ["Number of Text Columns", 0, 0],
            ["Number of Datetime Columns", 0, 0],
            ["Missing Values", int(reference_df.isna().sum().sum()), sum(int(item["current_df"].isna().sum().sum()) for item in batch_results)],
            [],
            ["Metric", "Value"],
            ["Monitoring Run Date", datetime.now(timezone.utc).isoformat()],
            ["Evidently Version", evidently_version],
            [],
        ]
        pd.DataFrame(run_rows).to_excel(writer, sheet_name="Run Information", index=False, header=False)

        method_rows = [["Batch ID", "Method Type", "Statistical Method", "Applied To", "Threshold", "Feature Count", "Description"]]
        for item in batch_results:
            methods: dict[str, dict[str, Any]] = {}
            summary = item["summary"]
            details = summary.get("feature_drift_details", []) + ([summary["target_drift_detail"]] if summary.get("target_drift_detail") else []) + summary.get("prediction_drift_details", [])
            for detail in details:
                method = detail.get("method") or "Unknown"
                methods.setdefault(method, {"names": [], "threshold": detail.get("threshold")})["names"].append(detail.get("feature_name"))
            for method, data in methods.items():
                applied_to = "Features"
                method_rows.append([item["batch_id"], "Data Drift", method, applied_to, data["threshold"], len(data["names"]), "Evidently-selected statistical method"])
        pd.DataFrame(method_rows[1:], columns=method_rows[0]).to_excel(writer, sheet_name="Drift Methodology", index=False)

        quality_rows = []
        for item in batch_results:
            current = item["current_df"]
            ref_missing = int(reference_df.isna().sum().sum())
            cur_missing = int(current.isna().sum().sum())
            quality_rows.extend([
                [item["batch_id"], "Row count", len(reference_df), len(current), "Healthy"],
                [item["batch_id"], "Missing values", ref_missing, cur_missing, "Healthy" if cur_missing == 0 else "Issue"],
                [item["batch_id"], "Missing value rate", ref_missing / max(1, reference_df.size), cur_missing / max(1, current.size), "Healthy" if cur_missing == 0 else "Issue"],
                [item["batch_id"], "Schema validation", "Valid", "Valid" if list(reference_df.columns) == list(current.columns) else "Invalid", "Healthy" if list(reference_df.columns) == list(current.columns) else "Issue"],
            ])
        pd.DataFrame(quality_rows, columns=["Batch ID", "Metric", "Reference", "Current", "Status"]).to_excel(writer, sheet_name="Data Quality", index=False)

        drift_rows = []
        for item in batch_results:
            summary = item["summary"]
            target_drift_detail = summary.get("target_drift_detail") or {}
            pred_label_detail = next((x for x in summary.get("prediction_drift_details", []) if x.get("feature_name") == prediction_column), {})
            pred_prob_detail = next((x for x in summary.get("prediction_drift_details", []) if x.get("feature_name") == probability_column), {})
            drift_rows.extend([
                [item["batch_id"], "Overall Dataset Drift Status", summary.get("overall_data_drift_status"), summary.get("overall_data_drift_status"), "Evidently DriftedColumnsCount() share vs. threshold"],
                [item["batch_id"], "Drifted Feature Count", summary.get("number_of_drifted_features"), "", "Evidently DriftedColumnsCount()"],
                [item["batch_id"], "Drifted Feature Share", summary.get("drift_share"), "", "Evidently DriftedColumnsCount() share"],
                [item["batch_id"], "Threshold", summary.get("drift_share_threshold"), "", "Configured drift_share threshold"],
                [item["batch_id"], "Target Drift Score", target_drift_detail.get("drift_score"), "", target_drift_detail.get("method", "N/A")],
                [item["batch_id"], "Target Drift Status", summary.get("target_drift_status"), "", target_drift_detail.get("method", "N/A")],
                [item["batch_id"], "Prediction Label Drift", pred_label_detail.get("drift_detected", "N/A"), "", pred_label_detail.get("method", "N/A")],
                [item["batch_id"], "Prediction Probability Drift", pred_prob_detail.get("drift_detected", "N/A"), "", pred_prob_detail.get("method", "N/A")],
            ])
        pd.DataFrame(drift_rows, columns=["Batch ID", "Metric", "Value", "Status", "Statistical Method"]).to_excel(writer, sheet_name="Drift Summary", index=False)

        feature_rows = []
        for item in batch_results:
            details = {x.get("feature_name"): x for x in item["summary"].get("feature_drift_details", [])}
            for feature in feature_columns:
                detail = details.get(feature, {})
                current_df = item["current_df"]
                unavailable_reason = _drift_unavailable_reason(reference_df, current_df, feature)
                drift_detected = detail.get("drift_detected")
                feature_rows.append([
                    item["batch_id"], feature,
                    "numerical" if pd.api.types.is_numeric_dtype(reference_df[feature]) else "categorical",
                    detail.get("method") or unavailable_reason,
                    detail.get("drift_score"), detail.get("threshold"),
                    drift_detected if drift_detected is not None else "N/A",
                    unavailable_reason if drift_detected is None else "",
                ])
        pd.DataFrame(feature_rows, columns=["Batch ID", "Feature Name", "Feature Type", "Statistical Method", "Drift Score", "Threshold", "Drift Detected", "Reason"]).to_excel(writer, sheet_name="Feature Drift", index=False)

        performance_rows = []
        ref_metrics = _compute_reference_classification_metrics(reference_df, target_column, prediction_column, probability_column)
        labels = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("log_loss", "LogLoss"), *LIFT_METRIC_LABELS]
        for item in batch_results:
            current_metrics = item["summary"].get("classification_performance_metrics", {})
            available = item["current_df"][target_column].nunique(dropna=True) == 2
            for key, label in labels:
                if not available:
                    performance_rows.append([item["batch_id"], label, "Not Available", "Not Available", "", "Current batch contains only one target class.", CLASSIFICATION_METRIC_METHODS.get(key, "N/A")])
                else:
                    reference = ref_metrics.get(key)
                    current = current_metrics.get(key)
                    performance_rows.append([item["batch_id"], label, reference, current, current - reference if reference is not None and current is not None else "", "Available" if current is not None else "Not Available", CLASSIFICATION_METRIC_METHODS.get(key, "N/A")])
        pd.DataFrame(performance_rows, columns=["Batch ID", "Metric", "Reference", "Current", "Change", "Status", "Statistical Method"]).to_excel(writer, sheet_name="Performance", index=False)

        target_rows = []
        prediction_rows = []
        for item in batch_results:
            current = item["current_df"]
            for value in sorted(set(reference_df[target_column].unique()) | set(current[target_column].unique()), key=str):
                target_rows.append([item["batch_id"], target_column, value, int((reference_df[target_column] == value).sum()), int((current[target_column] == value).sum())])
            if prediction_column:
                for value in sorted(set(reference_df[prediction_column].unique()) | set(current[prediction_column].unique()), key=str):
                    prediction_rows.append([item["batch_id"], prediction_column, value, int((reference_df[prediction_column] == value).sum()), int((current[prediction_column] == value).sum())])
        pd.DataFrame(target_rows, columns=["Batch ID", "Column", "Class", "Reference Count", "Current Count"]).to_excel(writer, sheet_name="Target Distribution", index=False)
        pd.DataFrame(prediction_rows, columns=["Batch ID", "Column", "Class", "Reference Count", "Current Count"]).to_excel(writer, sheet_name="Prediction Distribution", index=False)

        target_definition_rows = pd.DataFrame([
            ["COLUMN DEFINITIONS"],
            ["Class 0", f"Negative class: {negative_class_label}"],
            ["Class 1", f"Positive class: {positive_class_label}"],
            ["Reference Count", "Number of records in the development/reference dataset with this class."],
            ["Current Count", "Number of records in the current batch with this class."],
        ])
        prediction_definition_rows = pd.DataFrame([
            ["COLUMN DEFINITIONS"],
            ["Class 0", f"Negative prediction: {negative_class_label}"],
            ["Class 1", f"Positive prediction: {positive_class_label}"],
            ["Reference Count", "Number of reference records with this predicted class."],
            ["Current Count", "Number of current-batch records with this predicted class."],
        ])
        target_definition_rows.to_excel(
            writer, sheet_name="Target Distribution", index=False, header=False,
            startrow=len(target_rows) + 3,
        )
        prediction_definition_rows.to_excel(
            writer, sheet_name="Prediction Distribution", index=False, header=False,
            startrow=len(prediction_rows) + 3,
        )

    return excel_path


def _write_batch_html(
    html_path: Path,
    batch_results: list[dict[str, Any]],
    reference_path: Path,
    current_path: Path,
    reference_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    prediction_column: str | None,
    probability_column: str | None,
) -> Path:
    """Write a hand-built (non-Evidently-templated) HTML report summarizing all batches.

    Produces a minimal, dependency-free HTML page: a dataset column configuration
    table, then one section per batch with a small metrics table (row count, dataset
    drift status, drifted feature count, target/prediction drift status and method,
    classification performance availability) plus a collapsible <details> block
    containing the full raw Evidently snapshot JSON for that batch (for deep-dive
    debugging). This is the aggregate HTML report written by run_monitoring, distinct
    from the per-run HTML that Evidently itself renders in _run_single_monitoring.

    Args:
        html_path: Destination path; parent directories are created if needed.
        batch_results: List of per-batch result dicts (see _write_batch_csv).
        reference_path: Path to the reference prediction CSV (for display only).
        current_path: Path to the current prediction CSV (for display only).
        reference_df: The (normalized) reference/development dataset.
        feature_columns: List of model feature column names.
        target_column: Name of the target column.
        prediction_column: Name of the predicted-label column, or None.
        probability_column: Name of the predicted-probability column, or None.

    Returns:
        The html_path that was written to.
    """
    html_path.parent.mkdir(parents=True, exist_ok=True)
    column_config_rows = [
        ("ID", "None", "N/A"),
        ("Target", target_column, str(reference_df[target_column].dtype)),
        (
            "Prediction Label",
            prediction_column or "None",
            str(reference_df[prediction_column].dtype) if prediction_column else "N/A",
        ),
        (
            "Prediction Probability",
            probability_column or "None",
            str(reference_df[probability_column].dtype) if probability_column else "N/A",
        ),
        ("Date", "None", "N/A"),
        ("Number of Feature Columns", str(len(feature_columns)), ""),
    ]
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Evidently Batch Monitoring Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:2rem} table{border-collapse:collapse;margin-bottom:2rem} th,td{border:1px solid #ccc;padding:.4rem;text-align:left} pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem}</style>",
        "</head><body><h1>Evidently Batch Monitoring Report</h1>",
        f"<p>Reference: {reference_path}<br>Current: {current_path}</p>",
        "<h2>Dataset Column Configuration</h2>",
        "<table><tr><th>Column Type</th><th>Column Name</th><th>Data Type</th></tr>",
        *[
            "<tr>"
            f"<td>{html.escape(column_type)}</td>"
            f"<td>{html.escape(column_name)}</td>"
            f"<td>{html.escape(data_type)}</td>"
            "</tr>"
            for column_type, column_name, data_type in column_config_rows
        ],
        "</table>",
        "<h2>Batch Summary</h2>",
    ]
    for result in batch_results:
        summary = result["summary"]
        sections.extend([
            f"<h2>{result['batch_id']}: {result['start']} to {result['end']}</h2>",
            "<table><tr><th>Metric</th><th>Value</th><th>Statistical Method</th></tr>",
            f"<tr><td>Rows</td><td>{len(result['current_df'])}</td><td></td></tr>",
            f"<tr><td>Dataset Drift</td><td>{summary.get('overall_data_drift_status')}</td><td>Evidently DriftedColumnsCount() share vs. threshold</td></tr>",
            f"<tr><td>Drifted Features</td><td>{summary.get('number_of_drifted_features', 'N/A')}</td><td></td></tr>",
            f"<tr><td>Target Drift</td><td>{summary.get('target_drift_status', 'N/A')}</td><td>{(summary.get('target_drift_detail') or {}).get('method', 'N/A')}</td></tr>",
            f"<tr><td>Prediction Drift</td><td>{summary.get('prediction_drift_status', 'N/A')}</td><td>{', '.join(sorted({p.get('method', 'N/A') for p in summary.get('prediction_drift_details', [])})) or 'N/A'}</td></tr>",
            f"<tr><td>Classification Performance</td><td>{'Available' if summary.get('classification_performance_metrics') else 'Not Available'}</td><td>{', '.join(CLASSIFICATION_METRIC_METHODS.values())}</td></tr>",
            "</table>",
            "<details><summary>Full Evidently snapshot JSON</summary>",
            f"<pre>{html.escape(json.dumps(result['snapshot'], indent=2, default=_json_default))}</pre></details>",
        ])
    sections.append("</body></html>")
    html_path.write_text("".join(sections), encoding="utf-8")
    return html_path


def _write_analysis_excel(
    excel_path: Path,
    reference_df: pd.DataFrame,
    batch_results: list[dict[str, Any]],
    feature_columns: list[str],
    target_column: str,
    prediction_column: str | None,
    probability_column: str | None,
    evidently_version: str,
    reference_path: Path,
    current_path: Path,
    runtime_cfg: dict[str, Any],
    batch_size_label: str,
    negative_class_label: str = "N",
    positive_class_label: str = "Y",
) -> Path:
    """Write batch results in wide, row-oriented sheets for Excel analysis.

    This is the primary Excel writer used by run_monitoring (as opposed to the legacy
    _write_batch_excel). Each sheet has one row per batch (rather than one row per
    metric per batch), which makes it easy to filter/sort/pivot in Excel. Produces
    sheets: Run Information (setup + selected split/period inputs + per-batch
    one-line summary), Data Quality, Drift Summary, Feature Drift (per-feature drift
    score, one column per feature), Feature Drift Status (Drift/No Drift/N/A per
    feature), Feature Metadata (type + statistical method per monitored column),
    Performance (reference/current/change for every classification metric + lift),
    Target Distribution, and Prediction Distribution. Every sheet is prefixed with an
    "EXPLANATION" block (via the nested explanation_rows helper) describing what the
    sheet contains and what it tells the reader. Requires the openpyxl package.

    Args:
        excel_path: Destination .xlsx path; parent directories are created if needed.
        reference_df: The (normalized) reference/development dataset.
        batch_results: List of per-batch result dicts (see _write_batch_csv).
        feature_columns: List of model feature column names.
        target_column: Name of the target column.
        prediction_column: Name of the predicted-label column, or None.
        probability_column: Name of the predicted-probability column, or None.
        evidently_version: The installed evidently package version string.
        reference_path: Path to the reference prediction CSV (for display only).
        current_path: Path to the current prediction CSV (for display only).
        runtime_cfg: The monitoring_config.json runtime config dict, used to display
            the selected split/period inputs (development/current start/end dates,
            batch date column, monitoring mode, batch splitting option).
        batch_size_label: Human-readable description of the batch size.
        negative_class_label: Display label for target/prediction class 0.
        positive_class_label: Display label for target/prediction class 1.

    Returns:
        The excel_path that was written to.

    Raises:
        RuntimeError: If the openpyxl package is not installed.
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError("Excel output requires openpyxl. Install it with: py -m pip install openpyxl") from exc

    def write_sheet(
        writer: pd.ExcelWriter,
        name: str,
        frame: pd.DataFrame,
        top_rows: list[list[Any]] | None = None,
        spacer_rows: int = 2,
        freeze_header: bool = True,
        rotate_header: bool = False,
    ) -> None:
        start_row = 0
        if top_rows:
            pd.DataFrame(top_rows).to_excel(writer, sheet_name=name, index=False, header=False)
            start_row = len(top_rows) + spacer_rows
        frame.to_excel(writer, sheet_name=name, index=False, startrow=start_row)
        worksheet = writer.book[name]
        for row in worksheet.iter_rows():
            for cell in row:
                if cell.value is not None:
                    cell.alignment = openpyxl.styles.Alignment(vertical="top", wrap_text=True)
        header_row = start_row + 1
        for cell in worksheet[header_row]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            if rotate_header:
                cell.alignment = openpyxl.styles.Alignment(
                    text_rotation=90,
                    vertical="bottom",
                    horizontal="center",
                    wrap_text=True,
                )
        if rotate_header:
            worksheet.row_dimensions[header_row].height = 170
        if freeze_header:
            worksheet.freeze_panes = f"A{header_row + 1}"
        for column_cells in worksheet.columns:
            width = 14 if rotate_header else min(max(len(str(cell.value or "")) for cell in column_cells) + 2, 36)
            worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = width

    def _title_from_code(value: Any) -> str:
        text = str(value or "N/A").strip()
        if not text or text == "N/A":
            return "N/A"
        return text.replace("_", " ").replace("-", " ").title()

    def _split_option_label(value: Any) -> str:
        text = str(value or "N/A").strip()
        return "N/A" if text == "N/A" else f"Option {text}"

    def _batch_mode_label(value: Any) -> str:
        labels = {
            "fixed_2_week": "Fixed 2-Week Batches",
            "rolling_6_week": "Rolling 6-Week Window",
        }
        text = str(value or "N/A").strip().lower()
        return labels.get(text, _title_from_code(value))

    def explanation_rows(sheet_name: str) -> list[list[str]]:
        explanations = {
            "Run Information": [
                "Shows the monitoring setup, selected split inputs, dataset columns, and one-row summary for each batch.",
                "Use this first to confirm which development/current periods and batch splitting option were used before reading the metric sheets.",
            ],
            "Data Quality": [
                "Shows missing values, missing rates, schema status, empty columns, and constant columns for each current batch compared with the development reference data.",
                "This tells whether the current data is structurally healthy enough to trust drift and performance results.",
            ],
            "Drift Summary": [
                "Shows high-level dataset, target, and prediction drift results for every batch.",
                "Drift means the current batch distribution is statistically different from the development/reference distribution based on the listed Evidently method and threshold.",
            ],
            "Feature Drift": [
                "Shows the drift score for every monitored feature across every batch.",
                "Use this to identify which input features changed most between development data and current data.",
            ],
            "Feature Drift Status": [
                "Shows each feature as Drift, No Drift, or N/A (not evaluated because Evidently skipped the column, data was missing or constant, or no valid comparison was available) for every batch.",
                "This is a quick status view of the same feature-level drift checks without the numeric scores.",
            ],
            "Feature Metadata": [
                "Lists every monitored feature, target, prediction column, and performance metric with its type and drift method when applicable.",
                "This tells what each column represents and which statistical method Evidently used to compare distributions.",
            ],
            "Performance": [
                "Shows classification metrics such as accuracy, precision, recall, F1, ROC AUC, PR AUC, log loss, and lift for each batch.",
                "Performance tells how well predictions match the target; higher is better for most metrics, while lower log loss is better.",
            ],
            "Target Distribution": [
                "Shows counts for each target class in the development/reference data and each current batch.",
                f"The target is the actual outcome column ({target_column}); distribution means how records are spread across target classes such as {negative_class_label} and {positive_class_label}.",
            ],
            "Prediction Distribution": [
                "Shows counts for each predicted class in the development/reference data and each current batch.",
                "Prediction distribution tells whether the model is predicting classes in a similar pattern over time or if prediction behavior has shifted.",
            ],
        }
        contents, meaning = explanations[sheet_name]
        return [
            ["EXPLANATION"],
            ["What this sheet contains", contents],
            ["What this tells you", meaning],
        ]

    batch_base = [
        {"Batch ID": item["batch_id"], "Batch Start": item["start"], "Batch End": item["end"], "Days": item["days"], "Rows": len(item["current_df"])}
        for item in batch_results
    ]

    column_config_rows = [
        ["DATASET COLUMN CONFIGURATION"],
        ["Column Type", "Column Name", "Data Type"],
        ["ID", "None", "N/A"],
        ["Target", target_column, str(reference_df[target_column].dtype)],
        [
            "Prediction Label",
            prediction_column or "None",
            str(reference_df[prediction_column].dtype) if prediction_column else "N/A",
        ],
        [
            "Prediction Probability",
            probability_column or "None",
            str(reference_df[probability_column].dtype) if probability_column else "N/A",
        ],
        ["Date", "None", "N/A"],
        ["Number of Feature Columns", len(feature_columns), ""],
    ]

    run_top = [
        *explanation_rows("Run Information"),
        [],
        ["MONITORING RUN INFORMATION"],
        ["Metric", "Reference / Development", "Current / Production"],
        ["Dataset", str(reference_path), str(current_path)],
        ["Reference Row Count", len(reference_df), sum(len(item["current_df"]) for item in batch_results)],
        ["Number of Columns", len(reference_df.columns), len(reference_df.columns)],
        ["Number of Monitored Features", len(feature_columns), len(feature_columns)],
        ["Evidently Version", evidently_version, evidently_version],
        ["Monitoring Run Date", datetime.now(timezone.utc).isoformat(), ""],
        [],
        ["SELECTED SPLIT AND PERIOD INPUTS"],
        ["Metric", "Value", ""],
        ["Development Period Start", runtime_cfg.get("baseline_start_date", "N/A"), ""],
        ["Development Period End", runtime_cfg.get("baseline_end_date", "N/A"), ""],
        ["Current Period Start", runtime_cfg.get("current_start_date", "N/A"), ""],
        ["Current Period End", runtime_cfg.get("current_end_date", "N/A"), ""],
        ["Batch Date Column", runtime_cfg.get("batch_date_column", "N/A"), ""],
        ["Monitoring Mode", _title_from_code(runtime_cfg.get("monitoring_mode", "fortnight")), ""],
        ["Batch Splitting Option", _batch_mode_label(runtime_cfg.get("batch_mode", "fixed_2_week")), ""],
        ["Batch Size", batch_size_label, ""],
        [],
        *column_config_rows,
        [],
        ["BATCH SUMMARY"],
    ]
    summary_rows = []
    for item, base in zip(batch_results, batch_base):
        summary = item["summary"]
        summary_rows.append({
            **base,
            "Dataset Drift": summary.get("overall_data_drift_status"),
            "Drifted Feature Count": summary.get("number_of_drifted_features"),
            "Drifted Feature Share": summary.get("drift_share"),
            "Target Drift": summary.get("target_drift_status"),
            "Prediction Drift": summary.get("prediction_drift_status"),
            "Performance": "Available" if summary.get("classification_performance_metrics") else "Not Available",
            "Performance Reason": "" if summary.get("classification_performance_metrics") else "Current batch contains only one target class.",
        })

    quality_rows = []
    drift_rows = []
    performance_rows = []
    target_rows = []
    prediction_rows = []
    reference_metrics = _compute_reference_classification_metrics(reference_df, target_column, prediction_column, probability_column)
    metric_labels = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("log_loss", "LogLoss"), *LIFT_METRIC_LABELS]

    for item, base in zip(batch_results, batch_base):
        current = item["current_df"]
        summary = item["summary"]
        ref_missing = int(reference_df.isna().sum().sum())
        cur_missing = int(current.isna().sum().sum())
        quality_rows.append({
            **base,
            "Reference Missing Values": ref_missing,
            "Current Missing Values": cur_missing,
            "Reference Missing Rate": ref_missing / max(1, reference_df.size),
            "Current Missing Rate": cur_missing / max(1, current.size),
            "Schema Status": "Valid" if list(reference_df.columns) == list(current.columns) else "Invalid",
            "Empty Columns": _batch_metric_value(item["snapshot"], "EmptyColumnsCount()"),
            "Constant Columns": _batch_metric_value(item["snapshot"], "ConstantColumnsCount()"),
        })
        target_drift_detail = summary.get("target_drift_detail") or {}
        pred_label_detail = next((x for x in summary.get("prediction_drift_details", []) if x.get("feature_name") == prediction_column), {})
        pred_prob_detail = next((x for x in summary.get("prediction_drift_details", []) if x.get("feature_name") == probability_column), {})
        drift_row = {
            **base,
            "Threshold": summary.get("drift_share_threshold"),
            "Target Drift Score": target_drift_detail.get("drift_score"),
            "Target Drift Method": target_drift_detail.get("method", "N/A"),
        }
        for label, value in [
            ("Dataset Drift Status", summary.get("overall_data_drift_status")),
            ("Drifted Feature Count", summary.get("number_of_drifted_features")),
            ("Drifted Feature Share", summary.get("drift_share")),
            ("Target Drift Status", summary.get("target_drift_status")),
            ("Prediction Label Drift", next((x.get("drift_detected") for x in summary.get("prediction_drift_details", []) if x.get("feature_name") == prediction_column), "N/A")),
            ("Prediction Probability Drift", next((x.get("drift_detected") for x in summary.get("prediction_drift_details", []) if x.get("feature_name") == probability_column), "N/A")),
        ]:
            drift_row[label] = value
        drift_row["Prediction Label Drift Method"] = pred_label_detail.get("method", "N/A")
        drift_row["Prediction Probability Drift Method"] = pred_prob_detail.get("method", "N/A")
        drift_rows.append(drift_row)

        available = current[target_column].nunique(dropna=True) == 2
        performance_row = {**base}
        current_metrics = summary.get("classification_performance_metrics", {})
        for key, label in metric_labels:
            reference = reference_metrics.get(key)
            value = current_metrics.get(key) if available else "Not Available"
            performance_row[f"{label} Reference"] = reference if available else "Not Available"
            performance_row[f"{label} Current"] = value
            performance_row[f"{label} Change"] = value - reference if available and reference is not None and isinstance(value, (int, float)) else ""
        performance_row["Status"] = "Available" if available else "Not Available: Current batch contains only one target class."
        performance_rows.append(performance_row)

        target_row = {
            **base,
            "Target 0 Reference Count": int((reference_df[target_column] == 0).sum()),
            "Target 0 Current Count": int((current[target_column] == 0).sum()),
            "Target 1 Reference Count": int((reference_df[target_column] == 1).sum()),
            "Target 1 Current Count": int((current[target_column] == 1).sum()),
            "Target Drift Status": "Drift" if summary.get("target_drift_status") is True else (
                "No Drift" if summary.get("target_drift_status") is False else "N/A"
            ),
        }
        target_rows.append(target_row)
        if prediction_column:
            prediction_drift_status = summary.get("prediction_drift_status")
            prediction_rows.append({
                **base,
                "Prediction 0 Reference Count": int((reference_df[prediction_column] == 0).sum()),
                "Prediction 0 Current Count": int((current[prediction_column] == 0).sum()),
                "Prediction 1 Reference Count": int((reference_df[prediction_column] == 1).sum()),
                "Prediction 1 Current Count": int((current[prediction_column] == 1).sum()),
                "Prediction Drift Status": "Drift" if prediction_drift_status is True else (
                    "No Drift" if prediction_drift_status is False else "N/A"
                ),
            })

    feature_metadata = []
    for feature in feature_columns:
        first_detail = next((x for item in batch_results for x in item["summary"].get("feature_drift_details", []) if x.get("feature_name") == feature), {})
        feature_metadata.append({
            "Feature Name": feature,
            "Feature Type": "numerical" if pd.api.types.is_numeric_dtype(reference_df[feature]) else "categorical",
            "Statistical Method": first_detail.get("method") or _drift_unavailable_reason(
                reference_df, batch_results[0]["current_df"] if batch_results else reference_df, feature
            ),
            "Threshold": first_detail.get("threshold", "N/A"),
        })
    first_target_detail = next((item["summary"].get("target_drift_detail") for item in batch_results if item["summary"].get("target_drift_detail")), {}) or {}
    feature_metadata.append({
        "Feature Name": target_column,
        "Feature Type": "target",
        "Statistical Method": first_target_detail.get("method", "N/A"),
        "Threshold": first_target_detail.get("threshold", "N/A"),
    })
    if prediction_column:
        first_pred_detail = next((x for item in batch_results for x in item["summary"].get("prediction_drift_details", []) if x.get("feature_name") == prediction_column), {})
        feature_metadata.append({
            "Feature Name": prediction_column,
            "Feature Type": "prediction_label",
            "Statistical Method": first_pred_detail.get("method", "N/A"),
            "Threshold": first_pred_detail.get("threshold", "N/A"),
        })
    if probability_column:
        first_prob_detail = next((x for item in batch_results for x in item["summary"].get("prediction_drift_details", []) if x.get("feature_name") == probability_column), {})
        feature_metadata.append({
            "Feature Name": probability_column,
            "Feature Type": "prediction_probability",
            "Statistical Method": first_prob_detail.get("method", "N/A"),
            "Threshold": first_prob_detail.get("threshold", "N/A"),
        })
    # Statistical Method/Threshold describe drift detection only; performance metric formulas
    # are reported on the Performance sheet instead.
    for label in ["Accuracy", "Precision", "Recall", "F1", "ROC AUC", "PR AUC", "LogLoss", "Lift@5%", "Lift@10%", "Lift@25%"]:
        feature_metadata.append({
            "Feature Name": label,
            "Feature Type": "classification_performance",
            "Statistical Method": "",
            "Threshold": "",
        })
    feature_score_rows = []
    feature_status_rows = []
    for item, base in zip(batch_results, batch_base):
        details = {x.get("feature_name"): x for x in item["summary"].get("feature_drift_details", [])}
        score_row = {**base}
        status_row = {**base}
        for feature in feature_columns:
            detail = details.get(feature, {})
            score_row[feature] = detail.get("drift_score", "N/A")
            detected = detail.get("drift_detected", "N/A")
            status_row[feature] = "Drift" if detected is True else (
                "No Drift" if detected is False else f"N/A ({_drift_unavailable_reason(reference_df, item['current_df'], feature)})"
            )
        feature_score_rows.append(score_row)
        feature_status_rows.append(status_row)

    status_rows_by_feature = []
    for feature in feature_columns:
        status_row = {"Feature Name": feature}
        for item, batch_status in zip(batch_results, feature_status_rows):
            batch_id = item["batch_id"]
            status_row[
                f"{batch_id}\n{item['days']} days | {len(item['current_df'])} rows"
            ] = batch_status[feature]
        status_rows_by_feature.append(status_row)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        write_sheet(
            writer,
            "Run Information",
            pd.DataFrame(summary_rows),
            run_top,
            spacer_rows=2,
            freeze_header=False,
        )
        write_sheet(writer, "Data Quality", pd.DataFrame(quality_rows), explanation_rows("Data Quality"), freeze_header=False)
        write_sheet(writer, "Drift Summary", pd.DataFrame(drift_rows), explanation_rows("Drift Summary"), freeze_header=False)
        write_sheet(writer, "Feature Drift", pd.DataFrame(feature_score_rows), explanation_rows("Feature Drift"), freeze_header=False)
        write_sheet(writer, "Feature Drift Status", pd.DataFrame(status_rows_by_feature), explanation_rows("Feature Drift Status"), freeze_header=False)
        write_sheet(writer, "Feature Metadata", pd.DataFrame(feature_metadata), explanation_rows("Feature Metadata"), freeze_header=False)
        write_sheet(writer, "Performance", pd.DataFrame(performance_rows), explanation_rows("Performance"), freeze_header=False)
        write_sheet(writer, "Target Distribution", pd.DataFrame(target_rows), explanation_rows("Target Distribution"), freeze_header=False)
        write_sheet(writer, "Prediction Distribution", pd.DataFrame(prediction_rows), explanation_rows("Prediction Distribution"), freeze_header=False)

        for sheet_name, definition_rows in [
            (
                "Target Distribution",
                [
                    ["COLUMN DEFINITIONS"],
                    ["Target 0", f"Negative class: {negative_class_label}"],
                    ["Target 1", f"Positive class: {positive_class_label}"],
                    ["Reference Count", "Number of records in the reference/development dataset."],
                    ["Current Count", "Number of records in the current batch."],
                    ["Target Drift Status", "Whether the target class distribution changed between reference and current data."],
                ],
            ),
            (
                "Prediction Distribution",
                [
                    ["COLUMN DEFINITIONS"],
                    ["Prediction 0", f"Negative prediction: {negative_class_label}"],
                    ["Prediction 1", f"Positive prediction: {positive_class_label}"],
                    ["Reference Count", "Number of reference records with this predicted class."],
                    ["Current Count", "Number of current-batch records with this predicted class."],
                    ["Prediction Drift Status", "Whether the predicted-class distribution changed between reference and current data."],
                ],
            ),
        ]:
            worksheet = writer.book[sheet_name]
            worksheet.append([])
            worksheet.append([])
            worksheet.append([])
            for row in definition_rows:
                worksheet.append(row)
    return excel_path


def run_monitoring(config_path: Path) -> tuple[Path, Path, Path, dict[str, Any], dict[str, str | list[str]]]:
    """Run the full monitoring pipeline: batch the current period and write all reports.

    Loads train_config.json and monitoring_config.json (the runtime config written by
    train.py), resolves which prediction CSVs to compare (preferring
    monitoring_config.json's baseline/current paths over the static config), and reads
    them. Validates/parses the configured batch date column on the current dataset,
    dropping rows with unparseable dates.

    Splits the current dataset into batches according to monitoring_mode/batch_mode
    from the runtime config:
      - "month": one batch per calendar month present in the data.
      - "year": one batch per calendar year present in the data.
      - "rolling_6_week" (two code paths depending on split_type/split_option):
        overlapping 42-day windows stepping backward by 14 days from the latest date,
        trimmed at the configured current-period start boundary.
      - default ("fortnight"): non-overlapping 14-day batches from the earliest to the
        latest date.

    For each batch, calls _run_single_monitoring(write_outputs=False) to compute that
    batch's summary/snapshot without writing files yet, printing a short progress
    block to stdout (drift status, prediction drift, ROC AUC). After all batches are
    processed, writes the aggregated flat CSV (_write_batch_csv), the row-oriented
    Excel workbook (_write_analysis_excel, with an auto-incrementing filename suffix
    if the base name already exists), an aggregate JSON payload of every batch's
    summary/snapshot, and the aggregate HTML report (_write_batch_html).

    Args:
        config_path: Path to train_config.json.

    Returns:
        A tuple of (report_html_path, report_json_path, report_excel_path,
        aggregate_summary, resolved_columns), where aggregate_summary has
        total_reference_records/total_current_records/batch_count, and
        resolved_columns has target_column/prediction_column/
        prediction_probability_column/feature_columns.

    Raises:
        KeyError: If the configured batch date column is not found in the current
            prediction dataset.
        ValueError: If no valid dates remain after filtering, or (for the
            Month-Based-Split rolling-6-week path) no valid dates remain within the
            configured current monitoring period.
    """
    config = load_training_config(config_path)
    runtime_cfg = load_monitoring_runtime_config()
    monitoring_config = config.get("monitoring", {})
    output_dir = Path(monitoring_config.get("output_dir", str(DEFAULT_OUTPUT_DIR)))
    report_csv_path = output_dir / str(monitoring_config.get("report_csv_file", DEFAULT_REPORT_CSV))
    report_excel_path = output_dir / str(monitoring_config.get("report_excel_file", DEFAULT_REPORT_EXCEL))
    base_excel_path = report_excel_path
    run_number = 1
    report_excel_path = base_excel_path.with_name(
        f"{base_excel_path.stem}_{run_number}{base_excel_path.suffix}"
    )
    while report_excel_path.exists():
        run_number += 1
        report_excel_path = base_excel_path.with_name(
            f"{base_excel_path.stem}_{run_number}{base_excel_path.suffix}"
        )
    reference_path, current_path, report_html_path, report_json_path, _ = resolve_io_paths(config)
    configured_reference_path = runtime_cfg.get("baseline_prediction_path")
    configured_current_path = runtime_cfg.get("current_prediction_path")
    if configured_reference_path:
        reference_path = Path(str(configured_reference_path))
    if configured_current_path:
        current_path = Path(str(configured_current_path))

    reference_df = read_prediction_dataset(reference_path, "reference")
    current_df = read_prediction_dataset(current_path, "current")
    configured_date_column = str(config.get("dataset", {}).get("date_column", "opened_datetime"))
    date_column = str(runtime_cfg.get("batch_date_column", configured_date_column))
    monitoring_mode = str(runtime_cfg.get("monitoring_mode", "fortnight")).strip().lower()
    if monitoring_mode not in {"fortnight", "month", "year"}:
        monitoring_mode = "fortnight"
    batch_mode = str(runtime_cfg.get("batch_mode", "fixed_2_week")).strip().lower()

    actual_date_column = next((column for column in current_df.columns if str(column).strip().lower() == date_column.lower()), None)
    if actual_date_column is None:
        raise KeyError(f"Configured date column '{date_column}' is required for 2-week batch monitoring.")
    current_dates = pd.to_datetime(current_df[actual_date_column], errors="coerce")
    null_count = int(current_dates.isna().sum())
    if null_count:
        print(f"Warning: {null_count} rows have null/unparseable dates and will be excluded.")
    valid = current_df.loc[current_dates.notna()].copy()
    valid_dates = current_dates.loc[current_dates.notna()]
    if valid.empty:
        raise ValueError("No valid dates remaining in Current prediction dataset.")

    configured_target = str(config.get("model", {}).get("target", "change_failure_flag"))
    model_config = config.get("model", {})
    negative_class_label = str(model_config.get("neg_class_label", "N"))
    positive_class_label = str(model_config.get("pos_class_label", "Y"))
    target, prediction, probability, features = detect_columns(reference_df, valid, configured_target, date_column)
    normalized_reference, _, _ = normalize_binary_columns(reference_df, valid, target, prediction)
    split_type = str(runtime_cfg.get("split_type", "Default Split"))
    baseline_start = runtime_cfg.get("baseline_start_date", "N/A")
    baseline_end = runtime_cfg.get("baseline_end_date", "N/A")
    current_start_cfg = runtime_cfg.get("current_start_date", "N/A")
    current_end_cfg = runtime_cfg.get("current_end_date", "N/A")

    batch_results: list[dict[str, Any]] = []
    latest = valid_dates.max()
    latest_day = latest.normalize()
    latest_day_exclusive = latest_day + pd.Timedelta(days=1)
    batch_size_label = "14 days"

    if monitoring_mode == "month":
        mode_heading = "MONTHLY MONITORING"
        periods = sorted(valid_dates.dt.to_period("M").unique())
        print("=" * 60)
        print(mode_heading)
        print("=" * 60)
        print("Monitoring Mode: MONTH")
        print("Baseline Period:")
        print(f"Start Date: {baseline_start}")
        print(f"End Date: {baseline_end}")
        print("Current Period:")
        print(f"Start Date: {current_start_cfg}")
        print(f"End Date: {current_end_cfg}")
        print(f"Total Monthly Batches: {len(periods)}")
        batch_size_label = "1 month"
        for batch_number, month_period in enumerate(periods, start=1):
            batch_start = month_period.to_timestamp()
            month_end_exclusive = batch_start + pd.DateOffset(months=1)
            batch_end_exclusive = min(month_end_exclusive, latest_day_exclusive)
            if batch_start >= batch_end_exclusive:
                continue
            mask = (valid_dates >= batch_start) & (valid_dates < batch_end_exclusive)
            batch_df = valid.loc[mask].copy()
            if batch_df.empty:
                continue
            batch_end = batch_end_exclusive - pd.Timedelta(days=1)
            batch_id = f"MONTH_{batch_number:02d}"
            _, _, _, summary, resolved, snapshot = _run_single_monitoring(
                config_path,
                reference_df_override=reference_df,
                current_df_override=batch_df,
                write_outputs=False,
            )
            roc_auc_val = summary.get("classification_performance_metrics", {}).get("roc_auc")
            print("---")
            print(f"Batch ID: {batch_id}")
            print(f"Month Name: {batch_start.strftime('%B %Y')}")
            print(f"Start Date: {batch_start.date()}")
            print(f"End Date: {batch_end.date()}")
            print(f"Row Count: {len(batch_df)}")
            print(f"Dataset Drift: {summary['overall_data_drift_status']}")
            print(f"Prediction Drift: {summary.get('prediction_drift_status', 'N/A')}")
            print(f"ROC AUC: {_fmt(roc_auc_val)}")
            batch_results.append(
                {
                    "batch_id": batch_id,
                    "start": str(batch_start.date()),
                    "end": str(batch_end.date()),
                    "days": (batch_end - batch_start).days + 1,
                    "current_df": batch_df,
                    "summary": summary,
                    "snapshot": snapshot,
                }
            )
    elif monitoring_mode == "year":
        mode_heading = "YEARLY MONITORING"
        years = sorted(valid_dates.dt.year.unique())
        print("=" * 60)
        print(mode_heading)
        print("=" * 60)
        print("Monitoring Mode: YEAR")
        print("Baseline Period:")
        print(f"Start Date: {baseline_start}")
        print(f"End Date: {baseline_end}")
        print("Current Period:")
        print(f"Start Date: {current_start_cfg}")
        print(f"End Date: {current_end_cfg}")
        print(f"Total Yearly Batches: {len(years)}")
        batch_size_label = "1 year"
        for year in years:
            batch_start = pd.Timestamp(year=year, month=1, day=1)
            year_end_exclusive = pd.Timestamp(year=year + 1, month=1, day=1)
            batch_end_exclusive = min(year_end_exclusive, latest_day_exclusive)
            if batch_start >= batch_end_exclusive:
                continue
            mask = (valid_dates >= batch_start) & (valid_dates < batch_end_exclusive)
            batch_df = valid.loc[mask].copy()
            if batch_df.empty:
                continue
            batch_end = batch_end_exclusive - pd.Timedelta(days=1)
            batch_id = f"YEAR_{year}"
            _, _, _, summary, resolved, snapshot = _run_single_monitoring(
                config_path,
                reference_df_override=reference_df,
                current_df_override=batch_df,
                write_outputs=False,
            )
            roc_auc_val = summary.get("classification_performance_metrics", {}).get("roc_auc")
            print("---")
            print(f"Batch ID: {batch_id}")
            print(f"Year: {year}")
            print(f"Start Date: {batch_start.date()}")
            print(f"End Date: {batch_end.date()}")
            print(f"Row Count: {len(batch_df)}")
            print(f"Dataset Drift: {summary['overall_data_drift_status']}")
            print(f"Prediction Drift: {summary.get('prediction_drift_status', 'N/A')}")
            print(f"ROC AUC: {_fmt(roc_auc_val)}")
            batch_results.append(
                {
                    "batch_id": batch_id,
                    "start": str(batch_start.date()),
                    "end": str(batch_end.date()),
                    "days": (batch_end - batch_start).days + 1,
                    "current_df": batch_df,
                    "summary": summary,
                    "snapshot": snapshot,
                }
            )
    elif batch_mode == "rolling_6_week" and split_type == "Month-Based Split" and str(runtime_cfg.get("split_option", "")) == "3":
        print("=" * 60)
        print("ROLLING 6-WEEK MONITORING")
        print("=" * 60)
        print("Batch Mode: Rolling 6-Week Window")
        batch_size_label = "rolling 6 weeks (42 days, step 14 days)"

        configured_current_start = runtime_cfg.get("current_start_date")
        current_start = pd.to_datetime(configured_current_start, errors="coerce")
        if pd.isna(current_start):
            current_start = valid_dates.min().normalize()
        else:
            current_start = current_start.normalize()
        configured_current_end = runtime_cfg.get("current_end_date")
        current_end = pd.to_datetime(configured_current_end, errors="coerce")
        if pd.isna(current_end):
            current_end = valid_dates.max()
        else:
            current_end = current_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        current_period_mask = (valid_dates >= current_start) & (valid_dates <= current_end)
        current_period_dates = valid_dates.loc[current_period_mask]
        if current_period_dates.empty:
            raise ValueError("No valid dates remain within the configured Current monitoring period.")

        window_start = current_period_dates.max()
        window_number = 1
        while True:
            raw_window_end = window_start - pd.Timedelta(days=41)
            is_trimmed = raw_window_end < current_start
            window_end = current_start if is_trimmed else raw_window_end
            mask = (
                (valid_dates >= current_start)
                & (valid_dates <= current_end)
                & (valid_dates >= window_end)
                & (valid_dates <= window_start)
            )
            batch_df = valid.loc[mask].copy()
            batch_id = "ROLLING_LAST" if is_trimmed else f"ROLLING_{window_number:02d}"
            print("\n" + "-" * 60)
            print(f"Batch ID: {batch_id}")
            print("Batch Type: Rolling 6-Week Backward Window")
            print(f"Batch Start Date: {window_start.date()}")
            print(f"Batch End Date: {window_end.date()}")
            print(f"Window Length: {(window_start - window_end).days + 1} Days")
            print(f"Row Count: {len(batch_df)}")
            _, _, _, summary, resolved, snapshot = _run_single_monitoring(
                config_path,
                reference_df_override=reference_df,
                current_df_override=batch_df,
                write_outputs=False,
            )
            batch_results.append({
                "batch_id": batch_id,
                "start": str(window_start.date()),
                "end": str(window_end.date()),
                "days": (window_start - window_end).days + 1,
                "current_df": batch_df,
                "summary": summary,
                "snapshot": snapshot,
                "boundary_status": "TRIMMED_FINAL_WINDOW" if is_trimmed else "FULL_WINDOW",
            })

            if is_trimmed:
                print("Batch Boundary Status: TRIMMED_FINAL_WINDOW")
                print("Reached Current Start Date.")
                print("Stopping Rolling Batch Generation.")
                break

            window_start = window_start - pd.Timedelta(days=14)
            window_number += 1
    elif batch_mode == "rolling_6_week":
        print("=" * 60)
        print("ROLLING 6-WEEK MONITORING")
        print("=" * 60)
        print("Batch Mode: Rolling 6-Week Window")
        batch_size_label = "rolling 6 weeks (42 days, step 14 days)"

        configured_current_start = runtime_cfg.get("current_start_date")
        current_start = pd.to_datetime(configured_current_start, errors="coerce")
        if pd.isna(current_start):
            current_start = valid_dates.min().normalize()
        else:
            current_start = current_start.normalize()
        window_end = latest_day
        window_number = 1
        while True:
            window_start = window_end - pd.Timedelta(days=41)
            effective_start = max(window_start, current_start)
            mask = (valid_dates >= effective_start) & (valid_dates < window_end + pd.Timedelta(days=1))
            batch_df = valid.loc[mask].copy()
            if not batch_df.empty:
                is_trimmed = window_start < current_start
                batch_id = "ROLLING_LAST" if is_trimmed else f"ROLLING_{window_number:02d}"
                print("\n" + "-" * 60)
                print(f"Batch ID: {batch_id}")
                if is_trimmed:
                    print("Batch Type: Rolling 6-Week Window (Trimmed Final Batch)")
                print("Window: Latest 6 Weeks" if window_number == 1 else "Window: Previous Rolling Window")
                print(f"Start Date: {effective_start.date()}")
                print(f"End Date: {window_end.date()}")
                print("Window Length: Trimmed (Current Period Boundary)" if is_trimmed else "Window Length: 42 Days")
                print("Step Size: 14 Days")
                print(f"Row Count: {len(batch_df)}")
                _, _, _, summary, resolved, snapshot = _run_single_monitoring(
                    config_path,
                    reference_df_override=reference_df,
                    current_df_override=batch_df,
                    write_outputs=False,
                )
                batch_results.append({
                    "batch_id": batch_id,
                    "start": str(effective_start.date()),
                    "end": str(window_end.date()),
                    "days": (window_end - effective_start).days + 1,
                    "current_df": batch_df,
                    "summary": summary,
                    "snapshot": snapshot,
                    "boundary_status": "TRIMMED_FINAL_WINDOW" if is_trimmed else "FULL_WINDOW",
                })

                if is_trimmed:
                    print("Reached Current Start Date.")
                    print("Stopping Rolling Batch Generation.")
                    break

            window_end -= pd.Timedelta(days=14)
            window_number += 1
    else:
        print("=" * 60)
        print("FORTNIGHT MONITORING")
        print("=" * 60)
        print("Monitoring Mode: FORTNIGHT")
        print("Baseline Period:")
        print(f"Start Date: {baseline_start}")
        print(f"End Date: {baseline_end}")
        print("Current Period:")
        print(f"Start Date: {current_start_cfg}")
        print(f"End Date: {current_end_cfg}")
        batch_start = valid_dates.min().normalize()
        batch_number = 1
        while batch_start <= latest_day:
            batch_end_exclusive = batch_start + pd.Timedelta(days=14)
            mask = (valid_dates >= batch_start) & (valid_dates < batch_end_exclusive)
            batch_df = valid.loc[mask].copy()
            if not batch_df.empty:
                batch_id = f"batch_{batch_number:03d}"
                batch_end = min(batch_end_exclusive - pd.Timedelta(days=1), latest_day)
                _, _, _, summary, resolved, snapshot = _run_single_monitoring(
                    config_path,
                    reference_df_override=reference_df,
                    current_df_override=batch_df,
                    write_outputs=False,
                )
                roc_auc_val = summary.get("classification_performance_metrics", {}).get("roc_auc")
                print("---")
                print(f"Batch ID: {batch_id}")
                print(f"Start Date: {batch_start.date()}")
                print(f"End Date: {batch_end.date()}")
                print(f"Row Count: {len(batch_df)}")
                print(f"Dataset Drift: {summary['overall_data_drift_status']}")
                print(f"Prediction Drift: {summary.get('prediction_drift_status', 'N/A')}")
                print(f"ROC AUC: {_fmt(roc_auc_val)}")
                batch_results.append({
                    "batch_id": batch_id,
                    "start": str(batch_start.date()),
                    "end": str(batch_end.date()),
                    "days": (batch_end - batch_start).days + 1,
                    "current_df": batch_df,
                    "summary": summary,
                    "snapshot": snapshot,
                })
                batch_number += 1
            batch_start = batch_end_exclusive

    evidently_version = "N/A"
    try:
        import evidently as _ev
        evidently_version = _ev.__version__
    except Exception:
        pass
    _write_batch_csv(
        report_csv_path,
        reference_df,
        batch_results,
        features,
        target,
        prediction,
        probability,
        evidently_version,
        batch_size_label=batch_size_label,
        batch_mode=batch_mode,
    )
    _write_analysis_excel(
        report_excel_path,
        reference_df,
        batch_results,
        features,
        target,
        prediction,
        probability,
        evidently_version,
        reference_path,
        current_path,
        runtime_cfg,
        batch_size_label,
        negative_class_label,
        positive_class_label,
    )
    aggregate_payload = {
        "reference_dataset": str(reference_path),
        "current_dataset": str(current_path),
        "batch_count": len(batch_results),
        "batches": [
            {
                "batch_id": item["batch_id"],
                "start_date": item["start"],
                "end_date": item["end"],
                "row_count": len(item["current_df"]),
                "summary": item["summary"],
                "evidently_snapshot": item["snapshot"],
            }
            for item in batch_results
        ],
    }
    report_html_path.parent.mkdir(parents=True, exist_ok=True)
    with report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate_payload, handle, indent=2, default=_json_default)
    _write_batch_html(
        report_html_path,
        batch_results,
        reference_path,
        current_path,
        normalized_reference,
        features,
        target,
        prediction,
        probability,
    )
    aggregate_summary = {"total_reference_records": len(reference_df), "total_current_records": len(valid), "batch_count": len(batch_results)}
    resolved_columns = {"target_column": target, "prediction_column": prediction, "prediction_probability_column": probability, "feature_columns": features}
    return report_html_path, report_json_path, report_excel_path, aggregate_summary, resolved_columns


def main() -> None:
    """CLI entry point for standalone monitoring runs.

    Parses the --config argument (defaults to "train_config.json"), calls
    run_monitoring, and prints a completion summary (batch count, reference/current
    row counts, resolved column names, and the output CSV/Excel/HTML/JSON paths).
    """
    parser = argparse.ArgumentParser(description="Run Evidently monitoring on saved train/test prediction datasets.")
    parser.add_argument(
        "--config",
        default="train_config.json",
        help="Path to project training config JSON.",
    )
    args = parser.parse_args()

    report_html_path, report_json_path, report_csv_path, summary, resolved_columns = run_monitoring(Path(args.config))

    print("Evidently monitoring completed.")
    print(f"Total Batches Processed: {summary['batch_count']}")
    print(f"Reference Rows: {summary['total_reference_records']}")
    print(f"Total Current Rows Processed: {summary['total_current_records']}")
    print(f"Target column: {resolved_columns['target_column']}")
    print(f"Prediction column: {resolved_columns['prediction_column']}")
    print(f"Prediction probability column: {resolved_columns['prediction_probability_column']}")
    print(f"Feature columns ({len(resolved_columns['feature_columns'])}): {resolved_columns['feature_columns']}")
    print(f"Output Excel Workbook: {report_csv_path}")
    print(f"HTML Report: {report_html_path}")
    print(f"JSON Report: {report_json_path}")


if __name__ == "__main__":
    main()
