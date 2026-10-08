from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from snowflake.snowpark import Session

from app.database.snowflake_creds import (
    SF_ACCOUNT,
    SF_DATABASE,
    SF_PRIVATE_KEY_PASSPHRASE_PATH,
    SF_PRIVATE_KEY_PATH,
    SF_ROLE,
    SF_SCHEMA,
    SF_TEST_TABLE,
    SF_USER,
    SF_WAREHOUSE,
)
from app.database.snowflake_decrypt import extract_key_bytes, get_private_key
from app.training.preprocessed_data import PreprocessedData
from app.training.model_algorithm import SupervisedModel
from threshold_tuning import tune_thresholds_from_probabilities


PREDICTION_LABEL_COLUMN = "prediction_label"
PREDICTION_PROBABILITY_COLUMN = "prediction_probability"


def load_training_config(config_path: Path) -> dict:
    """Load and parse the training config JSON (BOM-tolerant)."""
    with config_path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve_feature_and_target_columns(config: dict) -> tuple[list[str], str]:
    """Resolve feature/target columns from either a flat or nested config shape.

    Raises ValueError if neither shape yields a valid feature list and target.
    """
    if "feature_columns" in config and "target_column" in config:
        feature_columns = [str(column) for column in config["feature_columns"]]
        target_column = str(config["target_column"])
        return feature_columns, target_column

    dataset = config.get("dataset", {})
    model = config.get("model", {})
    raw_features = dataset.get("features", [])
    feature_columns = [item["name"] for item in raw_features if isinstance(item, dict) and "name" in item]
    target_column = str(model.get("target", ""))

    if not feature_columns or not target_column:
        raise ValueError(
            "Config must contain either feature_columns/target_column or dataset.features/model.target."
        )

    return feature_columns, target_column


def build_query(config: dict, selected_columns: list[str]) -> str:
    """Build the SQL SELECT for Snowflake from dataset.dbconfig (table, filters, joins).

    Qualifies columns with the correct table alias when joins are configured.
    """
    dataset = config.get("dataset", {})
    dbconfig = dataset.get("dbconfig", {})
    table_name = dbconfig.get("table_name") or SF_TEST_TABLE

    if not table_name:
        raise ValueError("Table name is required in dataset.dbconfig.table_name or SF_TEST_TABLE.")

    filters = str(dbconfig.get("filters", "")).strip()
    joins = dbconfig.get("joins", []) or []

    if not joins:
        quoted_columns = ", ".join(selected_columns)
        base_query = f"SELECT {quoted_columns} FROM {table_name}"
        if filters:
            return f"{base_query} WHERE {filters}"
        return base_query

    base_alias = str(dbconfig.get("table_alias", "base")).strip()

    # Map each joined table's contributed columns to its alias so selections stay unambiguous.
    column_alias: dict[str, str] = {}
    from_parts = [f"{table_name} {base_alias}"]
    for join in joins:
        join_table = str(join.get("table", "")).strip()
        join_alias = str(join.get("alias", "")).strip()
        join_on = str(join.get("on", "")).strip()
        if not join_table or not join_alias or not join_on:
            raise ValueError("Each dataset.dbconfig.joins entry requires 'table', 'alias' and 'on'.")
        join_type = str(join.get("type", "left")).strip().upper()
        from_parts.append(f"{join_type} JOIN {join_table} {join_alias} ON {join_on}")
        for column in join.get("columns", []) or []:
            column_alias[str(column).strip().upper()] = join_alias

    qualified_columns = [
        f"{column_alias.get(str(column).strip().upper(), base_alias)}.{column}"
        for column in selected_columns
    ]

    base_query = f"SELECT {', '.join(qualified_columns)} FROM {' '.join(from_parts)}"
    if filters:
        return f"{base_query} WHERE {filters}"
    return base_query


def fetch_dataframe(query: str) -> pd.DataFrame:
    """Run a query on Snowflake (key-pair auth) and return the result as a DataFrame."""
    connection_params = {
        "account": SF_ACCOUNT,
        "user": SF_USER,
        "private_key": extract_key_bytes(
            get_private_key(SF_PRIVATE_KEY_PATH, SF_PRIVATE_KEY_PASSPHRASE_PATH)
        ),
        "warehouse": SF_WAREHOUSE,
        "database": SF_DATABASE,
        "schema": SF_SCHEMA,
        "role": SF_ROLE,
    }

    session = Session.builder.configs(connection_params).create()
    try:
        return session.sql(query).to_pandas()
    finally:
        session.close()


def split_data_like_production(
    data: pd.DataFrame,
    target: pd.Series,
    split_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test via random, stratified, or explicit date-range strategy
    (split_params["strategy"]["type"]).
    """
    params: dict[str, Any] = dict(split_params or {})
    strategy = params.pop("strategy", None)

    if not params and strategy is None:
        return train_test_split(data, target)

    if strategy is None:
        return train_test_split(data, target, **params)

    strategy_type = str(strategy.get("type", "")).strip().lower()
    if strategy_type == "stratify":
        return train_test_split(data, target, **params, stratify=target)

    if strategy_type == "date":
        date_col = str(strategy["column"]).strip().lower()
        date_data = data.copy()
        date_data[date_col] = pd.to_datetime(date_data[date_col]).dt.date

        min_date_train = strategy["min_date_train"]
        max_date_train = strategy["max_date_train"]
        min_date_test = strategy["min_date_test"]
        max_date_test = strategy["max_date_test"]

        train_split_ix = date_data.loc[
            (date_data[date_col] >= min_date_train) & (date_data[date_col] <= max_date_train)
        ]
        test_split_ix = date_data.loc[
            (date_data[date_col] >= min_date_test) & (date_data[date_col] <= max_date_test)
        ]

        x_train = data.loc[train_split_ix.index]
        x_test = data.loc[test_split_ix.index]
        y_train = target.loc[train_split_ix.index]
        y_test = target.loc[test_split_ix.index]
        return x_train, x_test, y_train, y_test

    return train_test_split(data, target, **params, stratify=target)


def build_preprocessed_data(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split_config: dict[str, Any],
    neg_label: str,
    pos_label: str,
) -> PreprocessedData:
    """Clean columns, binarize the target, detect categorical features, and split
    into train/validation/test via split_data_like_production.
    """
    dataframe = dataframe.copy()
    dataframe.columns = [str(column).strip().lower() for column in dataframe.columns]
    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]

    target_column = target_column.strip().lower()
    normalized_features = [column.strip().lower() for column in feature_columns]
    feature_columns = [column for column in normalized_features if column != target_column]
    feature_columns = list(dict.fromkeys(feature_columns))

    missing = [column for column in feature_columns + [target_column] if column not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required columns in Snowflake result: {missing}")

    selected = dataframe[feature_columns + [target_column]].copy()
    selected = selected.dropna(subset=[target_column])

    target_values = selected[target_column].astype(str).str.strip()
    class_labels = [neg_label, pos_label]
    unknown_labels = sorted(set(target_values.unique()) - set(class_labels))
    if unknown_labels:
        raise ValueError(
            f"Unexpected target labels {unknown_labels}. Expected labels: {class_labels}"
        )
    selected[target_column] = label_binarize(target_values, classes=class_labels).ravel().astype(int)

    X = selected[feature_columns]
    y = selected[target_column]

    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    split_params = dict(split_config or {})
    strategy = split_params.get("strategy")
    if isinstance(strategy, dict) and "column" in strategy:
        strategy = dict(strategy)
        strategy["column"] = str(strategy["column"]).strip().lower()
        split_params["strategy"] = strategy

    X_train, X_test, y_train, y_test = split_data_like_production(
        data=X,
        target=y,
        split_params=split_params,
    )

    for cat_col in categorical_features:
        if cat_col in X_train.columns:
            X_train[cat_col] = X_train[cat_col].fillna("nan").astype(str)
        if cat_col in X_test.columns:
            X_test[cat_col] = X_test[cat_col].fillna("nan").astype(str)

    return PreprocessedData(
        X_train=X_train,
        X_validation=X_test,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_test,
        y_test=y_test,
        categorical_features=categorical_features,
    )


def evaluate_split_metrics(
    trained_model,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Score a dataset with the model and compute accuracy/precision/recall/F1/ROC-AUC/PR-AUC/lift."""
    model_classes = list(getattr(trained_model, "classes_", []))
    y_prob_matrix = trained_model.predict_proba(features)
    if y_prob_matrix.shape[1] < 2:
        raise ValueError("Expected binary-class probability output with at least 2 columns.")
    if len(model_classes) < 2:
        raise ValueError("Expected trained model classes_ to contain two classes.")

    y_prob_pos = y_prob_matrix[:, 1]
    y_pred = (y_prob_pos >= threshold).astype(int)
    y_true = label_binarize(target.to_numpy(), classes=model_classes).ravel().astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

    if len(pd.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
        metrics["prauc"] = float(average_precision_score(y_true, y_prob_pos))

    overall_positive_rate = float(y_true.mean()) if y_true.mean() > 0 else 1.0
    sorted_indices = y_prob_pos.argsort()[::-1]
    n = len(y_true)
    for pct in (0.05, 0.10, 0.25):
        k = max(1, int(n * pct))
        top_k_positives = y_true[sorted_indices[:k]].sum()
        precision_at_k = top_k_positives / k
        metrics[f"lift_{int(pct * 100)}pct"] = float(precision_at_k / overall_positive_rate)

    return metrics


def resolve_monitoring_prediction_paths(config: dict) -> tuple[Path, Path]:
    """Resolve legacy train/test prediction paths for the no-date-column fallback workflow."""
    monitoring_config = config.get("monitoring", {})
    input_dir = Path(monitoring_config.get("input_dir", "monitoring/input"))
    train_prediction_name = str(monitoring_config.get("train_prediction_file", "train_prediction.csv"))
    test_prediction_name = str(monitoring_config.get("test_prediction_file", "test_prediction.csv"))
    return input_dir / train_prediction_name, input_dir / test_prediction_name


def save_prediction_artifact(
    trained_model,
    features: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    threshold: float,
    output_path: Path,
    artifact_metadata: pd.DataFrame | None = None,
) -> Path:
    """Score data with the model and write predictions + actuals to a CSV for monitoring."""
    y_prob_matrix = trained_model.predict_proba(features)
    if y_prob_matrix.shape[1] < 2:
        raise ValueError("Expected binary-class probability output with at least 2 columns.")

    y_prob_pos = y_prob_matrix[:, 1]
    y_pred = (y_prob_pos >= threshold).astype(int)

    prediction_df = features.copy().reset_index(drop=True)
    if artifact_metadata is not None:
        prediction_df = pd.concat(
            [prediction_df, artifact_metadata.reset_index(drop=True)],
            axis=1,
        )
    prediction_df[target_column] = target.astype(int).reset_index(drop=True)
    prediction_df[PREDICTION_LABEL_COLUMN] = y_pred
    prediction_df[PREDICTION_PROBABILITY_COLUMN] = y_prob_pos

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_path, index=False)
    return output_path


def build_prediction_dataframe(
    trained_model,
    features: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    threshold: float,
    artifact_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Like save_prediction_artifact but returns the DataFrame instead of saving it,
    so the caller can post-process it first (e.g. the development/reference file).
    """
    y_prob_matrix = trained_model.predict_proba(features)
    if y_prob_matrix.shape[1] < 2:
        raise ValueError("Expected binary-class probability output with at least 2 columns.")

    y_prob_pos = y_prob_matrix[:, 1]
    y_pred = (y_prob_pos >= threshold).astype(int)

    prediction_df = features.copy().reset_index(drop=True)
    if artifact_metadata is not None:
        prediction_df = pd.concat(
            [prediction_df, artifact_metadata.reset_index(drop=True)],
            axis=1,
        )

    prediction_df[target_column] = target.astype(int).reset_index(drop=True)
    prediction_df[PREDICTION_LABEL_COLUMN] = y_pred
    prediction_df[PREDICTION_PROBABILITY_COLUMN] = y_prob_pos
    return prediction_df


def print_temporal_split_validation(split_type: str, split_info: dict[str, Any]) -> None:
    """Pretty-print the chosen split's date ranges and row counts to stdout."""
    print("=" * 60)
    print("TEMPORAL SPLIT VALIDATION")
    print("=" * 60)
    print(f"Split Type: {split_type}")
    print(f"Development Start Date: {split_info['development_start_date']}")
    print(f"Development End Date: {split_info['development_end_date']}")
    print(f"Current Start Date: {split_info['current_start_date']}")
    print(f"Current End Date: {split_info['current_end_date']}")
    print(f"Development Row Count: {split_info['development_row_count']}")
    print(f"Current Row Count: {split_info['production_row_count']}")
    print("=" * 60)


def save_monitoring_runtime_config(
    monitoring_mode: str,
    baseline_prediction_path: Path,
    current_prediction_path: Path,
    split_option: str,
    split_type: str,
    split_summary: dict[str, Any],
    batch_date_column: str,
    batch_mode: str = "fixed_2_week",
) -> Path:
    """Write monitoring/monitoring_config.json, the handoff contract that tells
    evidently_monitor.py which prediction CSVs to compare and how to batch them.
    """
    payload = {
        "split_option": split_option,
        "split_type": split_type,
        "monitoring_mode": monitoring_mode,
        "batch_mode": batch_mode,
        "baseline_prediction_path": str(baseline_prediction_path),
        "current_prediction_path": str(current_prediction_path),
        "baseline_start_date": split_summary.get("development_start_date"),
        "baseline_end_date": split_summary.get("development_end_date"),
        "current_start_date": split_summary.get("current_start_date"),
        "current_end_date": split_summary.get("current_end_date"),
        "batch_date_column": batch_date_column,
        "split_summary": split_summary,
    }
    config_path = Path("monitoring/monitoring_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def prompt_positive_integer(label: str) -> int:
    """Prompt stdin for a positive integer, re-prompting until valid input is given."""
    while True:
        try:
            value = int(input(f"{label}: ").strip())
            if value > 0:
                return value
            print(f"Please enter a positive integer for {label.lower()}.")
        except ValueError:
            print(f"Please enter a valid integer for {label.lower()}.")


def prompt_monitoring_batch_mode() -> str:
    """Prompt for split-method-3's batch strategy: "fixed_2_week" or "rolling_6_week"."""
    print("\nSelect Monitoring Batch Mode:")
    print("1. Fixed 2-Week Batches")
    print("   Current dataset is split into consecutive non-overlapping 14-day batches.")
    print("2. Rolling 6-Week Window")
    print("   Current dataset is monitored using overlapping 6-week windows that move forward every 2 weeks.")
    while True:
        choice = input("Select monitoring batch mode: ").strip()
        if choice == "1":
            return "fixed_2_week"
        if choice == "2":
            return "rolling_6_week"
        print("Please select 1 or 2.")


def resolve_available_years(
    dataframe: pd.DataFrame,
    date_column: str,
) -> tuple[list[int], pd.Timestamp]:
    """Find valid historical years in date_column, excluding unparseable/future dates."""
    actual_col = next(
        (column for column in dataframe.columns if str(column).strip().lower() == date_column.strip().lower()),
        None,
    )
    if actual_col is None:
        raise KeyError(f"Date column '{date_column}' not found in dataset.")

    dates = pd.to_datetime(dataframe[actual_col], errors="coerce").dropna()
    today_boundary = pd.Timestamp.now().normalize()
    dates = dates.loc[dates < today_boundary]
    if dates.empty:
        raise ValueError(
            f"No valid historical dates found in '{date_column}' up to yesterday."
        )

    first_month = dates.min().to_period("M").to_timestamp()
    last_month = dates.max().to_period("M").to_timestamp()
    available_years = list(range(first_month.year, last_month.year + 1))
    return available_years, dates.max()


def select_year_plus_development_months_period(
    dataframe: pd.DataFrame,
    date_column: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp | None, int, int | None]:
    """Split method 4: prompt for a development start/duration, then a current-period mode
    (remaining data vs. a user-specified number of months).
    """
    available_years, latest_date = resolve_available_years(dataframe, date_column)

    print("\nSelect a starting year and Development duration in months. Current data begins immediately")
    print("after Development and can continue until the latest date or for a selected number of months.")
    development_start = prompt_month("Development Start", available_years)
    development_months = prompt_positive_integer("Development Period (months)")

    development_end = development_start + pd.DateOffset(months=development_months)
    current_start = development_end

    print("\nCurrent Period Selection")
    print("1. Remaining data until the latest available date.")
    print("2. Enter Current period in months.")
    while True:
        current_mode = input("Select current period mode (1 or 2): ").strip()
        if current_mode in ("1", "2"):
            break
        print("Please select 1 or 2.")

    current_months: int | None = None
    current_end: pd.Timestamp | None = None
    if current_mode == "2":
        current_months = prompt_positive_integer("Current Period (months)")
        current_end = current_start + pd.DateOffset(months=current_months)

    if development_start > latest_date:
        raise ValueError("Development start is after the latest available dataset date.")

    return development_start, development_end, current_start, current_end, development_months, current_months


def prompt_year_range(label: str, available_years: list[int]) -> tuple[int, int]:
    """Prompt for a start/end year pair, validating both are within the available range."""
    min_year, max_year = available_years[0], available_years[-1]
    while True:
        try:
            start_year = int(input(f"{label} Start Year ({min_year}-{max_year}): ").strip())
            end_year = int(input(f"{label} End Year ({min_year}-{max_year}): ").strip())
            if start_year not in available_years or end_year not in available_years:
                print(f"Please enter years between {min_year} and {max_year}.")
                continue
            if start_year > end_year:
                print(f"{label} Start Year must be less than or equal to End Year.")
                continue
            return start_year, end_year
        except ValueError:
            print("Please enter valid year integers.")


def select_historical_year_comparison_period(
    dataframe: pd.DataFrame,
    date_column: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, tuple[int, int], tuple[int, int]]:
    """Split method 6: prompt for a development year range and a later, non-overlapping current year range."""
    available_years, _ = resolve_available_years(dataframe, date_column)

    print("\nCompare one historical year or year range against selected future years.")
    development_start_year, development_end_year = prompt_year_range("Development", available_years)
    current_start_year, current_end_year = prompt_year_range("Current", available_years)

    if current_start_year <= development_end_year:
        raise ValueError("Current Start Year must be greater than Development End Year to avoid overlap.")

    development_start = pd.Timestamp(year=development_start_year, month=1, day=1)
    development_end = pd.Timestamp(year=development_end_year + 1, month=1, day=1)
    current_start = pd.Timestamp(year=current_start_year, month=1, day=1)
    current_end = pd.Timestamp(year=current_end_year + 1, month=1, day=1)

    return (
        development_start,
        development_end,
        current_start,
        current_end,
        (development_start_year, development_end_year),
        (current_start_year, current_end_year),
    )


def print_cv_metrics(cv_results: pd.DataFrame | None) -> None:
    """Print the CV results summary, handling both CatBoost-CV and GridSearchCV result shapes."""
    if cv_results is None or cv_results.empty:
        print("CV metrics: not available (set model.cv > 1 or enable grid search).")
        return

    print("CV metrics summary:")
    if "best_iter" in cv_results.columns:
        print(cv_results.to_string(index=True))
        return

    summary_columns = [column for column in cv_results.columns if column.startswith("mean_test_")]
    if summary_columns:
        score_series = cv_results[summary_columns[0]]
        if score_series.notna().sum() == 0:
            print("CV metrics summary unavailable: all grid-search scores are NaN.")
            print(cv_results[summary_columns].head(5).to_string(index=False))
            return
        best_row = cv_results.loc[score_series.idxmax()]
        print(best_row[summary_columns].to_string())
        return

    print(cv_results.head(5).to_string(index=False))


def perform_temporal_split(
    dataframe: pd.DataFrame,
    date_column: str,
    development_start: pd.Timestamp,
    development_end: pd.Timestamp,
    current_start: pd.Timestamp | None = None,
    current_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split a dataframe by date into development (training-eligible) and
    current/production (held out for monitoring) subsets, dropping null/future dates.
    """
    col_lower = date_column.strip().lower()
    actual_col = next(
        (c for c in dataframe.columns if str(c).strip().lower() == col_lower), None,
    )
    if actual_col is None:
        raise KeyError(
            f"Date column '{date_column}' not found in dataset. "
            f"Available columns: {sorted(dataframe.columns.tolist())}"
        )

    dates = pd.to_datetime(dataframe[actual_col], errors="coerce")
    null_count = int(dates.isna().sum())
    if null_count > 0:
        print(f"Warning: {null_count} rows have null/unparseable dates and will be excluded.")
        valid_mask = dates.notna()
        dataframe = dataframe.loc[valid_mask].copy()
        dates = dates.loc[valid_mask]

    today_boundary = pd.Timestamp.now().normalize()
    future_mask = dates >= today_boundary
    future_count = int(future_mask.sum())
    if future_count > 0:
        print(f"Warning: {future_count} rows have future opened_datetime and will be excluded.")
        observed_mask = ~future_mask
        dataframe = dataframe.loc[observed_mask].copy()
        dates = dates.loc[observed_mask]

    if dataframe.empty:
        raise ValueError("No valid historical dates remaining in dataset after filtering.")

    min_dataset_date = dates.min()
    max_dataset_date = dates.max()
    latest_data_boundary = max_dataset_date + pd.Timedelta(days=1)
    current_start = development_end if current_start is None else current_start
    current_end = latest_data_boundary if current_end is None else min(current_end, latest_data_boundary)
    if development_start >= development_end:
        raise ValueError("Development start month must be before development end month.")
    if current_start < development_end:
        raise ValueError("Current start month must be on or after the Development end month.")
    if current_start >= current_end:
        raise ValueError("Current start month must be before the Current end month.")
    dev_mask = (dates >= development_start) & (dates < development_end)
    prod_mask = (dates >= current_start) & (dates < current_end)

    development_df = dataframe.loc[dev_mask].copy()
    production_df = dataframe.loc[prod_mask].copy()

    if development_df.empty:
        raise ValueError(
            "Development dataset is empty for the selected date range. "
            f"Available date range: {min_dataset_date.date()} to {max_dataset_date.date()}. "
            f"Selected development range: {development_start.date()} to "
            f"{(development_end - pd.Timedelta(days=1)).date()}."
        )
    if production_df.empty:
        raise ValueError(
            "Current dataset is empty for the selected current date range. "
            f"Available date range: {min_dataset_date.date()} to {max_dataset_date.date()}. "
            f"Selected current range: {current_start.date()} to "
            f"{(current_end - pd.Timedelta(days=1)).date()}."
        )

    split_info: dict[str, Any] = {
        "max_dataset_date": str(max_dataset_date.date()),
        "development_start_date": str(development_start.date()),
        "development_end_date": str(development_end.date()),
        "current_start_date": str(current_start.date()),
        "current_end_date": str((current_end - pd.Timedelta(days=1)).date()),
        "development_row_count": len(development_df),
        "production_row_count": len(production_df),
    }
    return development_df, production_df, split_info


def prompt_month(label: str, available_years: list[int]) -> pd.Timestamp:
    """Prompt for a year (typed input) then a month (numbered menu) and return the first-of-month date.

    Args:
        label: Prompt label prefix (e.g. "Development Start").
        available_years: The valid years the user may choose from.

    Returns:
        A pd.Timestamp for day 1 of the chosen year/month.
    """
    min_year, max_year = available_years[0], available_years[-1]
    print(f"\n{label} Year (available range: {min_year}-{max_year}):")
    while True:
        try:
            year = int(input(f"Enter year ({min_year}-{max_year}): ").strip())
            if year in available_years:
                break
            print(f"Please enter a year between {min_year} and {max_year}.")
        except ValueError:
            print(f"Please enter a valid year between {min_year} and {max_year}.")

    months = list(range(1, 13))
    print(f"{label} Month:")
    for index, month in enumerate(months, start=1):
        print(f"  {index}. {pd.Timestamp(year=2000, month=month, day=1).strftime('%B')}")
    while True:
        try:
            month_choice = int(input("Select month number: ").strip())
            month = months[month_choice - 1]
            return pd.Timestamp(year=year, month=month, day=1)
        except (ValueError, IndexError):
            print("Please select a valid month number.")


def prompt_split_method() -> str:
    """Display the 6 temporal-split method options and return the user's choice ("1"-"6")."""
    print("=" * 60)
    print("DATA SPLITTING METHOD")
    print("=" * 60)
    print("\nHow would you like to split the data?\n")
    print("1. Default Split")
    print("   Latest 3 months of available data -> Current / Production.")
    print("   All historical data before the latest 3 months -> Development / Training.\n")
    print("2. Custom Date Range Split")
    print("   Choose the Development and Current date ranges manually.\n")
    print("3. Month-Based Split")
    print("   Enter Development months and Current months.")
    print("   The date ranges are calculated automatically from the latest available opened_datetime.\n")
    print("4. Year + Development Months Split")
    print("   Select a starting year and Development duration in months.")
    print("   Current data begins immediately after Development and can continue until")
    print("   the latest available date or use a user-selected Current period.\n")
    print("5. Monthly Iterative Monitoring Split")
    print("   Train on historical data once and monitor the Current period month by month.\n")
    print("6. Historical Year Comparison Split")
    print("   Compare one historical year or year range against selected future years.")
    while True:
        choice = input("Select splitting method: ").strip()
        if choice in ("1", "2", "3", "4", "5", "6"):
            return choice
        print("Please select 1, 2, 3, 4, 5, or 6.")


def prompt_month_based_split() -> tuple[int, int]:
    """Prompt for the Development and Production period durations in months (split methods 3/5)."""
    while True:
        try:
            development_months = int(input("Development Period (months): ").strip())
            if development_months > 0:
                break
            print("Please enter a positive integer for Development months.")
        except ValueError:
            print("Please enter a valid integer for Development months.")

    while True:
        try:
            production_months = int(input("Production Period (months): ").strip())
            if production_months > 0:
                break
            print("Please enter a positive integer for Production months.")
        except ValueError:
            print("Please enter a valid integer for Production months.")

    return development_months, production_months


def select_month_based_split_period(
    dataframe: pd.DataFrame,
    date_column: str,
    development_months: int,
    production_months: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Split methods 3/5: size the current/production period first (ending at the latest
    data), then the development period ends exactly where current begins.
    """
    actual_col = next(
        (column for column in dataframe.columns if str(column).strip().lower() == date_column.strip().lower()),
        None,
    )
    if actual_col is None:
        raise KeyError(f"Date column '{date_column}' not found in dataset.")

    dates = pd.to_datetime(dataframe[actual_col], errors="coerce").dropna()
    today_boundary = pd.Timestamp.now().normalize()
    observed_dates = dates.loc[dates < today_boundary]
    if observed_dates.empty:
        raise ValueError(f"No valid historical dates found in '{date_column}' up to yesterday.")

    max_dataset_date = observed_dates.max()
    current_end_exclusive = max_dataset_date.to_period("M").to_timestamp() + pd.DateOffset(months=1)
    # Current/Production period is calculated first, Development period ends where it begins.
    current_start = current_end_exclusive - pd.DateOffset(months=production_months)
    development_end = current_start
    development_start = development_end - pd.DateOffset(months=development_months)
    return development_start, development_end, current_end_exclusive


def select_default_split_period(
    dataframe: pd.DataFrame,
    date_column: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Split method 1 (default): latest 3 calendar months = current period, everything before = development."""
    actual_col = next(
        (column for column in dataframe.columns if str(column).strip().lower() == date_column.strip().lower()),
        None,
    )
    if actual_col is None:
        raise KeyError(f"Date column '{date_column}' not found in dataset.")

    dates = pd.to_datetime(dataframe[actual_col], errors="coerce").dropna()
    today_boundary = pd.Timestamp.now().normalize()
    observed_dates = dates.loc[dates < today_boundary]
    if observed_dates.empty:
        raise ValueError(f"No valid historical dates found in '{date_column}' up to yesterday.")

    latest_date = observed_dates.max()
    current_end_exclusive = latest_date.to_period("M").to_timestamp() + pd.DateOffset(months=1)
    current_start = current_end_exclusive - pd.DateOffset(months=3)
    development_start = observed_dates.min()
    return development_start, current_start


def select_development_period(
    dataframe: pd.DataFrame,
    date_column: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Split method 2: fully manual prompts for development and current date ranges
    (current must start after development ends).
    """
    actual_col = next(
        (column for column in dataframe.columns if str(column).strip().lower() == date_column.strip().lower()),
        None,
    )
    if actual_col is None:
        raise KeyError(f"Date column '{date_column}' not found in dataset.")

    dates = pd.to_datetime(dataframe[actual_col], errors="coerce").dropna()
    today_boundary = pd.Timestamp.now().normalize()
    dates = dates.loc[dates < today_boundary]
    if dates.empty:
        raise ValueError(f"No valid historical dates found in '{date_column}' up to yesterday.")

    first_month = dates.min().to_period("M").to_timestamp()
    last_month = dates.max().to_period("M").to_timestamp()
    available_years = list(range(first_month.year, last_month.year + 1))

    print("\nSelect the development period. The end month is the boundary:")
    print("development data = start month through the month before end month")
    development_start = prompt_month("Development Start", available_years)
    development_end = prompt_month("Development End", available_years)
    if development_start >= development_end:
        raise ValueError("Development start month must be before development end month.")

    print("\nSelect the current period after the Development end month:")
    print("current data = selected start month through the selected end month")
    current_start = prompt_month_after(
        "Current Start", development_end + pd.DateOffset(months=1), first_month, last_month,
    )
    current_end_month = prompt_month_after(
        "Current End", current_start, first_month, last_month,
    )
    current_end = current_end_month + pd.DateOffset(months=1)
    return development_start, development_end, current_start, current_end


def prompt_month_after(
    label: str,
    minimum_month: pd.Timestamp,
    first_month: pd.Timestamp,
    last_month: pd.Timestamp,
) -> pd.Timestamp:
    """Prompt for a month not before minimum_month (helper for select_development_period)."""
    available_years = list(range(minimum_month.year, last_month.year + 1))
    while True:
        selected_year = int(
            input(
                f"{label} Year (available range: {available_years[0]}-{available_years[-1]}): "
            ).strip()
        )
        if selected_year in available_years:
            break
        print(f"Please enter a year between {available_years[0]} and {available_years[-1]}.")

    first_allowed_month = minimum_month.month if selected_year == minimum_month.year else 1
    last_allowed_month = last_month.month if selected_year == last_month.year else 12
    months = list(range(first_allowed_month, last_allowed_month + 1))
    print(f"{label} Month:")
    for index, month in enumerate(months, start=1):
        print(f"  {index}. {pd.Timestamp(year=2000, month=month, day=1).strftime('%B')}")
    while True:
        try:
            month_choice = int(input("Select month number: ").strip())
            return pd.Timestamp(year=selected_year, month=months[month_choice - 1], day=1)
        except (ValueError, IndexError):
            print("Please select a valid month number.")


def prepare_production_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    neg_label: str,
    pos_label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess the held-out production/current data using the same cleaning rules as
    training, but only for scoring — never splits or trains on the result.
    """
    dataframe = dataframe.copy()
    dataframe.columns = [str(c).strip().lower() for c in dataframe.columns]

    target_col = target_column.strip().lower()
    norm_features = [c.strip().lower() for c in feature_columns]
    norm_features = [c for c in norm_features if c != target_col]
    norm_features = list(dict.fromkeys(norm_features))

    missing = [c for c in norm_features + [target_col] if c not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required columns in production data: {missing}")

    selected = dataframe[norm_features + [target_col]].copy()
    selected = selected.dropna(subset=[target_col])

    target_values = selected[target_col].astype(str).str.strip()
    class_labels = [neg_label, pos_label]
    unknown = sorted(set(target_values.unique()) - set(class_labels))
    if unknown:
        raise ValueError(f"Unexpected target labels {unknown}. Expected: {class_labels}")
    selected[target_col] = label_binarize(target_values, classes=class_labels).ravel().astype(int)

    X = selected[norm_features]
    y = selected[target_col]

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].fillna("nan").astype(str)

    return X, y


def main() -> None:
    """Run the end-to-end training pipeline: fetch, split, preprocess, train, tune
    threshold, and save the model + prediction artifacts + monitoring config.
    """
    parser = argparse.ArgumentParser(description="Minimal Snowflake-to-CatBoost training.")
    parser.add_argument(
        "--config",
        default="train_config.json",
        help="Path to training config JSON.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_training_config(config_path)

    feature_columns, target_column = resolve_feature_and_target_columns(config)

    # Temporal split configuration
    dataset_cfg = config.get("dataset", {})
    date_column = dataset_cfg.get("date_column")

    # Include date column in query when configured
    query_columns = list(dict.fromkeys(list(feature_columns) + [target_column]))
    if date_column:
        date_upper = date_column.strip().upper()
        if date_upper not in {c.strip().upper() for c in query_columns}:
            query_columns.append(date_column)

    query = build_query(config, query_columns)
    dataframe = fetch_dataframe(query)

    model_cfg = config.get("model", {})
    split_cfg = model_cfg.get("data_split", config.get("split", {}))
    pos_label = str(model_cfg.get("pos_class_label", "Y"))
    neg_label = str(model_cfg.get("neg_class_label", "N"))

    # Temporal split: development vs production
    production_df = None
    split_method = "1"
    split_type_label = "Default Split"
    monitoring_mode = "fortnight"
    batch_mode = "fixed_2_week"
    monthly_dev_months: int | None = None
    monthly_current_months: int | None = None
    option4_dev_months: int | None = None
    option4_current_months: int | None = None
    option6_dev_years: tuple[int, int] | None = None
    option6_cur_years: tuple[int, int] | None = None
    if date_column:
        split_method = prompt_split_method()
        if split_method == "1":
            development_start, development_end = select_default_split_period(dataframe, date_column)
            current_start = None
            current_end = None
            split_type_label = "Default Split"
        elif split_method == "2":
            development_start, development_end, current_start, current_end = select_development_period(
                dataframe, date_column,
            )
            split_type_label = "Custom Date Range Split"
        elif split_method == "3":
            development_months, production_months = prompt_month_based_split()
            batch_mode = prompt_monitoring_batch_mode()
            development_start, development_end, current_end = select_month_based_split_period(
                dataframe, date_column, development_months, production_months,
            )
            current_start = None
            split_type_label = "Month-Based Split"
            monitoring_mode = "fortnight"
            monthly_dev_months = development_months
            monthly_current_months = production_months
        elif split_method == "4":
            (
                development_start,
                development_end,
                current_start,
                current_end,
                option4_dev_months,
                option4_current_months,
            ) = select_year_plus_development_months_period(
                dataframe,
                date_column,
            )
            split_type_label = "Year + Development Months Split"
        elif split_method == "5":
            development_months, production_months = prompt_month_based_split()
            development_start, development_end, current_end = select_month_based_split_period(
                dataframe, date_column, development_months, production_months,
            )
            current_start = None
            split_type_label = "Monthly Iterative Monitoring Split"
            monitoring_mode = "month"
            monthly_dev_months = development_months
            monthly_current_months = production_months
        else:
            (
                development_start,
                development_end,
                current_start,
                current_end,
                option6_dev_years,
                option6_cur_years,
            ) = select_historical_year_comparison_period(dataframe, date_column)
            split_type_label = "Historical Year Comparison Split"
            monitoring_mode = "year"

        development_df, production_df, split_info = perform_temporal_split(
            dataframe,
            date_column,
            development_start,
            development_end,
            current_start=current_start,
            current_end=current_end,
        )

        if split_method in ("3", "5"):
            print("=" * 60)
            print("MONTH-BASED TEMPORAL SPLIT")
            print("=" * 60)
            print(f"Max Dataset Date: {split_info['max_dataset_date']}")
            print(f"Development Months: {monthly_dev_months}")
            print(f"Current Months: {monthly_current_months}")
            print("=" * 60)

        if split_method == "4":
            print("=" * 60)
            print("YEAR + DEVELOPMENT MONTHS SPLIT")
            print("=" * 60)
            print(f"Development Period (months): {option4_dev_months}")
            if option4_current_months is None:
                print("Current Period: Remaining data until latest available date")
            else:
                print(f"Current Period (months): {option4_current_months}")
            print("=" * 60)

        if split_method == "6" and option6_dev_years and option6_cur_years:
            print("=" * 60)
            print("HISTORICAL YEAR COMPARISON SPLIT")
            print("=" * 60)
            print(f"Development Years: {option6_dev_years[0]}-{option6_dev_years[1]}")
            print(f"Current Years: {option6_cur_years[0]}-{option6_cur_years[1]}")
            print("=" * 60)

        print_temporal_split_validation(split_type_label, split_info)

        training_dataframe = development_df
    else:
        training_dataframe = dataframe

    prepared = build_preprocessed_data(
        dataframe=training_dataframe,
        feature_columns=feature_columns,
        target_column=target_column,
        split_config=split_cfg,
        neg_label=neg_label,
        pos_label=pos_label,
    )

    training_cfg = config.get("training", {})

    algorithm = SupervisedModel()
    result = algorithm.fit(
        x_train=prepared.X_train,
        y_train=prepared.y_train,
        x_validation=prepared.X_validation,
        y_validation=prepared.y_validation,
        cat_features=prepared.categorical_features,
        model_config=model_cfg,
        training_config=training_cfg,
    )

    output_path = Path(config.get("model_output_path", "artifacts/model.cbm"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.model.save_model(output_path)

    threshold_result = tune_thresholds_from_probabilities(
        y_true=prepared.y_test.astype(int).to_numpy(),
        y_prob_pos=result.model.predict_proba(prepared.X_test)[:, 1],
        csv_path=output_path.parent / "threshold_tuning.csv",
        html_path=output_path.parent / "threshold_tuning.html",
        title="Threshold tuning report",
    )
    chosen_threshold = threshold_result.best_threshold
    target_column_name = str(prepared.y_train.name or target_column).strip().lower()

    gridsearch_enabled = bool(training_cfg.get("gridsearch", model_cfg.get("gridsearch", False)))
    optuna_enabled = bool(training_cfg.get("optuna", model_cfg.get("optuna", False)))
    if (gridsearch_enabled or optuna_enabled) and result.selected_hyperparameters:
        print(f"Selected model hyperparameters: {result.selected_hyperparameters}")

    if production_df is not None:
        # Temporal split workflow: development reference + production current
        monitoring_cfg = config.get("monitoring", {})
        input_dir = Path(monitoring_cfg.get("input_dir", "monitoring/input"))
        ref_name = monitoring_cfg.get("reference_prediction_file", "development_prediction.csv")
        cur_name = monitoring_cfg.get("current_prediction_file", "production_current_prediction.csv")

        dev_X = pd.concat([prepared.X_train, prepared.X_test], axis=0)
        dev_y = pd.concat([prepared.y_train, prepared.y_test], axis=0)
        date_actual_col = next(
            (column for column in development_df.columns
             if str(column).strip().lower() == str(date_column).strip().lower()),
            None,
        )
        if date_actual_col is None:
            raise KeyError(f"Date column '{date_column}' not found in development dataset.")
        reference_prediction_df = build_prediction_dataframe(
            trained_model=result.model,
            features=dev_X,
            target=dev_y,
            target_column=target_column_name,
            threshold=chosen_threshold,
            artifact_metadata=development_df.loc[dev_X.index, [date_actual_col]],
        )
        reference_prediction_df[date_actual_col] = pd.to_datetime(
            reference_prediction_df[date_actual_col],
            errors="coerce",
        )
        saved_ref_path = input_dir / ref_name
        saved_ref_path.parent.mkdir(parents=True, exist_ok=True)
        reference_prediction_df.to_csv(saved_ref_path, index=False)

        prod_X, prod_y = prepare_production_features(
            production_df, feature_columns, target_column, neg_label, pos_label,
        )
        saved_cur_path = save_prediction_artifact(
            trained_model=result.model, features=prod_X, target=prod_y,
            target_column=target_column_name, threshold=chosen_threshold,
            output_path=input_dir / cur_name,
            artifact_metadata=production_df.loc[prod_X.index, [date_actual_col]],
        )

        dev_train_metrics = evaluate_split_metrics(result.model, prepared.X_train, prepared.y_train, chosen_threshold)
        dev_test_metrics = evaluate_split_metrics(result.model, prepared.X_test, prepared.y_test, chosen_threshold)
        production_metrics = evaluate_split_metrics(result.model, prod_X, prod_y, chosen_threshold)

        print(f"Development train metrics: {dev_train_metrics}")
        print(f"Development test metrics: {dev_test_metrics}")
        print(f"Production/current metrics: {production_metrics}")
        print(f"Chosen decision threshold: {chosen_threshold:.2f}")
        print(f"Reference prediction artifact: {saved_ref_path}")
        print(f"Current prediction artifact: {saved_cur_path}")
        monitoring_config_path = save_monitoring_runtime_config(
            monitoring_mode=monitoring_mode,
            baseline_prediction_path=saved_ref_path,
            current_prediction_path=saved_cur_path,
            split_option=split_method,
            split_type=split_type_label,
            split_summary=split_info,
            batch_date_column="opened_datetime" if split_method == "1" else str(date_column),
            batch_mode=batch_mode,
        )
        print(f"Monitoring runtime config: {monitoring_config_path}")
    else:
        # Original workflow: train/test prediction artifacts
        train_prediction_path, test_prediction_path = resolve_monitoring_prediction_paths(config)
        saved_train_path = save_prediction_artifact(
            trained_model=result.model, features=prepared.X_train, target=prepared.y_train,
            target_column=target_column_name, threshold=chosen_threshold,
            output_path=train_prediction_path,
        )
        saved_test_path = save_prediction_artifact(
            trained_model=result.model, features=prepared.X_test, target=prepared.y_test,
            target_column=target_column_name, threshold=chosen_threshold,
            output_path=test_prediction_path,
        )

        train_metrics = evaluate_split_metrics(result.model, prepared.X_train, prepared.y_train, chosen_threshold)
        test_metrics = evaluate_split_metrics(result.model, prepared.X_test, prepared.y_test, chosen_threshold)

        print(f"Train metrics: {train_metrics}")
        print(f"Test metrics: {test_metrics}")
        print(f"Chosen decision threshold: {chosen_threshold:.2f}")
        print(f"Train prediction artifact: {saved_train_path}")
        print(f"Test prediction artifact: {saved_test_path}")


if __name__ == "__main__":
    main()
