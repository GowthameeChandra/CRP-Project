from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv as catboost_cv
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class ModelFitResult:
    """Immutable result of SupervisedModel.fit().

    Attributes:
        model: The trained CatBoostClassifier, ready for predict/predict_proba.
        best_iteration: The boosting iteration CatBoost's early stopping selected as
            best, or None if unavailable.
        best_score: Dict of evaluation-set metric name -> best score achieved, as
            reported by CatBoost.
        cv_results: Cross-validation results DataFrame (shape depends on strategy:
            CatBoost-CV summary, GridSearchCV cv_results_, or Optuna trials_dataframe),
            or None if CV/search wasn't run.
        selected_hyperparameters: The winning hyperparameter dict when GridSearch or
            Optuna was used, or None for the fixed-params path.
    """
    model: CatBoostClassifier
    best_iteration: int | None
    best_score: dict[str, Any]
    cv_results: pd.DataFrame | None
    selected_hyperparameters: dict[str, Any] | None


class SupervisedModel:
    """CatBoost training engine supporting fixed hyperparameters, GridSearchCV, or Optuna search.

    The public entry point is fit(), which reads model_config/training_config to decide
    which of the three mutually-exclusive optimization strategies to use, then trains a
    CatBoostClassifier using only the provided train/validation data (never test or
    production data). See fit() for full details of each strategy.
    """

    def __init__(self) -> None:
        """Initialize a module-level logger for this instance."""
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _resolve_positive_label(y_values: pd.Series, configured_pos_label: str) -> Any:
        """Determine which label value represents the positive class for scoring.

        Args:
            y_values: The target series (already binarized to 0/1 in most flows).
            configured_pos_label: The pos_class_label configured in train_config.json,
                used as a fallback when the target isn't already 0/1 encoded.

        Returns:
            1 if y_values only contains {0, 1}; otherwise configured_pos_label.
        """
        unique_values = set(y_values.dropna().tolist())
        if unique_values.issubset({0, 1}):
            return 1
        return configured_pos_label

    @staticmethod
    def _build_scoring(metric_names: list[str], pos_label: Any) -> dict[str, Any]:
        """Build a dict of sklearn scorer callables for use as GridSearchCV(scoring=...).

        Args:
            metric_names: Names of metrics to build scorers for; supported values are
                "average_precision", "precision", "recall", and "f1". Unrecognized
                names are silently skipped.
            pos_label: The label value considered the positive class.

        Returns:
            Dict mapping each recognized metric name to a make_scorer callable.
        """
        scoring_map: dict[str, Any] = {}
        for metric_name in metric_names:
            if metric_name == "average_precision":
                scoring_map[metric_name] = make_scorer(
                    average_precision_score,
                    response_method="predict_proba",
                    pos_label=pos_label,
                )
            elif metric_name == "precision":
                scoring_map[metric_name] = make_scorer(
                    precision_score,
                    pos_label=pos_label,
                    zero_division=0,
                )
            elif metric_name == "recall":
                scoring_map[metric_name] = make_scorer(
                    recall_score,
                    pos_label=pos_label,
                    zero_division=0,
                )
            elif metric_name == "f1":
                scoring_map[metric_name] = make_scorer(
                    f1_score,
                    pos_label=pos_label,
                    zero_division=0,
                )
        return scoring_map

    @staticmethod
    def get_catboost_cv_df(cb_cv_results: pd.DataFrame, metric: str = "test-PRAUC-mean") -> pd.DataFrame:
        """Summarize raw catboost.cv output at its best-scoring iteration.

        catboost.cv returns one row per boosting iteration with '-mean'/'-std' suffixed
        columns per metric. This finds the iteration where `metric` is maximized, then
        reshapes that single row into two rows ("mean" and "std") with the suffixes
        stripped from the column names, prefixed by a 'best_iter' column.

        Args:
            cb_cv_results: Raw DataFrame returned by catboost.cv(as_pandas=True).
            metric: Column name to maximize over to pick the best iteration (must
                exist in cb_cv_results.columns).

        Returns:
            A 2-row DataFrame indexed by ["mean", "std"], with a 'best_iter' column.

        Raises:
            ValueError: If `metric` is not a column in cb_cv_results.
        """
        if metric not in cb_cv_results.columns:
            raise ValueError(f"Metric {metric} not found. Available: {list(cb_cv_results.columns)}")

        best_iter = int(cb_cv_results[metric].idxmax())

        mean_cols = [column for column in cb_cv_results.columns if column.endswith("mean")]
        mean_row = cb_cv_results.loc[[best_iter], mean_cols].copy()
        mean_row.columns = [column.replace("-mean", "") for column in mean_row.columns]
        mean_row.index = ["mean"]
        mean_row.insert(0, "best_iter", best_iter)

        std_cols = [column for column in cb_cv_results.columns if column.endswith("std")]
        std_row = cb_cv_results.loc[[best_iter], std_cols].copy()
        std_row.columns = [column.replace("-std", "") for column in std_row.columns]
        std_row.index = ["std"]
        std_row.insert(0, "best_iter", best_iter)

        return pd.concat([mean_row, std_row])

    @staticmethod
    def _resolve_optimization_flags(model_config: dict[str, Any], training_config: dict[str, Any]) -> tuple[bool, bool]:
        """Decide which hyperparameter optimization strategy (if any) is enabled.

        Checks training_config first, falling back to model_config, for both the
        'gridsearch' and 'optuna' boolean flags.

        Args:
            model_config: The 'model' section of train_config.json.
            training_config: The 'training' section of train_config.json.

        Returns:
            A tuple of (gridsearch_enabled, optuna_enabled). At most one is True.

        Raises:
            ValueError: If both gridsearch and optuna are enabled simultaneously.
        """
        gridsearch_flag = training_config.get("gridsearch")
        if gridsearch_flag is None:
            gridsearch_flag = model_config.get("gridsearch", False)

        optuna_flag = training_config.get("optuna")
        if optuna_flag is None:
            optuna_flag = model_config.get("optuna", False)

        gridsearch = bool(gridsearch_flag)
        optuna_enabled = bool(optuna_flag)

        if gridsearch and optuna_enabled:
            raise ValueError("Invalid configuration: only one optimization strategy can be enabled at a time.")

        return gridsearch, optuna_enabled

    @staticmethod
    def _get_optuna_setting(
        training_config: dict[str, Any],
        key: str,
        default: Any,
        fallback_key: str | None = None,
    ) -> Any:
        """Look up an Optuna-related setting, checking multiple config locations in order.

        Checks (1) `key` directly in training_config, (2) `key` inside
        training_config["optuna_config"], (3) `fallback_key` inside
        training_config["optuna_config"] if provided, then (4) returns `default`.

        Args:
            training_config: The 'training' section of train_config.json.
            key: Primary setting name to look up (e.g. "optuna_trials").
            default: Value to return if the setting isn't found anywhere.
            fallback_key: Alternate name to check inside optuna_config (e.g. "n_trials").

        Returns:
            The resolved setting value.
        """
        if key in training_config:
            return training_config[key]
        optuna_cfg = training_config.get("optuna_config", {})
        if key in optuna_cfg:
            return optuna_cfg[key]
        if fallback_key and fallback_key in optuna_cfg:
            return optuna_cfg[fallback_key]
        return default

    @staticmethod
    def _sanitize_custom_metrics(params: dict[str, Any]) -> None:
        """Remove noisy per-class precision/recall/f1 entries from a custom_metric list, in place.

        CatBoost logs a separate line per class for metrics starting with 'precision',
        'recall', or 'f1', which clutters training output without adding useful signal.
        Mutates params directly; if no metrics remain after filtering, the
        'custom_metric' key is removed entirely.

        Args:
            params: A CatBoost parameter dict, potentially containing a 'custom_metric'
                list. No-op if the key is absent or not a list.
        """
        custom_metrics = params.get("custom_metric")
        if not isinstance(custom_metrics, list):
            return

        noisy_metric_prefixes = ("precision", "recall", "f1")
        filtered_metrics = [
            metric
            for metric in custom_metrics
            if isinstance(metric, str)
            and not any(metric.strip().lower().startswith(prefix) for prefix in noisy_metric_prefixes)
        ]

        if filtered_metrics:
            params["custom_metric"] = filtered_metrics
        else:
            params.pop("custom_metric", None)

    def _run_optuna_search(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame,
        y_validation: pd.Series,
        cat_features: list[str],
        hyper_cfg: dict[str, Any],
        training_config: dict[str, Any],
        cv_folds: int,
        pos_label: Any,
    ) -> tuple[CatBoostClassifier, pd.DataFrame, dict[str, Any]]:
        """Run an Optuna hyperparameter search and refit a final model with the best trial's params.

        Each trial samples learning_rate, depth, iterations, l2_leaf_reg,
        random_strength, and bagging_temperature from configured (or default) ranges,
        then scores that combination by running catboost.cv on x_train/y_train and
        reading the final 'test-PRAUC-mean' value (to maximize). After n_trials (or a
        timeout), the best-found parameters are merged with the base params and used to
        fit one final CatBoostClassifier on the full x_train/y_train, using
        x_validation/y_validation as the eval_set for early stopping if enabled.

        Args:
            x_train: Training features.
            y_train: Training target.
            x_validation: Validation features, used only for early stopping in the
                final refit (not during the Optuna trials themselves).
            y_validation: Validation target.
            cat_features: List of categorical feature column names/indices.
            hyper_cfg: The model.hyperparams.catboost config section (contains
                'default' base params and an 'optuna' search-space section).
            training_config: The 'training' config section (optuna_trials,
                optuna_timeout, optuna_seed settings).
            cv_folds: Number of CV folds to use when scoring each trial (must be >= 2).
            pos_label: The label value considered the positive class (unused directly
                here but kept for interface consistency with fit()).

        Returns:
            A tuple of (final_model, trials_dataframe, best_params) where
            trials_dataframe is Optuna's study.trials_dataframe() and best_params is
            the winning hyperparameter dict (learning_rate, depth, etc.).

        Raises:
            ValueError: If cv_folds < 2, or if the 'test-PRAUC-mean' metric is missing
                from a trial's catboost.cv output.
        """
        base_params = dict(hyper_cfg.get("default", {}))
        self._sanitize_custom_metrics(base_params)
        use_eval_set = bool(base_params.pop("eval_set", True))
        base_params.setdefault("logging_level", "Silent")

        optuna_space = dict(hyper_cfg.get("optuna", {}))
        n_trials = int(self._get_optuna_setting(training_config, "optuna_trials", 30, fallback_key="n_trials"))
        timeout_raw = self._get_optuna_setting(training_config, "optuna_timeout", None, fallback_key="timeout")
        timeout = int(timeout_raw) if timeout_raw not in (None, "") else None
        seed = int(self._get_optuna_setting(training_config, "optuna_seed", base_params.get("random_seed", 42), fallback_key="seed"))
        if cv_folds < 2:
            raise ValueError("Invalid configuration: model.cv must be >= 2 when Optuna is enabled.")

        train_pool = Pool(data=x_train, label=y_train, cat_features=cat_features)

        def _range(name: str, default_low: float, default_high: float) -> tuple[float, float]:
            value = optuna_space.get(name)
            if isinstance(value, list) and len(value) == 2:
                return float(value[0]), float(value[1])
            return default_low, default_high

        def _int_range(name: str, default_low: int, default_high: int) -> tuple[int, int]:
            low, high = _range(name, float(default_low), float(default_high))
            return int(low), int(high)

        def objective(trial: optuna.trial.Trial) -> float:
            params = dict(base_params)
            params.update(
                {
                    "learning_rate": trial.suggest_float("learning_rate", *_range("learning_rate", 0.01, 0.3), log=True),
                    "depth": trial.suggest_int("depth", *_int_range("depth", 4, 10)),
                    "iterations": trial.suggest_int("iterations", *_int_range("iterations", 300, 1200)),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *_range("l2_leaf_reg", 1.0, 15.0), log=True),
                    "random_strength": trial.suggest_float("random_strength", *_range("random_strength", 0.0, 5.0)),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", *_range("bagging_temperature", 0.0, 5.0)),
                    "random_seed": seed,
                }
            )

            cv_params = dict(params)
            cv_raw = catboost_cv(
                params=cv_params,
                pool=train_pool,
                nfold=cv_folds,
                stratified=True,
                as_pandas=True,
            )
            metric = "test-PRAUC-mean"
            if metric not in cv_raw.columns:
                raise ValueError(f"Metric {metric} not found in CatBoost CV output. Available: {list(cv_raw.columns)}")
            mean_score = float(cv_raw[metric].iloc[-1])
            self.logger.info(
                "Optuna trial=%s params=%s cv_prauc=%.6f",
                trial.number,
                trial.params,
                mean_score,
            )
            return mean_score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        started_at = time.perf_counter()

        def on_trial_complete(study_obj: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            best_value = study_obj.best_value if study_obj.best_trial is not None else trial.value
            self.logger.info(
                "Optuna trial complete: trial=%s score=%.6f current_best=%.6f",
                trial.number,
                float(trial.value) if trial.value is not None else float("nan"),
                float(best_value) if best_value is not None else float("nan"),
            )

        self.logger.info(
            "Starting Optuna optimization: n_trials=%s timeout=%s seed=%s",
            n_trials,
            timeout,
            seed,
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[on_trial_complete])
        elapsed_seconds = time.perf_counter() - started_at

        best_params = dict(study.best_params)
        self.logger.info("Optuna best score (PR AUC): %.6f", float(study.best_value))
        self.logger.info("Optuna best parameters: %s", best_params)
        self.logger.info("Optuna total optimization time (seconds): %.2f", elapsed_seconds)

        final_params = dict(base_params)
        final_params.update(best_params)
        final_params["random_seed"] = seed

        final_model = CatBoostClassifier(**final_params)
        fit_kwargs: dict[str, Any] = {
            "X": x_train,
            "y": y_train,
            "cat_features": cat_features,
            "verbose": False,
        }
        if use_eval_set:
            fit_kwargs["eval_set"] = (x_validation, y_validation)
        final_model.fit(**fit_kwargs)

        trials_df = study.trials_dataframe()
        return final_model, trials_df, best_params

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame,
        y_validation: pd.Series,
        cat_features: list[str],
        model_config: dict[str, Any],
        training_config: dict[str, Any],
    ) -> ModelFitResult:
        """Train a CatBoost model using the strategy configured in model_config/training_config.

        Reads model_config["cv"] for the number of CV folds and resolves the
        optimization strategy via _resolve_optimization_flags, then dispatches to one
        of three branches:

          1. GridSearch (model_config/training_config["gridsearch"] == True): builds a
             base CatBoostClassifier plus a param_grid from
             hyper_cfg["gridsearch"] (normalizing the depth/max_depth synonym
             conflict), wraps it in sklearn's GridSearchCV with StratifiedKFold(cv_folds),
             fits on x_train/y_train, and returns the best estimator plus
             search.cv_results_ and search.best_params_.
          2. Optuna (training_config/model_config["optuna"] == True): delegates to
             _run_optuna_search.
          3. Fixed params (neither flag set): builds a CatBoostClassifier from
             model_config["catboost_params"] (or hyper_cfg["default"]) and fits it
             directly on x_train/y_train, using x_validation/y_validation as the
             eval_set for early stopping if enabled. If cv_folds > 1, additionally
             runs a standalone catboost.cv purely for reporting purposes (attached as
             cv_results via get_catboost_cv_df).

        In all three branches, only x_train/y_train (and x_validation/y_validation for
        early stopping) are used — the held-out test set and production data are never
        seen during training.

        Args:
            x_train: Training features.
            y_train: Training target.
            x_validation: Validation features, used for early stopping.
            y_validation: Validation target.
            cat_features: List of categorical feature column names/indices.
            model_config: The 'model' section of train_config.json (cv, gridsearch,
                optuna, hyperparams, pos_class_label, catboost_params).
            training_config: The 'training' section of train_config.json (gridsearch,
                optuna, verbose, optuna_* settings).

        Returns:
            A ModelFitResult containing the trained model and, depending on strategy,
            CV results and/or selected hyperparameters.

        Raises:
            ValueError: If GridSearch is requested but no grid is configured, or if
                both GridSearch and Optuna are enabled at once (via
                _resolve_optimization_flags).
        """
        verbose = int(training_config.get("verbose", 100))
        cv_folds = int(model_config.get("cv", 1))
        configured_pos_label = str(model_config.get("pos_class_label", "1"))
        pos_label = self._resolve_positive_label(y_train, configured_pos_label)

        hyper_cfg = model_config.get("hyperparams", {}).get("catboost", {})
        gridsearch, optuna_enabled = self._resolve_optimization_flags(model_config, training_config)

        if gridsearch:
            if not hyper_cfg.get("gridsearch"):
                raise ValueError("GridSearch is enabled but no gridsearch parameter grid is configured.")
            base_params = dict(hyper_cfg.get("default", {}))
            # sklearn.clone can fail on some CatBoost list-style params.
            base_params.pop("custom_metric", None)
            base_params.pop("eval_set", None)
            base_params.setdefault("logging_level", "Silent")
            param_grid = {f"{key}": value for key, value in hyper_cfg.get("gridsearch", {}).items()}
            # CatBoost treats depth and max_depth as synonyms; keep only one.
            if "max_depth" in param_grid and "depth" not in param_grid:
                param_grid["depth"] = param_grid.pop("max_depth")
            if "depth" in param_grid:
                base_params.pop("max_depth", None)
                base_params.pop("depth", None)
            estimator = CatBoostClassifier(**base_params)
            scoring_names = hyper_cfg.get("scoring_eval_metrics", ["average_precision"])
            scoring = self._build_scoring(scoring_names, pos_label)
            refit = hyper_cfg.get("gridsearch_refit_score", "average_precision")
            if refit not in scoring:
                refit = next(iter(scoring))
            use_eval_set = bool(hyper_cfg.get("eval_set", True))

            cv = StratifiedKFold(n_splits=max(cv_folds, 2), shuffle=False, random_state=None)
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit,
                n_jobs=-1,
                cv=cv,
                verbose=0,
            )
            fit_kwargs: dict[str, Any] = {
                "cat_features": cat_features,
                "verbose": False,
            }
            if use_eval_set:
                fit_kwargs["eval_set"] = (x_validation, y_validation)
            search.fit(x_train, y_train, **fit_kwargs)

            best_model: CatBoostClassifier = search.best_estimator_
            cv_results = pd.DataFrame(search.cv_results_)
            return ModelFitResult(
                model=best_model,
                best_iteration=best_model.get_best_iteration(),
                best_score=best_model.get_best_score(),
                cv_results=cv_results,
                selected_hyperparameters=dict(search.best_params_),
            )

        if optuna_enabled:
            best_model, cv_results, best_params = self._run_optuna_search(
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                cat_features=cat_features,
                hyper_cfg=hyper_cfg,
                training_config=training_config,
                cv_folds=cv_folds,
                pos_label=pos_label,
            )
            return ModelFitResult(
                model=best_model,
                best_iteration=best_model.get_best_iteration(),
                best_score=best_model.get_best_score(),
                cv_results=cv_results,
                selected_hyperparameters=best_params,
            )

        params = dict(model_config.get("catboost_params", {}))
        if not params:
            params = dict(hyper_cfg.get("default", {}))
        self._sanitize_custom_metrics(params)
        params.setdefault("logging_level", "Silent")

        use_eval_set = bool(params.pop("eval_set", True))
        model = CatBoostClassifier(**params)

        fit_kwargs: dict[str, Any] = {
            "X": x_train,
            "y": y_train,
            "cat_features": cat_features,
            "verbose": False,
        }
        if use_eval_set:
            fit_kwargs["eval_set"] = (x_validation, y_validation)

        model.fit(**fit_kwargs)

        cv_df: pd.DataFrame | None = None
        if cv_folds > 1:
            pool = Pool(data=x_train, label=y_train, cat_features=cat_features)
            cv_params = dict(params)
            if not use_eval_set:
                cv_params["od_type"] = "IncToDec"
            cv_raw = catboost_cv(
                params=cv_params,
                pool=pool,
                nfold=cv_folds,
                stratified=True,
                as_pandas=True,
            )
            cv_df = self.get_catboost_cv_df(cv_raw, metric="test-PRAUC-mean")

        return ModelFitResult(
            model=model,
            best_iteration=model.get_best_iteration(),
            best_score=model.get_best_score(),
            cv_results=cv_df,
            selected_hyperparameters=None,
        )
