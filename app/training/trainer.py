from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

from app.training.preprocessed_data import PreprocessedData


@dataclass(frozen=True)
class TrainingResult:
    model: CatBoostClassifier
    best_iteration: int | None
    best_score: dict[str, Any]


class CatBoostTrainer:
    def __init__(
        self,
        catboost_config: dict[str, Any],
        training_config: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.catboost_config = catboost_config
        self.training_config = training_config
        self.logger = logger

    def train(self, data: PreprocessedData) -> TrainingResult:
        self.logger.info("Training Started")
        categorical_feature_indices = self._categorical_feature_indices(
            data.X_train,
            data.categorical_features,
        )

        model_params = dict(self.catboost_config)
        use_eval_set = bool(model_params.pop("eval_set", True))
        verbose = int(self.training_config.get("verbose", 100))

        model = CatBoostClassifier(**model_params)
        fit_kwargs: dict[str, Any] = {
            "X": data.X_train,
            "y": data.y_train,
            "cat_features": categorical_feature_indices,
            "verbose": verbose,
        }

        if use_eval_set:
            fit_kwargs["eval_set"] = (data.X_validation, data.y_validation)

        model.fit(**fit_kwargs)

        best_iteration = model.get_best_iteration()
        best_score = model.get_best_score()
        self.logger.info("Best Iteration: %s", best_iteration)
        self.logger.info("Best Score: %s", best_score)
        self.logger.info("Training Completed")

        return TrainingResult(
            model=model,
            best_iteration=best_iteration,
            best_score=best_score,
        )

    @staticmethod
    def _categorical_feature_indices(
        dataframe: pd.DataFrame,
        categorical_features: list[str],
    ) -> list[int]:
        return [dataframe.columns.get_loc(column) for column in categorical_features]
