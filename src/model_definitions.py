from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_OPTIMIZER = "lbfgs"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    target_type: str


def build_classification_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            solver=NUMERIC_OPTIMIZER,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            target_type="classification",
        ),
        ModelSpec(
            name="random_forest_classifier",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            target_type="classification",
        ),
        ModelSpec(
            name="extra_trees_classifier",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            target_type="classification",
        ),
    ]


def build_regression_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="ridge",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            target_type="regression",
        ),
        ModelSpec(
            name="hist_gb",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=200,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            target_type="regression",
        ),
        ModelSpec(
            name="extra_trees_regressor",
            estimator=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            target_type="regression",
        ),
    ]
