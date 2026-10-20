"""RandomForest / tabular trainer (scikit-learn) [doc 08 §2, doc 14 §3.4].

Trains on the tabular CSV that the verification manifest points at, emits real
train/val metrics and saves the fitted model into the run artifacts. No
pretrained checkpoint is used — random forests train from scratch by design.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any

import joblib  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, mean_squared_error  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OrdinalEncoder  # type: ignore[import-untyped]

from app.training.base import Issue, RunContext, TrainerPlugin, TrainResult

_NUMERIC_HINTS = {"float", "int", "integer", "number", "numeric"}


class RandomForestTrainer(TrainerPlugin):
    family = "random_forest"
    supported_tasks = ["tabular_classification", "tabular_regression"]
    supported_formats = ["tabular"]
    description = "Classic tabular learner; trains on CPU in seconds. No pretrained weights."
    needs_gpu = False
    hparam_schema = {
        "type": "object",
        "properties": {
            "n_estimators": {"type": "integer", "minimum": 5, "default": 100},
            "max_depth": {"type": ["integer", "null"], "default": None},
        },
    }
    default_hparams = {"n_estimators": 100, "max_depth": None, "dev_preset": False}

    # -- state --------------------------------------------------------------

    def __init__(self) -> None:
        self._features: list[str] = []
        self._target: str = ""
        self._task: str = "tabular_classification"

    # -- plugin interface ---------------------------------------------------

    def validate_config(self, config: dict[str, Any], facts: dict[str, Any]) -> list[Issue]:
        del facts
        issues: list[Issue] = []
        n_estimators = int(config.get("n_estimators", 100))
        if n_estimators < 5:
            issues.append(Issue("n_estimators", "must be at least 5"))
        if config.get("dev_preset") and n_estimators > 50:
            issues.append(Issue("n_estimators", "dev_preset caps n_estimators at 50", "warn"))
        return issues

    def prepare(self, ctx: RunContext) -> None:
        csv_file = next(ctx.dataset_root.rglob("*.csv"), None)
        if csv_file is None:
            raise ValueError("tabular dataset has no CSV file")
        schema = ctx.facts.get("tabular_schema") or {}
        target = schema.get("target") or "target"
        lines = csv_file.read_text(encoding="utf-8", errors="replace").splitlines()
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        fieldnames = list(reader.fieldnames or [])
        if target not in fieldnames:
            raise ValueError(f"target column {target!r} not in CSV columns {fieldnames}")
        self._features = [name for name in fieldnames if name != target]
        self._target = target
        self._task = ctx.task_type or "tabular_classification"
        self._classifier = self._task == "tabular_classification"

        rows = list(reader)
        x_values = [
            [(record.get(feature) or "").strip() for feature in self._features] for record in rows
        ]
        y_values = [(record.get(target) or "").strip() for record in rows]

        numeric = [
            feature
            for feature in self._features
            if feature in schema.get("numeric_columns", [])  # fallback below
        ]
        if not numeric:
            numeric = [
                feature
                for feature in self._features
                if _looks_numeric(x_values, self._features.index(feature))
            ]
        categorical = [feature for feature in self._features if feature not in numeric]
        # ColumnTransformer accepts string names only for DataFrames; we feed a
        # list of rows, so use positional indices.
        numeric_idx = [self._features.index(feature) for feature in numeric]
        categorical_idx = [self._features.index(feature) for feature in categorical]

        numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
        transformer = ColumnTransformer(
            [
                ("num", numeric_pipeline, numeric_idx),
                ("cat", categorical_pipeline, categorical_idx),
            ]
        )
        self._pipeline = Pipeline(
            [
                ("transform", transformer),
                (
                    "model",
                    (
                        RandomForestClassifier(
                            n_estimators=self._n_estimators(ctx), random_state=ctx.seed
                        )
                        if self._classifier
                        else RandomForestRegressor(
                            n_estimators=self._n_estimators(ctx), random_state=ctx.seed
                        )
                    ),
                ),
            ]
        )
        self._x = x_values
        self._y = y_values

    def _n_estimators(self, ctx: RunContext) -> int:
        n = int(ctx.config.get("n_estimators", 100))
        return min(n, 20) if ctx.config.get("dev_preset") else n

    def train(self, ctx: RunContext) -> TrainResult:
        if ctx.should_stop():
            return TrainResult(best={}, epochs=0, note="stopped before training")
        stratify = self._y if self._classifier else None
        try:
            x_train, x_val, y_train, y_val = train_test_split(
                self._x, self._y, test_size=0.2, random_state=ctx.seed, stratify=stratify
            )
        except ValueError:  # a class with a single sample cannot be stratified
            x_train, x_val, y_train, y_val = train_test_split(
                self._x, self._y, test_size=0.2, random_state=ctx.seed
            )
        ctx.emit_log("INFO", f"fitting {self.family} on {len(x_train)} rows")
        self._pipeline.fit(x_train, y_train)

        train_pred = self._pipeline.predict(x_train)
        val_pred = self._pipeline.predict(x_val)
        if self._classifier:
            train_metric = float(accuracy_score(y_train, train_pred))
            val_metric = float(accuracy_score(y_val, val_pred))
            metric_name = "accuracy"
        else:
            train_metric = float(math.sqrt(mean_squared_error(y_train, train_pred)))
            val_metric = float(math.sqrt(mean_squared_error(y_val, val_pred)))
            metric_name = "rmse"
        self._best_metric = val_metric
        ctx.emit_metric(f"train/{metric_name}", train_metric, 1, 1, "train")
        ctx.emit_metric(f"val/{metric_name}", val_metric, 1, 1, "val")
        ctx.emit_log(
            "INFO", f"train/{metric_name}={train_metric:.4f} val/{metric_name}={val_metric:.4f}"
        )

        model_path = ctx.output_dir / "model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, model_path)
        ctx.emit_event("artifact", {"name": "model.joblib", "path": str(model_path)})
        return TrainResult(
            best={"metric": f"val/{metric_name}", "value": round(val_metric, 4), "epoch": 1},
            epochs=1,
            note=f"{metric_name}={val_metric:.4f}",
        )


def _looks_numeric(rows: list[list[str]], index: int) -> bool:
    sample = [row[index] for row in rows if row[index] != ""][:20]
    if not sample:
        return True
    for value in sample:
        try:
            float(value)
        except ValueError:
            return False
    return True
