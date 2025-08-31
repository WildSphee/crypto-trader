# model_manager.py
from __future__ import annotations

from typing import List, Literal, Optional

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

# Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ModelName = Literal["logreg", "sgdlog", "rf", "hgb", "linsvc"]


def _needs_scaling(model_name: ModelName) -> bool:
    # trees don't; linear/svm do
    return model_name in {"logreg", "sgdlog", "linsvc"}


class ModelManager:
    """
    Flexible sklearn pipeline manager with multiple back-ends.
    All models expose predict_proba via native method or calibration.
    """

    def __init__(
        self,
        predictor_cols: List[str],
        model_name: ModelName = "logreg",
        class_weight: Optional[str] = None,  # "balanced" or None
        calibrate_svc: bool = True,  # for LinearSVC
        random_state: int = 1,
    ) -> None:
        self.predictor_cols = predictor_cols
        self.numeric_cols = predictor_cols[:-2]  # all but last 2 patterns
        self.pattern_cols = predictor_cols[-2:]  # PATT_3OUT, PATT_CMB
        self.model_name = model_name
        self.class_weight = class_weight
        self.calibrate_svc = calibrate_svc
        self.random_state = random_state

        self.pipeline: Optional[Pipeline] = None
        self.last_test_accuracy: Optional[float] = None

    def _build_estimator(self):
        if self.model_name == "logreg":
            return LogisticRegression(
                max_iter=400,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        if self.model_name == "sgdlog":
            # logistic loss with L2, probability via CalibratedClassifierCV
            base = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            return CalibratedClassifierCV(base, cv=3)
        if self.model_name == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        if self.model_name == "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_depth=None,
                max_bins=255,
                random_state=self.random_state,
            )
        if self.model_name == "linsvc":
            svc = LinearSVC(
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            if self.calibrate_svc:
                return CalibratedClassifierCV(svc, cv=3)
            # if not calibrated, we can’t do predict_proba; calibration recommended
            return CalibratedClassifierCV(svc, cv=3)

        raise ValueError(f"Unknown model_name: {self.model_name}")

    def _build_pipeline(self) -> Pipeline:
        if _needs_scaling(self.model_name):
            prep = ColumnTransformer(
                transformers=[("scale", StandardScaler(), self.numeric_cols)],
                remainder="passthrough",
            )
        else:
            prep = "passthrough"

        return Pipeline(
            [
                ("prep", prep),
                ("clf", self._build_estimator()),
            ]
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        if y.nunique() < 2:
            raise RuntimeError("Target has fewer than 2 classes; wait for more data.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.last_test_accuracy = float(acc)
        return self.last_test_accuracy

    def predict_proba_up(self, X_one: pd.DataFrame) -> float:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        proba = self.pipeline.predict_proba(X_one)[0][1]
        return float(proba)
