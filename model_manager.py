from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# from binance.enums import *
class ModelManager:
    """
    Wraps an sklearn pipeline (scaler + logistic regression) with clean APIs.
    """

    def __init__(self, predictor_cols: list[str]) -> None:
        self.predictor_cols = predictor_cols
        self.numeric_cols = predictor_cols[:-2]  # all but last 2 patterns
        self.pattern_cols = predictor_cols[-2:]  # PATT_3OUT, PATT_CMB
        self.pipeline: Optional[Pipeline] = None
        self.last_test_accuracy: Optional[float] = None

    def _build_pipeline(self) -> Pipeline:
        prep = ColumnTransformer(
            transformers=[("scale", StandardScaler(), self.numeric_cols)],
            remainder="passthrough",  # patterns pass through
        )
        pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=200))])
        return pipe

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Fit on train split, evaluate on test split. Returns test accuracy.
        """
        # Ensure both classes are present; otherwise, logistic regression will fail.
        if y.nunique() < 2:
            raise RuntimeError("Target has fewer than 2 classes; wait for more data.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1, stratify=y
        )
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.last_test_accuracy = float(acc)
        return self.last_test_accuracy

    def predict_proba_up(self, X_one: pd.DataFrame) -> float:
        """
        Returns P(up) for a single-row features DataFrame.
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        proba = self.pipeline.predict_proba(X_one)[0][1]
        return float(proba)
