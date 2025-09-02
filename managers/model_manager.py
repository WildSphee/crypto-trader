from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import tensorflow as tf

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore
try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore
try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

# Deep learning (sequence models) via SciKeras + Keras
try:
    from scikeras.wrappers import KerasClassifier  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
except Exception:  # pragma: no cover
    KerasClassifier = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore

ModelName = Literal[
    "logreg",
    "sgdlog",
    "rf",
    "hgb",
    "linsvc",
    "bilstm",
    "gru_lstm",
    "hybrid_transformer",
    "voting_soft",
    "stacking",
    "metalabel",
]
tf.get_logger().setLevel("ERROR")


def _needs_scaling(model_name: ModelName) -> bool:
    return model_name in {"logreg", "sgdlog", "linsvc", "voting_soft", "stacking"}


class GARCHVolatilityTransformer(TransformerMixin, BaseEstimator):
    """
    Appends a conditional volatility (GARCH(p,q)) feature from a returns column.
    Produces a new numeric column `out_col` (default 'garch_vol').
    """

    def __init__(
        self, returns_col: str, out_col: str = "garch_vol", p: int = 1, q: int = 1
    ):
        self.returns_col = returns_col
        self.out_col = out_col
        self.p = p
        self.q = q

    def fit(self, X: pd.DataFrame, y=None):
        # nothing to fit persistently
        return self

    def transform(self, X: pd.DataFrame):
        Xc = X.copy()
        try:
            from arch import arch_model
        except Exception as e:
            raise ImportError(
                "Install 'arch' to use GARCHVolatilityTransformer: pip install arch"
            ) from e

        s = pd.Series(Xc[self.returns_col].astype(float))
        keep = s.dropna()
        if keep.empty:
            Xc[self.out_col] = np.nan
            return Xc

        am = arch_model(keep.values, vol="GARCH", p=self.p, q=self.q, rescale=False)
        res = am.fit(disp="off")
        vol = pd.Series(res.conditional_volatility, index=keep.index)
        # align back to original index
        aligned = vol.reindex(s.index).ffill().bfill()
        Xc[self.out_col] = aligned.values
        return Xc


class SequenceMaker(TransformerMixin, BaseEstimator):
    """
    If X is already a 3D numpy array (n, t, d) it passes through.
    Otherwise, calls user-provided function to create (n, t, d).
    """

    def __init__(self, fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray) and X.ndim == 3:
            return X
        if isinstance(X, pd.DataFrame) and self.fn is not None:
            arr = self.fn(X)
            if not (isinstance(arr, np.ndarray) and arr.ndim == 3):
                raise ValueError(
                    "sequence_maker must return a 3D numpy array (n, t, d)."
                )
            return arr
        raise RuntimeError(
            "For sequence models, provide X as a 3D numpy array or pass sequence_maker=callable."
        )


class MetaLabelingClassifier(BaseEstimator, ClassifierMixin):
    """
    base: trains on all X,y and outputs p_base.
    meta: trains only on cases base deems 'actionable' (p_base >= threshold).
    final probability := p_base * p_meta (others get p_meta=0.5 by default).
    """

    def __init__(
        self, base: BaseEstimator, meta: BaseEstimator, threshold: float = 0.5
    ):
        self.base = base
        self.meta = meta
        self.threshold = threshold

    def fit(self, X, y):
        self.base.fit(X, y)
        p = self.base.predict_proba(X)[:, 1]
        mask = p >= self.threshold
        # support DataFrame or ndarray
        if isinstance(X, pd.DataFrame):
            X_mask = X.loc[mask]
        else:
            X_mask = X[mask]
        y_mask = (
            y[mask] if isinstance(y, (pd.Series, np.ndarray)) else np.array(y)[mask]
        )

        if len(y_mask) == 0:
            # fallback: train meta on all data
            self.meta.fit(X, y)
        else:
            self.meta.fit(X_mask, y_mask)
        return self

    def predict_proba(self, X):
        pb = self.base.predict_proba(X)[:, 1]
        meta_p = np.full_like(pb, 0.5, dtype=float)
        mask = pb >= self.threshold
        if mask.any():
            if isinstance(X, pd.DataFrame):
                Xm = X.loc[mask]
            else:
                Xm = X[mask]
            meta_p[mask] = self.meta.predict_proba(Xm)[:, 1]
        final = pb * meta_p
        return np.vstack([1 - final, final]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _ensure_keras():
    """Keras builders for sequence models"""
    if KerasClassifier is None or keras is None or layers is None:
        raise ImportError(
            "Sequence models require TensorFlow + SciKeras. Install:\n"
            "  pip install tensorflow scikeras"
        )


def _bilstm_builder(meta):
    _ensure_keras()
    t, d = meta["X_shape_"][1], meta["X_shape_"][2]
    inp = keras.Input(shape=(t, d))
    x = layers.Bidirectional(layers.LSTM(64))(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def _gru_lstm_builder(meta):
    _ensure_keras()
    t, d = meta["X_shape_"][1], meta["X_shape_"][2]
    inp = keras.Input(shape=(t, d))
    x = layers.GRU(64, return_sequences=True)(inp)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def _hybrid_transformer_builder(meta):
    _ensure_keras()
    t, d = meta["X_shape_"][1], meta["X_shape_"][2]
    inp = keras.Input(shape=(t, d))
    # Transformer-like encoder block
    x = layers.LayerNormalization()(inp)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=max(8, d // 2))(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ffn = keras.Sequential([layers.Dense(128, activation="relu"), layers.Dense(d)])
    x = layers.Add()([x, ffn(x)])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


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
        # NEW: sequence / deep-learning knobs
        input_kind: Literal["tabular", "sequence"] = "tabular",
        sequence_maker: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        nn_epochs: int = 20,
        nn_batch_size: int = 32,
        metalabel_base: ModelName = "hgb",
        metalabel_meta: ModelName = "logreg",
        metalabel_threshold: float = 0.5,
        garch_returns_col: Optional[str] = None,
        garch_out_col: str = "garch_vol",
        garch_pq: Tuple[int, int] = (1, 1),
        voting_members: Optional[List[ModelName]] = None,
        stacking_members: Optional[List[ModelName]] = None,
    ) -> None:
        self.predictor_cols = predictor_cols
        self.numeric_cols = predictor_cols[:-2]  # original convention
        self.pattern_cols = predictor_cols[-2:]  # PATT_3OUT, PATT_CMB
        self.model_name = model_name
        self.class_weight = class_weight
        self.calibrate_svc = calibrate_svc
        self.random_state = random_state

        # sequence
        self.input_kind = input_kind
        self.sequence_maker = sequence_maker
        self.nn_epochs = nn_epochs
        self.nn_batch_size = nn_batch_size

        # meta-labeling
        self.metalabel_base = metalabel_base
        self.metalabel_meta = metalabel_meta
        self.metalabel_threshold = metalabel_threshold

        # garch
        self.garch_returns_col = garch_returns_col
        self.garch_out_col = garch_out_col
        self.garch_pq = garch_pq

        # ensembles
        self.voting_members = voting_members or ["logreg", "rf", "hgb"]
        self.stacking_members = stacking_members or ["logreg", "rf", "hgb"]

        self.pipeline: Optional[Pipeline] = None
        self.last_test_accuracy: Optional[float] = None

    # -------------------- builders --------------------
    def _simple_estimator(self, name: ModelName):
        if name == "logreg":
            return LogisticRegression(
                max_iter=5000,
                solver="saga",
                C=0.5,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        if name == "sgdlog":
            base = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            return CalibratedClassifierCV(base, cv=3)
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        if name == "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_depth=None,
                max_bins=255,
                random_state=self.random_state,
            )
        if name == "linsvc":
            svc = LinearSVC(
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            # Calibrate to get predict_proba
            return CalibratedClassifierCV(svc, cv=3)

        if name == "bilstm":
            _ensure_keras()
            return KerasClassifier(
                model=_bilstm_builder,
                epochs=self.nn_epochs,
                batch_size=self.nn_batch_size,
                verbose=0,
            )
        if name == "gru_lstm":
            _ensure_keras()
            return KerasClassifier(
                model=_gru_lstm_builder,
                epochs=self.nn_epochs,
                batch_size=self.nn_batch_size,
                verbose=0,
            )
        if name == "hybrid_transformer":
            _ensure_keras()
            return KerasClassifier(
                model=_hybrid_transformer_builder,
                epochs=self.nn_epochs,
                batch_size=self.nn_batch_size,
                verbose=0,
            )

        raise ValueError(f"Unknown estimator name: {name}")

    def _build_estimator(self):
        # ensembles & meta-labeling are composed here
        if self.model_name == "voting_soft":
            members = [(m, self._simple_estimator(m)) for m in self.voting_members]
            return VotingClassifier(estimators=members, voting="soft")

        if self.model_name == "stacking":
            members = [(m, self._simple_estimator(m)) for m in self.stacking_members]
            final = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            return StackingClassifier(
                estimators=members, final_estimator=final, passthrough=False
            )

        if self.model_name == "metalabel":
            base = self._simple_estimator(self.metalabel_base)
            meta = self._simple_estimator(self.metalabel_meta)
            return MetaLabelingClassifier(
                base=base, meta=meta, threshold=self.metalabel_threshold
            )

        # regular single estimators, including deep ones
        return self._simple_estimator(self.model_name)

    def _build_pipeline(self) -> Pipeline:
        steps: List[Tuple[str, object]] = []

        # Optional GARCH feature for tabular pipelines
        if self.input_kind == "tabular" and self.garch_returns_col is not None:
            steps.append(
                (
                    "garch",
                    GARCHVolatilityTransformer(
                        returns_col=self.garch_returns_col,
                        out_col=self.garch_out_col,
                        p=self.garch_pq[0],
                        q=self.garch_pq[1],
                    ),
                )
            )
            # treat the new column as numeric for scaling if we scale
            if self.garch_out_col not in self.numeric_cols:
                self.numeric_cols = list(self.numeric_cols) + [self.garch_out_col]

        # Preprocessing
        if self.input_kind == "sequence":
            # sequences are handled by the deep models themselves; just ensure shape is (n,t,d)
            steps.append(("to_seq", SequenceMaker(self.sequence_maker)))
            prep = "passthrough"
        else:
            if _needs_scaling(self.model_name):
                prep = ColumnTransformer(
                    transformers=[("scale", StandardScaler(), self.numeric_cols)],
                    remainder="passthrough",
                )
            else:
                prep = "passthrough"
            steps.append(("prep", prep))

        # Estimator
        steps.append(("clf", self._build_estimator()))
        return Pipeline(steps)

    def train(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> float:
        # classification sanity
        if pd.Series(y).nunique() < 2:
            raise RuntimeError("Target has fewer than 2 classes; wait for more data.")

        stratify = y if isinstance(y, (pd.Series, np.ndarray)) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=stratify
        )
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.last_test_accuracy = float(acc)
        return self.last_test_accuracy

    def predict_proba_up(self, X_one: Union[pd.DataFrame, np.ndarray]) -> float:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        proba = self.pipeline.predict_proba(X_one)[0][1]
        return float(proba)
