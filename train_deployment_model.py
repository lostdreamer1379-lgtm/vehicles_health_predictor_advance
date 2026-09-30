"""Build lightweight scikit-learn deployment artifacts."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from train_failure_modes import RAW_NUMERIC, DATASET_RENAME, TYPE_MAP, engineer_features


BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset" / "ai4i2020.csv"
FEATURES_PATH = BASE / "feature_names.pkl"
SCALER_PATH = BASE / "scaler.pkl"
MODEL_PATH = BASE / "best_model.pkl"
MODES_PATH = BASE / "failure_modes_model.pkl"
MODE_LABELS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_leaf_nodes=31,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=42,
    )


def main() -> None:
    data = pd.read_csv(DATASET).rename(columns=DATASET_RENAME)
    data["Type_enc"] = data["Type"].map(TYPE_MAP).fillna(1).astype(int)
    engineered = engineer_features(data)
    features = list(pickle.loads(FEATURES_PATH.read_bytes()))
    scaler = StandardScaler()
    X = scaler.fit_transform(engineered[features])

    y_failure = data["Machine failure"].astype(int)
    failure_model = make_model()
    failure_weights = np.where(y_failure.to_numpy() == 1, 3.0, 1.0)
    failure_model.fit(X, y_failure, sample_weight=failure_weights)

    mode_models = {}
    for label in MODE_LABELS:
        y_mode = data[label].astype(int)
        positives = int(y_mode.sum())
        negatives = len(y_mode) - positives
        mode_model = make_model()
        mode_weights = np.where(
            y_mode.to_numpy() == 1, negatives / max(positives, 1), 1.0
        )
        mode_model.fit(X, y_mode, sample_weight=mode_weights)
        mode_models[label] = mode_model

    with MODEL_PATH.open("wb") as handle:
        pickle.dump(failure_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with SCALER_PATH.open("wb") as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with MODES_PATH.open("wb") as handle:
        pickle.dump(
            {
                "models": mode_models,
                "labels": MODE_LABELS,
                "features": features,
                "training_rows": len(data),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print("Saved lightweight scikit-learn deployment artifacts.")


if __name__ == "__main__":
    main()
