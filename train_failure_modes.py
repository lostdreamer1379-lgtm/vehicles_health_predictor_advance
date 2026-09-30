"""Train model-based failure-mode predictors for the AI4I application."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset" / "ai4i2020.csv"
SCALER_PATH = BASE / "scaler.pkl"
FEATURES_PATH = BASE / "feature_names.pkl"
OUTPUT_PATH = BASE / "failure_modes_model.pkl"

RAW_NUMERIC = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
]
DATASET_RENAME = {
    "Air temperature [K]": "Air_temperature_K",
    "Process temperature [K]": "Process_temperature_K",
    "Rotational speed [rpm]": "Rotational_speed_rpm",
    "Torque [Nm]": "Torque_Nm",
    "Tool wear [min]": "Tool_wear_min",
}
MODE_LABELS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TYPE_MAP = {"L": 0, "M": 1, "H": 2}


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Temp_Differential"] = (
        data["Process_temperature_K"] - data["Air_temperature_K"]
    )
    data["Power_W"] = data["Torque_Nm"] * (
        data["Rotational_speed_rpm"] * 2 * np.pi / 60
    )
    data["Wear_Torque"] = data["Tool_wear_min"] * data["Torque_Nm"]
    data["Mechanical_Stress"] = (
        data["Rotational_speed_rpm"] * data["Torque_Nm"]
    )
    data["Thermal_Efficiency"] = (
        data["Process_temperature_K"] / data["Air_temperature_K"]
    )
    data["Wear_per_Torque"] = data["Tool_wear_min"] / (
        data["Torque_Nm"] + 0.001
    )
    data["Speed_Torque_Ratio"] = data["Rotational_speed_rpm"] / (
        data["Torque_Nm"] + 0.001
    )
    return data


def main() -> None:
    with DATASET.open("rb") as handle:
        data = pd.read_csv(handle).rename(columns=DATASET_RENAME)

    with SCALER_PATH.open("rb") as handle:
        scaler = pickle.load(handle)
    with FEATURES_PATH.open("rb") as handle:
        features = pickle.load(handle)

    data["Type_enc"] = data["Type"].map(TYPE_MAP).fillna(1).astype(int)
    engineered = engineer_features(data)
    X = scaler.transform(engineered[features])

    models = {}
    for label in MODE_LABELS:
        y = data[label].astype(int)
        positives = int(y.sum())
        negatives = len(y) - positives
        models[label] = HistGradientBoostingClassifier(
            max_iter=250,
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=42,
        )
        sample_weight = np.where(y.to_numpy() == 1, negatives / max(positives, 1), 1.0)
        models[label].fit(X, y, sample_weight=sample_weight)

    artifact = {
        "models": models,
        "labels": MODE_LABELS,
        "features": features,
        "training_rows": len(data),
    }
    with OUTPUT_PATH.open("wb") as handle:
        pickle.dump(artifact, handle)
    print(f"Saved {OUTPUT_PATH.name} with {len(models)} failure-mode models.")


if __name__ == "__main__":
    main()
