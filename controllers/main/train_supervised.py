from __future__ import annotations

import argparse
import json
import glob
import pickle
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "speed_norm",
    "psidot",
    "cross_track_error_norm",
    "heading_error",
    "track_heading_error",
    "battery_soc_norm",
    "track_progress",
    "v_target_norm",
    "corner_feasible_speed_norm",
    "preview_speed_limit_norm",
    "max_heading_change",
    "distance_to_turn_start_norm",
    "deploy_track_factor",
    "soc_factor",
    "brake_required",
]

TARGET_COLUMNS = ["F_cmd", "delta_cmd"]


def load_dataset(data_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(data_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No expert logs found for pattern: {data_glob}")

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df["source_file"] = str(file_path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def split_by_session(df: pd.DataFrame, test_size: float, seed: int):
    sessions = sorted(df["session_id"].dropna().unique().tolist())
    if len(sessions) < 2:
        raise ValueError("Need at least 2 sessions (laps/runs) to create train/val split.")

    train_sessions, val_sessions = train_test_split(
        sessions,
        test_size=test_size,
        random_state=seed,
    )
    train_df = df[df["session_id"].isin(train_sessions)].copy()
    val_df = df[df["session_id"].isin(val_sessions)].copy()
    return train_df, val_df


def fit_linear_policy(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    ones = np.ones((x_train.shape[0], 1), dtype=float)
    design = np.hstack([x_train, ones])
    coeffs, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
    weights = coeffs[:-1, :]
    bias = coeffs[-1, :]
    return {
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "weights": weights.tolist(),
        "bias": bias.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised controller from expert lap logs")
    parser.add_argument(
        "--data-glob",
        default="data/expert/**/*.csv",
        help="Glob for expert CSV logs relative to controllers/main",
    )
    parser.add_argument(
        "--model-dir",
        default="models/supervised",
        help="Directory to save model artifacts",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.data_glob)
    required_cols = FEATURE_COLUMNS + TARGET_COLUMNS + ["session_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df.dropna(subset=required_cols).copy()
    train_df, val_df = split_by_session(df, test_size=args.test_size, seed=args.seed)

    x_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMNS].values
    x_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df[TARGET_COLUMNS].values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 128),
                    activation="relu",
                    max_iter=300,
                    learning_rate_init=1e-3,
                    early_stopping=True,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    f_mae = mean_absolute_error(y_val[:, 0], y_pred[:, 0])
    delta_mae = mean_absolute_error(y_val[:, 1], y_pred[:, 1])
    f_r2 = r2_score(y_val[:, 0], y_pred[:, 0])
    delta_r2 = r2_score(y_val[:, 1], y_pred[:, 1])

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "expert_policy.joblib"
    pickle_path = model_dir / "expert_policy.pkl"
    linear_path = model_dir / "expert_policy_linear.json"
    metadata_path = model_dir / "metadata.json"

    joblib.dump(model, model_path)
    with pickle_path.open("wb") as pickle_file:
        pickle.dump(model, pickle_file)
    linear_policy = fit_linear_policy(x_train, y_train)
    linear_path.write_text(json.dumps(linear_policy, indent=2))
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train_df)),
        "n_rows_val": int(len(val_df)),
        "metrics": {
            "F_mae": float(f_mae),
            "delta_mae": float(delta_mae),
            "F_r2": float(f_r2),
            "delta_r2": float(delta_r2),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Training complete.")
    print(f"Model: {model_path}")
    print(f"Pickle: {pickle_path}")
    print(f"Linear policy: {linear_path}")
    print(f"Metadata: {metadata_path}")
    print("Validation metrics:")
    print(f"  F MAE: {f_mae:.3f}")
    print(f"  delta MAE: {delta_mae:.4f}")
    print(f"  F R2: {f_r2:.3f}")
    print(f"  delta R2: {delta_r2:.3f}")


if __name__ == "__main__":
    main()
