from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy_features import PolicyFeatureConfig, build_policy_features


class SupervisedPolicy:
    """Load and run a trained supervised policy artifact."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        linear_path = model_dir / "expert_policy_linear.json"
        model_path_joblib = model_dir / "expert_policy.joblib"
        model_path_pickle = model_dir / "expert_policy.pkl"

        self.linear_weights = None
        self.linear_bias = None
        self.feature_columns = None
        self.model = None

        if linear_path.exists():
            payload = json.loads(linear_path.read_text())
            self.feature_columns = payload["feature_columns"]
            self.linear_weights = np.asarray(payload["weights"], dtype=float)
            self.linear_bias = np.asarray(payload["bias"], dtype=float)
        else:
            # Optional fallback for environments that do have joblib/scikit-learn.
            try:
                import pickle

                if model_path_pickle.exists():
                    with model_path_pickle.open("rb") as model_file:
                        self.model = pickle.load(model_file)
                elif model_path_joblib.exists():
                    import joblib  # type: ignore

                    self.model = joblib.load(model_path_joblib)
            except Exception as error:
                raise RuntimeError(
                    f"Could not load supervised policy from {model_dir}. "
                    f"Expected {linear_path.name}, {model_path_joblib.name}, or {model_path_pickle.name}."
                ) from error

            metadata = json.loads((model_dir / "metadata.json").read_text())
            self.feature_columns = metadata["feature_columns"]

    def predict(self, feature_dict: dict) -> tuple[float, float]:
        feature_row = [float(feature_dict[k]) for k in self.feature_columns]
        if self.linear_weights is not None and self.linear_bias is not None:
            pred = np.asarray(feature_row, dtype=float) @ self.linear_weights + self.linear_bias
            return float(pred[0]), float(pred[1])

        pred = self.model.predict(np.array([feature_row]))[0]
        return float(pred[0]), float(pred[1])

    def predict_from_state(
        self,
        trajectory,
        X: float,
        Y: float,
        xdot: float,
        ydot: float,
        psi: float,
        psidot: float,
        battery_soc: float,
        config: PolicyFeatureConfig,
        current_node_index: int | None = None,
    ) -> tuple[float, float]:
        feature_dict = build_policy_features(
            trajectory=trajectory,
            X=X,
            Y=Y,
            xdot=xdot,
            ydot=ydot,
            psi=psi,
            psidot=psidot,
            battery_soc=battery_soc,
            config=config,
            current_node_index=current_node_index,
        )
        return self.predict(feature_dict)
