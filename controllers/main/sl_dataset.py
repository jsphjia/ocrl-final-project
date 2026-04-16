from __future__ import annotations

import csv
import time
import uuid
from pathlib import Path


class ExpertDataLogger:
    """Stream expert controller features/actions to CSV for supervised learning."""

    FIELDNAMES = [
        "track_name",
        "session_id",
        "step_idx",
        "sim_time_s",
        "x",
        "y",
        "xdot",
        "ydot",
        "speed",
        "speed_norm",
        "psi",
        "psidot",
        "cross_track_error",
        "cross_track_error_norm",
        "heading_error",
        "track_heading_error",
        "battery_soc",
        "battery_soc_norm",
        "track_progress",
        "v_target",
        "v_target_norm",
        "corner_feasible_speed",
        "corner_feasible_speed_norm",
        "preview_speed_limit",
        "preview_speed_limit_norm",
        "max_heading_change",
        "distance_to_turn_start",
        "distance_to_turn_start_norm",
        "deploy_track_factor",
        "soc_factor",
        "brake_required",
        "throttle_fraction",
        "brake_fraction",
        "F_cmd",
        "delta_cmd",
        "delT",
    ]

    def __init__(self, output_dir: Path, track_name: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.track_name = track_name or self.output_dir.name
        self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.path = self.output_dir / f"expert_{self.session_id}.csv"

        self.step_idx = 0
        self.sim_time_s = 0.0

        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def log(self, row: dict) -> None:
        del_t = float(row.get("delT", 0.0))
        self.sim_time_s += max(del_t, 0.0)

        record = {
            "track_name": self.track_name,
            "session_id": self.session_id,
            "step_idx": self.step_idx,
            "sim_time_s": self.sim_time_s,
        }
        for key in self.FIELDNAMES:
            if key in record:
                continue
            record[key] = row.get(key)

        self._writer.writerow(record)
        self.step_idx += 1

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()
