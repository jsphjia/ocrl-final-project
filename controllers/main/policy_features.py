from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from util import closestNode, wrapToPi


@dataclass(frozen=True)
class PolicyFeatureConfig:
    max_speed_mps: float
    preview_offsets: tuple[int, ...]
    preview_segment_nodes: int
    min_turn_speed_mps: float
    a_lat_limit: float
    approach_decel: float
    preview_safety_margin_m: float
    approach_speed_buffer: float
    first_corner_nodes: int
    first_corner_ramp_nodes: int
    first_corner_speed_cap_mps: float
    straight_heading_ref: float
    straight_soc_threshold: float
    battery_soc_max: float
    deploy_heading_margin: float
    deploy_crosstrack_margin: float
    turn_detect_heading_threshold: float


def _cumulative_track_lengths(trajectory: np.ndarray) -> np.ndarray:
    segment = np.diff(trajectory, axis=0)
    segment_lengths = np.sqrt(np.sum(segment**2, axis=1))
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _heading_from_nodes(trajectory: np.ndarray, current_node_index: int, start_offset: int, segment_nodes: int) -> float:
    n_points = len(trajectory)
    i0 = int((current_node_index + start_offset) % n_points)
    i1 = int((i0 + segment_nodes) % n_points)
    dx = trajectory[i1, 0] - trajectory[i0, 0]
    dy = trajectory[i1, 1] - trajectory[i0, 1]
    return float(np.arctan2(dy, dx))


def build_policy_features(
    trajectory: np.ndarray,
    X: float,
    Y: float,
    xdot: float,
    ydot: float,
    psi: float,
    psidot: float,
    battery_soc: float,
    config: PolicyFeatureConfig,
    current_node_index: int | None = None,
) -> dict[str, float]:
    trajectory = np.asarray(trajectory)
    if current_node_index is None:
        _, current_node_index = closestNode(X, Y, trajectory)

    speed = float(np.sqrt(xdot**2 + ydot**2))
    cumulative_lengths = _cumulative_track_lengths(trajectory)
    track_length = max(float(cumulative_lengths[-1]), 1e-6)
    track_progress = float(cumulative_lengths[current_node_index] / track_length)

    base_heading = _heading_from_nodes(trajectory, current_node_index, 0, config.preview_segment_nodes)
    target_offset = int(config.preview_offsets[-1])
    max_heading_change = 0.0
    first_turn_offset = None
    for offset in config.preview_offsets:
        heading = _heading_from_nodes(trajectory, current_node_index, offset, config.preview_segment_nodes)
        heading_delta = abs(wrapToPi(heading - base_heading))
        if first_turn_offset is None and heading_delta >= config.turn_detect_heading_threshold:
            first_turn_offset = offset
        if heading_delta > max_heading_change:
            max_heading_change = heading_delta
            target_offset = offset

    if first_turn_offset is None:
        first_turn_offset = target_offset

    estimated_curve_radius = 8.0 + 45.0 / (max_heading_change + 0.08)
    curve_speed_limit = np.sqrt(config.a_lat_limit * estimated_curve_radius)
    curve_speed_limit = float(np.clip(curve_speed_limit, config.min_turn_speed_mps, config.max_speed_mps))

    distance_to_turn_start = max(
        first_turn_offset * max(track_length / max(len(trajectory), 1), 0.2) - config.preview_safety_margin_m,
        0.0,
    )
    preview_speed_limit = np.sqrt(
        max(curve_speed_limit**2 + 2.0 * config.approach_decel * distance_to_turn_start, 0.0)
    )
    preview_speed_limit = float(
        np.clip(
            preview_speed_limit - config.approach_speed_buffer,
            config.min_turn_speed_mps,
            config.max_speed_mps,
        )
    )

    target_node_index = int((current_node_index + max(config.preview_offsets)) % len(trajectory))
    target_angle = float(np.arctan2(
        trajectory[target_node_index, 1] - Y,
        trajectory[target_node_index, 0] - X,
    ))
    heading_error = float(wrapToPi(psi - target_angle))
    track_heading_error = float(wrapToPi(psi - base_heading))

    cross_track_error, _ = closestNode(X, Y, trajectory)
    cross_track_error = float(cross_track_error)

    heading_term = min(abs(heading_error) / 1.2, 1.0)
    turn_speed_limit = config.max_speed_mps * (1.0 - 0.88 * heading_term)
    turn_speed_limit -= 2.2 * abs(psidot)
    turn_speed_limit -= 1.5 * abs(cross_track_error)
    turn_speed_limit = float(np.clip(turn_speed_limit, config.min_turn_speed_mps, config.max_speed_mps))

    v_target = float(np.clip(min(turn_speed_limit, preview_speed_limit), config.min_turn_speed_mps, config.max_speed_mps))
    corner_feasible_speed = float(np.clip(preview_speed_limit, config.min_turn_speed_mps, config.max_speed_mps))

    if current_node_index < config.first_corner_ramp_nodes:
        blend = np.clip(
            (current_node_index - config.first_corner_nodes)
            / max(config.first_corner_ramp_nodes - config.first_corner_nodes, 1),
            0.0,
            1.0,
        )
        early_corner_cap = config.first_corner_speed_cap_mps + blend * (config.max_speed_mps - config.first_corner_speed_cap_mps)
        v_target = min(v_target, early_corner_cap)
        corner_feasible_speed = min(corner_feasible_speed, early_corner_cap)

    straight_factor = float(
        np.clip(
            (config.straight_heading_ref - max_heading_change) / max(config.straight_heading_ref, 1e-6),
            0.0,
            1.0,
        )
    )
    speed_headroom = float(np.clip((v_target - speed) / max(config.max_speed_mps, 1e-6), 0.0, 1.0))
    heading_stability = float(np.clip(1.0 - abs(heading_error) / max(config.deploy_heading_margin, 1e-6), 0.0, 1.0))
    track_stability = float(np.clip(1.0 - abs(cross_track_error) / max(config.deploy_crosstrack_margin, 1e-6), 0.0, 1.0))
    deploy_track_factor = max(straight_factor, speed_headroom * heading_stability * track_stability)
    soc_factor = float(
        np.clip(
            (battery_soc - config.straight_soc_threshold)
            / max(config.battery_soc_max - config.straight_soc_threshold, 1.0),
            0.0,
            1.0,
        )
    )
    brake_required = float(speed > (corner_feasible_speed + 0.6))

    return {
        "speed": speed,
        "speed_norm": speed / max(config.max_speed_mps, 1e-6),
        "psidot": float(psidot),
        "cross_track_error": cross_track_error,
        "cross_track_error_norm": cross_track_error / max(track_length / max(len(trajectory), 1), 0.2),
        "heading_error": heading_error,
        "track_heading_error": track_heading_error,
        "battery_soc": float(battery_soc),
        "battery_soc_norm": float(np.clip(battery_soc / max(config.battery_soc_max, 1e-6), 0.0, 1.0)),
        "track_progress": track_progress,
        "v_target": v_target,
        "v_target_norm": v_target / max(config.max_speed_mps, 1e-6),
        "corner_feasible_speed": corner_feasible_speed,
        "corner_feasible_speed_norm": corner_feasible_speed / max(config.max_speed_mps, 1e-6),
        "preview_speed_limit": preview_speed_limit,
        "preview_speed_limit_norm": preview_speed_limit / max(config.max_speed_mps, 1e-6),
        "max_heading_change": max_heading_change,
        "distance_to_turn_start": float(distance_to_turn_start),
        "distance_to_turn_start_norm": float(distance_to_turn_start / max(track_length, 1e-6)),
        "deploy_track_factor": float(deploy_track_factor),
        "soc_factor": soc_factor,
        "brake_required": brake_required,
    }
