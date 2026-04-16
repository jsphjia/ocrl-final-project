# Supervised Data Collection Setup

This setup is tuned for learning battery deployment and corner braking on the current track.

## Battery Profiles

Set `BATTERY_PROFILE` before launching the simulation:

- `aggressive_train` (default): strongest pressure to learn energy timing
- `medium_train`: bridge profile after aggressive training
- `deploy`: runtime profile for final driving

Profile values are configured in `your_controller.py`:

- `aggressive_train`: discharge `5.2`, regen `11.0`, high-power penalty `0.9`
- `medium_train`: discharge `4.1`, regen `13.0`, high-power penalty `0.8`
- `deploy`: discharge `2.5`, regen `17.0`, high-power penalty `0.7`

## Suggested Curriculum

1. Collect expert laps with `BATTERY_PROFILE=aggressive_train`.
2. Collect additional laps with `BATTERY_PROFILE=medium_train`.
3. Fine-tune/validate with `BATTERY_PROFILE=deploy`.

## Track Selection

Use `TRACK_FILE` to switch tracks (already supported in `main.py`).

## Logging

Expert data logging is enabled by default (`EXPERT_LOG=1`).
Logs are written under `controllers/main/data/expert/<track_name>/`.

## Reliability Switch (recommended)

`LEARNING_SAFE_MODE=1` is recommended during data collection.
It applies a conservative speed/braking configuration and automatic anti-stall recovery so laps complete more consistently.

## Free-Acceleration Battery Threshold

Battery discharge now starts above a fixed 25 mph threshold.

Below this speed, discharge is gated to zero; above it, full discharge applies.

You can also set `BATTERY_SPEED_DISCHARGE_RATE` (default `2.4`) to add baseline drain above the threshold even at low throttle cruise.

## Empty-SoC Cruise Cap

When SoC reaches 0, the car tracks a fixed 15 mph cap instead of stalling.

You can set `BATTERY_EMPTY_THRESHOLD` (default `0.5`) so empty-battery behavior triggers slightly before true zero to avoid floating-point/rounding edge cases.

## Example launch environment

```bash
export BATTERY_PROFILE=aggressive_train
export TRACK_FILE=raceline_xy.csv
export EXPERT_LOG=1
export LEARNING_SAFE_MODE=1
export BATTERY_SPEED_DISCHARGE_RATE=2.4
export BATTERY_EMPTY_THRESHOLD=0.5
```

Then run Webots as usual and collect laps.

## Train The Model

From `controllers/main`:

```bash
pip install -r ml_requirements.txt
python train_supervised.py --data-glob "data/expert/**/*.csv" --model-dir "models/supervised"
```

This writes:

- `models/supervised/expert_policy.joblib`
- `models/supervised/metadata.json`

## Run With Learned Battery Policy

To let the trained model control longitudinal force (battery deployment / braking decisions), set:

```bash
export USE_SUPERVISED_POLICY=1
export SUPERVISED_BLEND=0.85
export SUPERVISED_MODEL_DIR=/Users/josephjia/Documents/16745/ocrl-final-project/controllers/main/models/supervised
```

Notes:

- `SUPERVISED_BLEND=0.0` means fully analytic controller.
- `SUPERVISED_BLEND=1.0` means fully model-driven longitudinal force.
- Steering remains on the existing analytic lateral controller.
