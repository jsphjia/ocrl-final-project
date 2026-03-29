import numpy as np
import pandas as pd
from pathlib import Path

from commonroad_raceline_planner.racetrack import RaceTrack, RaceTrackFactory
from commonroad_raceline_planner.planner.ftm_planner.ftm_mc_planner import MinimumCurvaturePlanner
from commonroad_raceline_planner.configuration.ftm_config.ftm_config import FTMConfigFactory
from commonroad_raceline_planner.configuration.ftm_config.optimization_config import OptimizationType

# --- Paths (all relative to repo root) ---
BASE = Path(__file__).parent
CSV_PATH     = BASE / "inputs" / "buggyTrace.csv"
INI_PATH     = BASE / "params" / "racecar.ini"
GGV_PATH     = BASE / "inputs" / "ggv.csv"
AX_PATH      = BASE / "inputs" / "ax_max_machines.csv"

# --- Load your centerline ---
df = pd.read_csv(CSV_PATH, header=None, names=["x_m", "y_m"])
df = df.iloc[::20].reset_index(drop=True)  # downsample every 20th point

# Drop duplicate closing point if present
first = df.iloc[0]
last = df.iloc[-1]
if abs(last.x_m - first.x_m) < 0.01 and abs(last.y_m - first.y_m) < 0.01:
    df = df.iloc[:-1].reset_index(drop=True)

print(f"Using {len(df)} points")

x_m = df["x_m"].values
y_m = df["y_m"].values

# --- Set track width (edit to match your actual track) ---
HALF_WIDTH = 5  # meters per side (= 4m total track width)
w_tr_right = np.full(len(x_m), HALF_WIDTH)
w_tr_left  = np.full(len(x_m), HALF_WIDTH)

# --- Build RaceTrack ---
race_track = RaceTrack(
    x_m=x_m, y_m=y_m,
    w_tr_right_m=w_tr_right,
    w_tr_left_m=w_tr_left,
)

# Flip to clockwise if needed (required by the optimizer)
pts = race_track.to_2d_np_array().T
if not RaceTrackFactory.check_clockwise(pts):
    race_track = RaceTrack(
        x_m=x_m[::-1], y_m=y_m[::-1],
        w_tr_right_m=w_tr_right[::-1],
        w_tr_left_m=w_tr_left[::-1],
    )

# --- Load config ---
config = FTMConfigFactory().generate_from_files(
    path_to_ini=INI_PATH,
    ggv_file=GGV_PATH,
    ax_max_machines_file=AX_PATH,
    optimization_type=OptimizationType.MINIMUM_CURVATURE,
)

# --- Plan ---
planner = MinimumCurvaturePlanner(race_track=race_track, config=config)
raceline = planner.plan()

# --- Export ---
raceline.export_trajectory_to_csv_file(
    export_path=BASE / "outputs" / "min_curv_raceline.csv",
    ggv_file_path=GGV_PATH  # point to the input ggv file, not an output
)
print("Done! Max velocity:", raceline.velocity_long_per_point.max(), "m/s")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Get raceline data
rl_x = raceline.points[:, 0]
rl_y = raceline.points[:, 1]

# Compute centerline boundaries from normal vectors
# Normal vector at each point (perpendicular to track direction)
dx = np.gradient(x_m)
dy = np.gradient(y_m)
lengths = np.sqrt(dx**2 + dy**2)
nx = -dy / lengths  # normal x
ny =  dx / lengths  # normal y

# Left and right boundaries
left_x  = x_m + nx * HALF_WIDTH
left_y  = y_m + ny * HALF_WIDTH
right_x = x_m - nx * HALF_WIDTH
right_y = y_m - ny * HALF_WIDTH

plt.figure(figsize=(12, 10))
plt.plot(left_x,  left_y,  'k-',  linewidth=1, label='Left boundary')
plt.plot(right_x, right_y, 'k-',  linewidth=1, label='Right boundary')
plt.plot(x_m,     y_m,     'b--', linewidth=1, label='Centerline')
plt.plot(rl_x,    rl_y,    'r-',  linewidth=2, label='Min Curvature Raceline')

plt.legend()
plt.axis('equal')
plt.title('Minimum Curvature Raceline with Track Boundaries')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.savefig(BASE / 'outputs' / 'raceline_plot.png')
plt.show()

raceline_df = pd.DataFrame({
    'x_m': raceline.points[:, 0],
    'y_m': raceline.points[:, 1]
})
raceline_df.to_csv(BASE / 'outputs' / 'raceline_xy.csv', index=False)
print("Saved raceline x/y to outputs/raceline_xy.csv")

import pandas as pd
rl = pd.read_csv(BASE / 'outputs' / 'raceline_xy.csv')
print('Raceline:')
print('X range:', rl.x_m.min(), 'to', rl.x_m.max())
print('Y range:', rl.y_m.min(), 'to', rl.y_m.max())
print(rl.head())

print('\nOriginal trace:')
print('X range:', x_m.min(), 'to', x_m.max())
print('Y range:', y_m.min(), 'to', y_m.max())