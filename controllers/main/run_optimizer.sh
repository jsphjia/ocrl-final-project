#!/usr/bin/env bash
# Launcher for optimizer-only controller (bash/zsh)

set -euo pipefail

# Optimizer mode
export USE_OPTIMIZER=1
export OPTIMIZER_BLEND=1.0

# MPC tuning
export MPC_HORIZON=12
export MPC_W_SPEED=3.0
export MPC_W_ENERGY=0.02
export MPC_W_SOC=200.0
export MPC_SOC_RESERVE=30.0
export MPC_HARD_SOC_PENALTY=2000.0
export MPC_MAXITER=600
export MPC_FTOL=1e-5

# Logging and battery profile
export EXPERT_LOG=1
export BATTERY_PROFILE=deploy

# Launch Webots (adjust path if needed)
open -a Webots "$(cd "$(dirname "$0")/../.." && pwd)/worlds/automotive_new.wbt"

echo "Launched Webots with optimizer settings."
