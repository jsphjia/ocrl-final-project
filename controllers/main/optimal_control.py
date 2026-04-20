from __future__ import annotations

import os
import numpy as np
from scipy.optimize import minimize


def optimize_longitudinal(controller, trajectory, X, Y, xdot, ydot, psi, psidot, battery_soc, config, current_node_index, delT, horizon_steps: int = 8):
    """Simple receding-horizon optimizer for longitudinal force.

    Returns (F_opt, info) where F_opt is scalar force to apply this step.
    This optimizer uses a simplified longitudinal dynamic: v_{k+1} = v_k + (F_k/m)*dt
    and the battery discharge/regen approximations from the controller.
    """
    # Pull needed params from controller
    m = controller.m
    maxF = controller.max_longitudinal_force
    V_th = controller.battery_free_accel_speed_mps
    r_d = controller.battery_discharge_rate
    p = controller.battery_high_power_penalty
    r_s = controller.battery_speed_discharge_rate
    r_regen = controller.battery_regen_rate
    eta = controller.battery_regen_efficiency
    regen_cap = controller.regen_force_cap
    regen_ref_speed = controller.regen_reference_speed
    regen_alpha = controller.regen_nonlinear_exponent
    regen_beta = controller.regen_brake_bias
    regen_mult = controller.battery_regen_multiplier
    straight_mult = controller.straight_deploy_multiplier
    deploy_soc_min = controller.straight_deploy_soc_min

    # compute a current deploy_track_factor and soc_factor (assume roughly constant over horizon)
    feat = controller.policy_feature_config
    from policy_features import build_policy_features

    feat_row = build_policy_features(
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
    deploy_track_factor = float(feat_row.get("deploy_track_factor", 0.0))
    soc_factor = float(feat_row.get("soc_factor", 0.0))

    v0 = float(np.sqrt(xdot**2 + ydot**2))

    # objective weights (configurable via env vars)
    try:
        w_speed = float(os.environ.get("MPC_W_SPEED", "2.0"))
    except Exception:
        w_speed = 2.0
    try:
        w_energy = float(os.environ.get("MPC_W_ENERGY", "0.02"))
    except Exception:
        w_energy = 0.02
    try:
        w_smooth = float(os.environ.get("MPC_W_SMOOTH", "1e-4"))
    except Exception:
        w_smooth = 1e-4
    # soc terminal penalty weight and reserve level (percent)
    try:
        w_soc = float(os.environ.get("MPC_W_SOC", "200.0"))
    except Exception:
        w_soc = 200.0
    try:
        soc_reserve = os.environ.get("MPC_SOC_RESERVE", "30.0")
        soc_reserve = None if soc_reserve == "" else float(soc_reserve)
    except Exception:
        soc_reserve = 30.0
    try:
        hard_soc_penalty = float(os.environ.get("MPC_HARD_SOC_PENALTY", "1000.0"))
    except Exception:
        hard_soc_penalty = 1000.0
    # target: drive toward v_target
    v_target = float(feat_row.get("v_target", v0))

    dt = delT

    # horizon (allow override via env var)
    try:
        H = int(os.environ.get("MPC_HORIZON", str(int(max(1, horizon_steps)))))
    except Exception:
        H = int(max(1, horizon_steps))

    # initial guess: hold current PID-based force as zero
    x0 = np.zeros(H)

    bounds = [(-maxF, maxF) for _ in range(H)]

    def simulate_and_cost(F_seq):
        v = v0
        soc = battery_soc
        cost = 0.0
        prevF = None
        for k in range(H):
            Fk = float(F_seq[k])
            # speed evolution
            a = Fk / m
            v = max(0.0, v + a * dt)

            # throttle / brake fractions
            throttle = np.clip(Fk / maxF, 0.0, 1.0)
            brake_force = np.clip(-Fk, 0.0, maxF)
            brake_frac = brake_force / max(maxF, 1e-6)

            # discharge
            base = r_d * (throttle + p * throttle ** 2)
            speed_gate = 1.0 if v > V_th else 0.0
            speed_excess_ratio = np.clip((v - V_th) / max(V_th, 1e-6), 0.0, 1.0)
            parasitic = r_s * speed_gate * (0.5 + 1.5 * speed_excess_ratio)
            eff_dis = (base * speed_gate) + parasitic
            if throttle > 0.0 and soc > deploy_soc_min:
                straight_deploy = straight_mult * deploy_track_factor * throttle
                eff_dis *= (1.0 + max(straight_deploy, 0.0))

            # regen
            regen_force = min(brake_force, regen_cap)
            regen_power = regen_force * max(v, 0.0)
            regen_ref_power = regen_cap * regen_ref_speed
            regen_fraction = np.clip(regen_power / max(regen_ref_power, 1e-6), 0.0, 1.0)
            regen_fraction = regen_fraction ** regen_alpha
            regen_fraction *= (1.0 + regen_beta * brake_frac)
            regen_fraction = np.clip(regen_fraction, 0.0, 1.0)
            eff_reg = r_regen * eta * regen_fraction * regen_mult
            # corner bonus ignored for simplicity

            soc += (eff_reg - eff_dis) * dt

            # penalties / costs
            cost += w_speed * (v_target - v) ** 2
            energy_net = eff_dis - eff_reg
            # energy_net positive => discharge; scale cost by positive discharge
            cost += w_energy * max(0.0, energy_net)
            if prevF is not None:
                cost += w_smooth * (Fk - prevF) ** 2
            prevF = Fk

            # heavy penalty for dropping below min SoC
            if soc < controller.battery_soc_min:
                cost += 1e6 * (controller.battery_soc_min - soc)

            # if predicted SoC falls below the reserve, penalize strongly
            if soc_reserve is not None:
                try:
                    soc_reserve_val = float(soc_reserve)
                except Exception:
                    soc_reserve_val = soc_reserve
                if soc < soc_reserve_val:
                    gap = max(0.0, soc_reserve_val - soc)
                    cost += w_soc * (gap ** 2) + hard_soc_penalty * gap
                # if getting close to reserve, increase energy penalty to discourage discharge
                elif soc < (soc_reserve_val + 5.0):
                    cost += (w_energy * 5.0) * max(0.0, energy_net)

        # terminal penalty (after horizon) for violating reserve
        if soc_reserve is not None:
            try:
                soc_reserve_val = float(soc_reserve)
            except Exception:
                soc_reserve_val = soc_reserve
            if soc < soc_reserve_val:
                cost += w_soc * (max(0.0, soc_reserve_val - soc) ** 2)

        return cost

    try:
        maxiter = int(os.environ.get("MPC_MAXITER", "400"))
    except Exception:
        maxiter = 400
    try:
        ftol = float(os.environ.get("MPC_FTOL", "1e-4"))
    except Exception:
        ftol = 1e-4

    res = minimize(simulate_and_cost, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': ftol})
    # compute predicted terminal soc and return full sequence for logging
    F_opt_seq = res.x if getattr(res, 'x', None) is not None else x0

    def simulate_sequence(F_seq):
        v = v0
        soc = battery_soc
        seq = []
        for k in range(len(F_seq)):
            Fk = float(F_seq[k])
            a = Fk / m
            v = max(0.0, v + a * dt)
            throttle = np.clip(Fk / maxF, 0.0, 1.0)
            brake_force = np.clip(-Fk, 0.0, maxF)

            # discharge
            base = r_d * (throttle + p * throttle ** 2)
            speed_gate = 1.0 if v > V_th else 0.0
            speed_excess_ratio = np.clip((v - V_th) / max(V_th, 1e-6), 0.0, 1.0)
            parasitic = r_s * speed_gate * (0.5 + 1.5 * speed_excess_ratio)
            eff_dis = (base * speed_gate) + parasitic
            if throttle > 0.0 and soc > deploy_soc_min:
                straight_deploy = straight_mult * deploy_track_factor * throttle
                eff_dis *= (1.0 + max(straight_deploy, 0.0))

            # regen
            regen_force = min(brake_force, regen_cap)
            regen_power = regen_force * max(v, 0.0)
            regen_ref_power = regen_cap * regen_ref_speed
            regen_fraction = np.clip(regen_power / max(regen_ref_power, 1e-6), 0.0, 1.0)
            regen_fraction = regen_fraction ** regen_alpha
            regen_fraction *= (1.0 + regen_beta * (brake_force / max(maxF, 1e-6)))
            regen_fraction = np.clip(regen_fraction, 0.0, 1.0)
            eff_reg = r_regen * eta * regen_fraction * regen_mult

            soc += (eff_reg - eff_dis) * dt
            seq.append({'F': Fk, 'v': v, 'soc': soc, 'eff_dis': eff_dis, 'eff_reg': eff_reg})
        return seq

    seq = simulate_sequence(F_opt_seq)
    pred_soc = float(seq[-1]['soc']) if len(seq) > 0 else float(battery_soc)

    F0 = float(np.clip(F_opt_seq[0], -maxF, maxF))
    info = {
        'success': bool(getattr(res, 'success', False)),
        'fun': float(getattr(res, 'fun', 0.0)),
        'nit': getattr(res, 'nit', None),
        'pred_soc': pred_soc,
        'F_seq': [float(x) for x in list(F_opt_seq)],
    }
    return F0, info
