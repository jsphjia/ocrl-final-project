# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import closestNode, wrapToPi

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000.0
        self.Iz = 25854.0
        self.m = 1888.6
        self.g = 9.81
        
        # longitudinal PID state variables
        self.e_int_long = 0.0
        self.e_prev_long = 0.0

        # longitudinal PID parameters 
        self.max_speed_mps = 100.0 * 0.44704  # 120 mph in m/s
        self.V_target = self.max_speed_mps     # target velocity (m/s)
        self.Kp_long = 990.0     # P 
        self.Ki_long = 10.0       # I
        self.Kd_long = 100.0      # D
        self.min_turn_speed_mps = 12.0
        self.turn_heading_slow_gain = 0.88
        self.turn_yaw_slow_gain = 2.2
        self.turn_error_slow_gain = 1.5
        self.turn_brake_gain = 3800.0
        self.turn_brake_deadband = 0.15
        self.preview_offsets = [120, 190, 280, 380, 500, 650]
        self.preview_segment_nodes = 18
        self.a_lat_limit = 2.8
        self.max_brake_decel = 3.2
        self.preview_safety_margin_m = 95.0
        self.turn_detect_heading_threshold = 0.06
        self.approach_decel = 2.6
        self.approach_speed_buffer = 4.5
        self.first_corner_nodes = 1100
        self.first_corner_ramp_nodes = 1700
        self.first_corner_speed_cap_mps = 10.5
        self.straight_heading_ref = 0.12
        self.straight_soc_threshold = 60.0
        self.straight_boost_gain = 0.55
        self.straight_boost_max = 0.35

        # Mean path spacing used to convert preview node offsets to metric distance.
        seg = np.diff(trajectory, axis=0)
        seg_len = np.sqrt(np.sum(seg**2, axis=1))
        self.avg_waypoint_spacing = max(float(np.mean(seg_len)), 0.2)

        # Battery SoC model parameters
        self.battery_soc = 100.0
        self.battery_soc_min = 0.0
        self.battery_soc_max = 100.0
        self.battery_discharge_rate = 2.8  # %/s at full throttle
        self.battery_high_power_penalty = 0.7
        self.battery_regen_rate = 2.5      # %/s at full braking
        self.max_longitudinal_force = 10000.0
        self.prev_speed = 0.0

        # lateral controller parameters
        self.LOOKAHEAD_NODES = 120 # lookahead parameter
        
        self.poles = np.array([-1.0, -1.1, -1.2, -2.0])

    def update(self, timestep):

        trajectory = self.trajectory
        
        # Vehicle Constants
        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        
        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        
        # Current longitudinal velocity
        V = np.sqrt(xdot**2 + ydot**2)

        # --------------------|Turn-aware Speed Planning|-------------------------
        cross_track_error, current_node_index = closestNode(X, Y, trajectory)

        N = len(trajectory)
        target_node_index = int((current_node_index + self.LOOKAHEAD_NODES) % N)
        X_target = trajectory[target_node_index, 0]
        Y_target = trajectory[target_node_index, 1]

        target_angle = np.arctan2(Y_target - Y, X_target - X)
        e_psi = wrapToPi(psi - target_angle)

        def heading_from_nodes(start_offset):
            i0 = int((current_node_index + start_offset) % N)
            i1 = int((i0 + self.preview_segment_nodes) % N)
            dx = trajectory[i1, 0] - trajectory[i0, 0]
            dy = trajectory[i1, 1] - trajectory[i0, 1]
            return np.arctan2(dy, dx)

        base_heading = heading_from_nodes(0)
        max_heading_change = 0.0
        critical_offset = self.preview_offsets[-1]
        first_turn_offset = None
        for off in self.preview_offsets:
            h = heading_from_nodes(off)
            dh = abs(wrapToPi(h - base_heading))
            if first_turn_offset is None and dh >= self.turn_detect_heading_threshold:
                first_turn_offset = off
            if dh > max_heading_change:
                max_heading_change = dh
                critical_offset = off

        if first_turn_offset is None:
            first_turn_offset = critical_offset

        # Convert heading change into a curve-limited speed and trigger braking before turn-in.
        est_curve_radius = 8.0 + 45.0 / (max_heading_change + 0.08)
        v_curve_limit = np.sqrt(self.a_lat_limit * est_curve_radius)
        v_curve_limit = np.clip(v_curve_limit, self.min_turn_speed_mps, self.max_speed_mps)

        # Kinematic approach speed limit: velocity that still allows slowing to
        # curve speed by the time the turn starts (not apex), with extra buffer.
        distance_to_turn_start = max(
            first_turn_offset * self.avg_waypoint_spacing - self.preview_safety_margin_m,
            0.0,
        )
        preview_speed_limit = np.sqrt(
            max(v_curve_limit**2 + 2.0 * self.approach_decel * distance_to_turn_start, 0.0)
        )
        preview_speed_limit = np.clip(
            preview_speed_limit - self.approach_speed_buffer,
            self.min_turn_speed_mps,
            self.max_speed_mps,
        )

        # Reduce target speed for larger heading error, yaw rate, and cross-track error.
        heading_term = min(abs(e_psi) / 1.2, 1.0)
        turn_speed_limit = self.max_speed_mps * (1.0 - self.turn_heading_slow_gain * heading_term)
        turn_speed_limit -= self.turn_yaw_slow_gain * abs(psidot)
        turn_speed_limit -= self.turn_error_slow_gain * abs(cross_track_error)
        turn_speed_limit = np.clip(turn_speed_limit, self.min_turn_speed_mps, self.max_speed_mps)
        self.V_target = np.clip(min(turn_speed_limit, preview_speed_limit), self.min_turn_speed_mps, self.max_speed_mps)

        # Force an extra-conservative speed envelope at the start so braking begins well
        # before the first major corner, then smoothly release it.
        if current_node_index < self.first_corner_ramp_nodes:
            blend = np.clip(
                (current_node_index - self.first_corner_nodes)
                / max(self.first_corner_ramp_nodes - self.first_corner_nodes, 1),
                0.0,
                1.0,
            )
            early_corner_cap = self.first_corner_speed_cap_mps + blend * (self.max_speed_mps - self.first_corner_speed_cap_mps)
            self.V_target = min(self.V_target, early_corner_cap)

        # --------------------|Longitudinal Controller (PID)|-------------------------
        
        # error calculation
        e_long = self.V_target - V
        
        # I term update
        self.e_int_long += e_long * delT
        self.e_int_long = np.clip(self.e_int_long, -10, 10)
        
        # D term update
        e_dot_long = (e_long - self.e_prev_long) / delT
        self.e_prev_long = e_long
        
        # PID control law for F
        F = (self.Kp_long * e_long) + \
            (self.Ki_long * self.e_int_long) + \
            (self.Kd_long * e_dot_long)

        # On straight track sections, apply extra propulsion when battery is healthy.
        straight_factor = np.clip(
            (self.straight_heading_ref - max_heading_change) / max(self.straight_heading_ref, 1e-6),
            0.0,
            1.0,
        )
        soc_factor = np.clip(
            (self.battery_soc - self.straight_soc_threshold)
            / max(self.battery_soc_max - self.straight_soc_threshold, 1.0),
            0.0,
            1.0,
        )
        if F > 0.0 and e_long > 0.0:
            boost = min(self.straight_boost_max, self.straight_boost_gain * straight_factor * soc_factor)
            F *= (1.0 + boost)

        # Add braking assist if current speed exceeds turn-safe target speed.
        speed_excess = V - self.V_target
        if speed_excess > self.turn_brake_deadband:
            F -= self.turn_brake_gain * speed_excess

        # Enforce speed cap: no additional propulsion above target max speed.
        if V >= self.max_speed_mps and F > 0.0:
            F = 0.0

        # limit F
        F = np.clip(F, -self.max_longitudinal_force, self.max_longitudinal_force)

        # Battery cannot discharge below 0%; block positive propulsion when empty.
        if self.battery_soc <= self.battery_soc_min and F > 0.0:
            F = 0.0

        throttle_fraction = np.clip(F / self.max_longitudinal_force, 0.0, 1.0)
        brake_fraction = np.clip(-F / self.max_longitudinal_force, 0.0, 1.0)
        effective_discharge = self.battery_discharge_rate * (
            throttle_fraction + self.battery_high_power_penalty * throttle_fraction**2
        )
        soc_delta = (
            self.battery_regen_rate * brake_fraction
            - effective_discharge
        ) * delT
        self.battery_soc = np.clip(
            self.battery_soc + soc_delta,
            self.battery_soc_min,
            self.battery_soc_max,
        )

        # --------------------|Lateral Controller (Pole Placement)|-------------------------
        
        x = np.array([cross_track_error, ydot, e_psi, psidot])

        # coeffs for lateral velocity dynamics
        a22 = -4.0 * Ca / (m * V)
        a23 = 4.0 * Ca / m
        a24 = -2.0 * Ca * (lf - lr) / (m * V)
        a42 = -2.0 * Ca * (lr - lf) / (Iz * V)
        a43 = 2.0 * Ca * (lf - lr) / Iz
        a44 = -2.0 * Ca * (lf**2 + lr**2) / (Iz * V)
        
        b1 = 2.0 * Ca / m
        b2 = 2.0 * Ca * lf / Iz

        # A matrix
        A = np.array([
            [0, 1, 0, 0],
            [0, a22, a23, a24],
            [0, 0, 0, 1],
            [0, a42, a43, a44]
        ])

        # B matrix
        B = np.array([
            [0], 
            [b1], 
            [0], 
            [b2]
        ])

        A = A.astype(np.float64)
        B = B.astype(np.float64)
        
        K = signal.place_poles(A, B, self.poles, method='YT').gain_matrix
        
        delta = (-K.dot(x)).item()
                
        # apply steering limit
        delta = np.clip(delta, -0.5, 0.5)

        # If battery is depleted, do not allow speed to increase.
        if self.battery_soc <= self.battery_soc_min and V > self.prev_speed:
            F = min(F, -1000.0)

        self.prev_speed = V

        # Return all states and calculated control inputs (F, delta, battery SoC)
        return X, Y, xdot, ydot, psi, psidot, F, delta, self.battery_soc
