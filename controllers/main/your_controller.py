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
        self.e_int_lat = 0.0
        self.e_prev_lat = 0.0

        # longitudinal PID parameters 
        self.max_speed_mps = 150.0 * 0.44704  # 120 mph in m/s
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
        self.deploy_heading_margin = 0.45
        self.deploy_crosstrack_margin = 2.5
        self.corner_brake_margin_mps = 0.6
        self.deploy_over_target_margin_mps = 10.0
        self.deploy_force_gain = 1800.0
        self.deploy_soc_min = 25.0

        # Mean path spacing used to convert preview node offsets to metric distance.
        seg = np.diff(trajectory, axis=0)
        seg_len = np.sqrt(np.sum(seg**2, axis=1))
        self.avg_waypoint_spacing = max(float(np.mean(seg_len)), 0.2)

        # Battery SoC model parameters
        self.battery_soc = 100.0
        self.battery_soc_min = 0.0
        self.battery_soc_max = 100.0
        self.battery_discharge_rate = 2.5  # %/s at full throttle
        self.battery_high_power_penalty = 0.7
        self.straight_deploy_multiplier = 0.95
        self.straight_deploy_soc_min = 20.0
        self.battery_regen_rate = 18.0     # %/s at normalized peak regen power
        self.battery_regen_efficiency = 0.95
        self.regen_reference_speed = 14.0
        self.regen_force_cap = 8500.0
        self.regen_nonlinear_exponent = 0.5
        self.regen_brake_bias = 0.9
        self.corner_regen_heading_ref = 0.18
        self.corner_regen_gain = 1.25
        self.max_longitudinal_force = 10000.0
        self.prev_speed = 0.0

        # lateral controller parameters
        self.Kp_lat = 1.6         # P
        self.Ki_lat = 0.0001      # I
        self.Kd_lat = 1.1         # D
        self.LOOKAHEAD_NODES = 120 # lookahead parameter

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

        # Trajectory-based feasibility speed for deciding brake vs. deployment.
        corner_feasible_speed = np.clip(preview_speed_limit, self.min_turn_speed_mps, self.max_speed_mps)

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
            corner_feasible_speed = min(corner_feasible_speed, early_corner_cap)

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
        speed_headroom = np.clip((self.V_target - V) / max(self.max_speed_mps, 1e-6), 0.0, 1.0)
        heading_stability = np.clip(1.0 - abs(e_psi) / max(self.deploy_heading_margin, 1e-6), 0.0, 1.0)
        track_stability = np.clip(1.0 - abs(cross_track_error) / max(self.deploy_crosstrack_margin, 1e-6), 0.0, 1.0)
        deploy_track_factor = max(straight_factor, speed_headroom * heading_stability * track_stability)
        brake_required = V > (corner_feasible_speed + self.corner_brake_margin_mps)
        soc_factor = np.clip(
            (self.battery_soc - self.straight_soc_threshold)
            / max(self.battery_soc_max - self.straight_soc_threshold, 1.0),
            0.0,
            1.0,
        )
        if F > 0.0 and e_long > 0.0 and not brake_required:
            boost = min(self.straight_boost_max, self.straight_boost_gain * deploy_track_factor * soc_factor)
            F *= (1.0 + boost)

        # Battery deployment can push speed above V_target when trajectory preview
        # indicates the car can still make the upcoming corner.
        deploy_speed_ceiling = min(
            self.max_speed_mps,
            corner_feasible_speed + self.deploy_over_target_margin_mps * deploy_track_factor,
        )
        if (
            not brake_required
            and self.battery_soc > self.deploy_soc_min
            and V < deploy_speed_ceiling
        ):
            deploy_headroom = deploy_speed_ceiling - V
            F += self.deploy_force_gain * deploy_track_factor * soc_factor * deploy_headroom

        # Brake only when trajectory preview indicates corner entry is too fast.
        speed_excess = V - corner_feasible_speed
        if brake_required and speed_excess > self.turn_brake_deadband:
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
        brake_force = np.clip(-F, 0.0, self.max_longitudinal_force)
        brake_fraction = brake_force / max(self.max_longitudinal_force, 1e-6)
        effective_discharge = self.battery_discharge_rate * (
            throttle_fraction + self.battery_high_power_penalty * throttle_fraction**2
        )

        # Aggressive energy deployment on straights when SoC is healthy.
        if throttle_fraction > 0.0 and self.battery_soc > self.straight_deploy_soc_min:
            straight_deploy = self.straight_deploy_multiplier * deploy_track_factor * throttle_fraction
            effective_discharge *= (1.0 + max(straight_deploy, 0.0))

        # Regen depends on instantaneous braking power, not just brake duration.
        regen_force = min(brake_force, self.regen_force_cap)
        regen_power = regen_force * max(V, 0.0)
        regen_reference_power = self.regen_force_cap * self.regen_reference_speed
        regen_fraction = np.clip(regen_power / max(regen_reference_power, 1e-6), 0.0, 1.0)
        regen_fraction = regen_fraction**self.regen_nonlinear_exponent
        regen_fraction *= (1.0 + self.regen_brake_bias * brake_fraction)
        regen_fraction = np.clip(regen_fraction, 0.0, 1.0)
        effective_regen = self.battery_regen_rate * self.battery_regen_efficiency * regen_fraction

        # Extra corner regen weighting so heavy corner braking can recover large SoC chunks.
        corner_factor = np.clip(max_heading_change / max(self.corner_regen_heading_ref, 1e-6), 0.0, 1.0)
        effective_regen *= (1.0 + self.corner_regen_gain * corner_factor * (0.4 + 0.6 * brake_fraction))

        soc_delta = (
            effective_regen
            - effective_discharge
        ) * delT
        self.battery_soc = np.clip(
            self.battery_soc + soc_delta,
            self.battery_soc_min,
            self.battery_soc_max,
        )

        # --------------------|Lateral Controller (Pole Placement)|-------------------------
        
        # current closest node index
        cross_track_error, current_node_index = closestNode(X, Y, trajectory)
        
        # target node using lookahead
        N = len(trajectory)
        target_node_index = int((current_node_index + self.LOOKAHEAD_NODES) % N)
        
        # target position
        X_target = trajectory[target_node_index, 0]
        Y_target = trajectory[target_node_index, 1]
        
        # steering error
        
        # heading to the target point:
        target_angle = np.arctan2(Y_target - Y, X_target - X)
        
        # heading error component
        e_psi = wrapToPi(target_angle - psi)

        # PID calculation on heading error
        
        # I term update
        self.e_int_lat += e_psi * delT
        self.e_int_lat = np.clip(self.e_int_lat, -10, 10)
        
        # D term update
        e_dot_lat = (e_psi - self.e_prev_lat) / delT
        self.e_prev_lat = e_psi
        
        # PID control law for delta
        delta = (self.Kp_lat * e_psi) + \
                (self.Ki_lat * self.e_int_lat) + \
                (self.Kd_lat * e_dot_lat)
                
        # apply steering limit
        delta = np.clip(delta, -0.5, 0.5)

        # If battery is depleted, do not allow speed to increase.
        if self.battery_soc <= self.battery_soc_min and V > self.prev_speed:
            F = min(F, -1000.0)

        self.prev_speed = V

        # Return all states and calculated control inputs (F, delta, battery SoC)
        return X, Y, xdot, ydot, psi, psidot, F, delta, self.battery_soc
