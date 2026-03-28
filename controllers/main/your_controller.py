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
        self.V_target = 10.0      # target velocity (m/s)
        self.Kp_long = 990.0     # P 
        self.Ki_long = 10.0       # I
        self.Kd_long = 100.0      # D

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

        # limit F
        F = np.clip(F, -10000.0, 10000.0)

        # --------------------|Lateral Controller (Pole Placement)|-------------------------

        # e1
        cross_track_error, current_node_index = closestNode(X, Y, trajectory)
        
        # target node using lookahead
        N = len(trajectory)
        target_node_index = int((current_node_index + self.LOOKAHEAD_NODES) % N)
        
        # target position
        X_target = trajectory[target_node_index, 0]
        Y_target = trajectory[target_node_index, 1]
        
        # heading to the target point
        target_angle = np.arctan2(Y_target - Y, X_target - X)
        
        # e2
        e_psi = wrapToPi(psi - target_angle)
        
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

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
