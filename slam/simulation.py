
import numpy as np

class Simulation:
    def __init__(self, num_steps=100, x_dim=4, z_dim=2):
        self.num_steps = num_steps
        self.dt = 0.1
        self.time = 0.0
        
        # State: [x, y, vx, vy]
        self.true_state = np.zeros(x_dim)
        
        # Landmarks for corridor (degenerate along x-axis - no features to fix x position well if we were doing scan matching, 
        # but for point features, collinearity along y=constant lines means poor y-constraining? 
        # Actually a corridor allows moving along it (longitudinal) with low observability if walls are featureless.
        # Here we simulate feature-based degeneracy.
        
        self.landmarks = []
        self._generate_corridor_landmarks()

    def _generate_corridor_landmarks(self):
        # Generate two parallel lines of landmarks
        # moving along X axis
        for x in range(0, 100, 5):
            self.landmarks.append([x, 2.0])  # Left wall
            self.landmarks.append([x, -2.0]) # Right wall
        
        self.landmarks = np.array(self.landmarks)

    def step(self, u):
        """
        Move robot with control input u = [ax, ay]
        """
        # Constant acceleration model for simplicity or just velocity model
        # State: x, y, vx, vy
        # F matrix
        F = np.eye(4)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        
        # B matrix (effect of acceleration)
        B = np.zeros((4, 2))
        B[2, 0] = self.dt
        B[3, 1] = self.dt
        
        noise = np.random.normal(0, 0.1, 4) # Process noise
        
        self.true_state = np.dot(F, self.true_state) + np.dot(B, u) + noise
        self.time += self.dt
        
        return self.true_state

    def get_measurements(self):
        """
        Returns visible landmarks and a synthetic Information Matrix.
        In a real SLAM, H_info comes from scan matching.
        Here we simulate it:
        If we are in the 'corridor' (which we always are in this sim),
        and we are moving parallel to it, the longitudinal constraint might be weak 
        if landmarks were featureless lines. But with points, it's actually fine.
        
        To SIMULATE degeneracy for the sake of the algorithm:
        We will return a high condition number H_info periodically or based on location.
        """
        
        # Simulate measurements (simple GPS-like for position only, or range-bearing)
        # Let's say we measure landmarks.
        # z = [range, bearing] or just [x, y] of landmarks relative to robot.
        # For this KF task, let's simplify: Measurement is just Robot Position (e.g. from a noisy GPS or generic 'pose estimate')
        # z = [x, y]
        
        noise = np.random.normal(0, 0.5, 2)
        z = self.true_state[0:2] + noise
        
        # Simulate Hessian/Information Matrix
        # In a corridor (aligned with X), uncertainty in X is high (low info), uncertainty in Y is low (high info).
        # H = J^T * J
        # If we have weak constraints in X:
        
        # High information in Y (walls are close), Low in X (corridor is long)
        # Eigenvalues: large for Y, small for X.
        
        eig_val_y = 100.0 # High certainty perpendicular to walls
        
        # Vary X certainty: sometimes good, sometimes bad (degenerate)
        if 20 < self.true_state[0] < 60: # In the middle of the corridor
            eig_val_x = 0.1 # Very low certainty/degenerate
        else:
            eig_val_x = 50.0 # Normal
            
        # Construct Hessian from these eigenvalues
        # H = V * D * V^T, let's assume axis aligned for simplicity
        H_info = np.diag([eig_val_x, eig_val_y])
        
        return z, H_info

