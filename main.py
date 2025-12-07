# Adaptive Kalman Filter-based SLAM in LiDAR Degenerated Environments

import numpy as np
import matplotlib.pyplot as plt
from slam import AdaptiveKalmanFilter, Simulation

def run_slam_system():
    # 1. Setup Simulation
    steps = 100
    sim = Simulation(num_steps=steps)
    
    # 2. Setup Adaptive Kalman Filter
    # State: x, y, vx, vy
    # Measurement: x, y
    akf = AdaptiveKalmanFilter(x_dim=4, z_dim=2, degeneracy_threshold=10.0)
    
    # Initialize Filter State
    akf.x = np.array([0, 0, 1.0, 0]) # Initial guess
    akf.P *= 1.0 # Initial uncertainty
    akf.Q *= 0.1 # Process noise
    akf.R *= 0.5 # Measurement noise

    # Matrices for Filter Prediction (Constant Velocity Model)
    dt = 0.1
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    
    B = np.zeros((4, 2))
    B[2, 0] = dt
    B[3, 1] = dt

    # Measurement Matrix H (maps state to measurement)
    # We measure x, y directly
    H_meas = np.zeros((2, 4))
    H_meas[0, 0] = 1
    H_meas[1, 1] = 1

    # History for plotting
    true_path = []
    est_path = []
    degeneracy_status = []
    cond_numbers = []

    print("Starting SLAM Simulation...")
    control = np.array([0.0, 0.0]) # Constant velocity, zero accel

    for i in range(steps):
        # A. Ground Truth Simulation Step
        true_x = sim.step(control)
        true_path.append(true_x[:2])

        # B. Get Measurements (and simulated Optimization Hessian from Scan Matching)
        z, H_info = sim.get_measurements()

        # C. Filter Prediction
        akf.predict(F, u=control, B=B)

        # D. Filter Update (Adaptive)
        akf.update(z, H_meas, information_matrix=H_info)

        # Store results
        est_path.append(akf.x[:2])
        degeneracy_status.append(akf.is_degenerate)
        cond_numbers.append(akf.condition_number)
        
        if i % 10 == 0:
            print(f"Step {i}: Degenerate={akf.is_degenerate} (Cond={akf.condition_number:.1f})")

    # 3. Visualization
    true_path = np.array(true_path)
    est_path = np.array(est_path)
    
    plt.figure(figsize=(12, 6))
    
    # Trajectory Plot
    plt.subplot(1, 2, 1)
    plt.plot(true_path[:, 0], true_path[:, 1], 'g--', label='Ground Truth')
    plt.plot(est_path[:, 0], est_path[:, 1], 'b-', label='Estimated (AKF)')
    
    # Highlight degenerate sections
    deg_indices = [i for i, d in enumerate(degeneracy_status) if d]
    if deg_indices:
        plt.scatter(est_path[deg_indices, 0], est_path[deg_indices, 1], c='r', marker='x', label='Degeneracy Detected')

    plt.legend()
    plt.title("Robot Trajectory")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True)
    
    # Condition Number Plot
    plt.subplot(1, 2, 2)
    plt.plot(cond_numbers, 'k-')
    plt.axhline(y=10.0, color='r', linestyle='--', label='Threshold')
    plt.title("Hessian Condition Number")
    plt.xlabel("Step")
    plt.ylabel("Condition Number")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_slam_system()