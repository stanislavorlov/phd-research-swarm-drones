import numpy as np
from scipy.optimize import differential_evolution
import os
import json

def objective_function(x, n_uavs, n_users, user_positions, n_channels):
    """
    Objective: Maximize coverage (minimize distance) + Minimize interference.
    x: [x1, y1, c1, x2, y2, c2, ...] where (x,y) is position and c is channel.
    """
    total_cost = 0
    uav_positions = np.zeros((n_uavs, 2))
    uav_channels = np.zeros(n_uavs, dtype=int)
    
    for j in range(n_uavs):
        idx = j * 3
        uav_positions[j] = [x[idx], x[idx+1]]
        uav_channels[j] = int(round(x[idx+2]))
        
    # Coverage Cost: Sum of distance from each user to their nearest serving UAV
    for i in range(n_users):
        dists = np.linalg.norm(uav_positions - user_positions[i], axis=1)
        total_cost += np.min(dists)
        
    # Interference Cost: Penalize UAVs that are close to each other AND on the same channel
    interference_penalty = 0
    for j1 in range(n_uavs):
        for j2 in range(j1 + 1, n_uavs):
            if uav_channels[j1] == uav_channels[j2]:
                dist_between_uavs = np.linalg.norm(uav_positions[j1] - uav_positions[j2])
                # If they are closer than a threshold, heavily penalize
                if dist_between_uavs < 4.0:
                    interference_penalty += (4.0 - dist_between_uavs) * 10.0
                    
    return total_cost + interference_penalty

def generate_expert_dataset(num_samples=100, n_uavs=3, n_users=10, world_size=10.0, n_channels=3):
    dataset = []
    print(f"Generating expert dataset with {num_samples} scenarios...")
    
    # x bounds: (0, world_size), y bounds: (0, world_size), c bounds: (0, n_channels-1)
    bounds = []
    for _ in range(n_uavs):
        bounds.extend([(0, world_size), (0, world_size), (0, n_channels - 1)])
        
    for _ in range(num_samples):
        user_positions = np.random.uniform(0, world_size, size=(n_users, 2))
        
        # Differential evolution handles mixed-integer by integrality constraints (available in newer scipy)
        # We specify the 3rd variable of every UAV as an integer
        integrality = np.zeros(len(bounds))
        for j in range(n_uavs):
            integrality[j*3 + 2] = 1 # 1 means integer
            
        res = differential_evolution(
            objective_function, 
            bounds=bounds,
            args=(n_uavs, n_users, user_positions, n_channels),
            integrality=integrality,
            maxiter=100,
            popsize=10,
            tol=0.01,
            disp=False
        )
        
        if res.success:
            optimal_uavs = []
            for j in range(n_uavs):
                idx = j * 3
                optimal_uavs.append({
                    "pos": [res.x[idx], res.x[idx+1]],
                    "channel": int(round(res.x[idx+2]))
                })
                
            dataset.append({
                "users": user_positions.tolist(),
                "optimal_uavs": optimal_uavs,
                "cost": res.fun
            })
            
    print(f"Successfully generated {len(dataset)} expert trajectories.")
    return dataset

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = generate_expert_dataset(num_samples=5, n_uavs=3, n_users=10, n_channels=3)
    
    with open("data/expert_dataset_sample.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print("Saved sample to data/expert_dataset_sample.json")
