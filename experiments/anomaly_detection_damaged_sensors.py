import numpy as np
import time

def run_anomaly_simulation(rate, n_agents=100, iterations=20):
    """
    Performs a single simulation of the anomaly detection process.
    """
    np.random.seed(42)
    # Create base cluster at 0
    points = np.random.normal(0, 1, (n_agents, 2))
    
    # Define Anomaly
    anomaly_idx = n_agents // 2
    anomaly_pos = np.array([5.0, 5.0])
    points[anomaly_idx] = anomaly_pos.copy()
    
    damage_mask = np.random.rand(n_agents) < rate
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_pts = []
        for i in range(n_agents):
            p_i = snapshot[i]
            # Calculate distances to all other points
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            # Neighbors within radius 2.0 (excluding self)
            neighbors_mask = (dists < 2.0) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                if damage_mask[i]:
                    # Damaged sensor sees random noise
                    obs = np.random.uniform(-5, 5, size=(2,))
                else:
                    # Healthy sensor sees local mean
                    obs = np.mean(snapshot[neighbors_mask], axis=0)
                
                # Move towards observed mean (local update rule)
                new_p = p_i + 0.5 * (obs - p_i)
                new_pts.append(new_p)
            else:
                new_pts.append(p_i)
        current_points = np.array(new_pts)

    # Evaluate detection success
    cluster_mask = np.arange(n_agents) != anomaly_idx
    cluster_points = current_points[cluster_mask]
    c_mean = np.mean(cluster_points, axis=0)
    c_std = np.std(cluster_points)
    
    dist_to_anom = np.linalg.norm(current_points[anomaly_idx] - c_mean)
    # Success: Anomaly is still statistically distinct (2 sigma)
    success = 1.0 if dist_to_anom > (2.0 * c_std) else 0.0
    return success, dist_to_anom

def run_experiment():
    damage_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    print(f"experiment: anomaly_detection_damage")
    
    for rate in damage_rates:
        start_time = time.time()
        success, dist = run_anomaly_simulation(rate)
        end_time = time.time()
        print(f"damage_{rate:.2f}_success_{success}_{dist:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()