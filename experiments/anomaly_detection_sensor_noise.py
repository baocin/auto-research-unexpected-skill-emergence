import numpy as np
import time

def run_anomaly_sensitivity_experiment(n_agents=100, noise_level=0.5, iterations=30):
    """
    Investigates how the level of sensor noise affects the ability to detect 
    an anomaly (a single outlier) in a local averaging system.
    """
    np.random.seed(42)
    # Base cluster at origin
    points = np.random.normal(0, 1, (n_agents, 2))
    
    # The Anomaly
    anomaly_idx = n_agents // 2
    anomaly_pos = np.array([8.0, 8.0])
    points[anomaly_yst_idx := anomaly_idx] = anomaly_pos
    
    # Noise level: probability that a sensor reports random noise instead of true position
    noise_mask = np.random.rand(n_agents) < noise_level
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_agents):
            p_i = snapshot[i]
            # Local radius for interaction
            radius = 2.5
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= radius) & (dists > 0)
            
            if np.any(neighbors_mask):
                if noise_mask[i]:
                    # Sensor error: reports a random point in the domain
                    obs = np.random.uniform(-10, 10, size=(2,))
                else:
                    obs = np.mean(snapshot[neighbors_mask], axis=0)
                
                new_p = p_i + 0.5 * (obs - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)

    # Metric: Ability to maintain the anomaly's distance from the cluster mean
    cluster_mask = np.arange(n_agents) != anomaly_idx
    cluster_points = current_points[cluster_mask]
    cluster_mean = np.mean(cluster_points, axis=0)
    cluster_std = np.std(cluster_points)
    
    dist_to_anom = np.linalg.norm(current_points[anomaly_idx] - cluster_mean)
    # Success: Anomaly remains at least 2 standard deviations away
    success = 1.0 if dist_to_anom > (2.0 * cluster_std) else 0.0
    return success, dist_to_anom

def run_experiment():
    print(f"experiment: anomaly_detection_noise_sensitivity")
    noise_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    for rate in noise_rates:
        start_time = time.time()
        success, dist = run_anomaly_sensitivity_experiment(noise_level=rate)
        end_time = time.time()
        print(f"noise_{rate:.2f}_success_{success}_{dist:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()