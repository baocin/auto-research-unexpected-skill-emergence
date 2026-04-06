import numpy as np
import time

def run_anomaly_detection_v3(n_agents=100, anomaly_rate=0.05, sensor_damage_rate=0.2):
    """
    Investigates 'Anomaly Detection': 
    Comparing Mean vs Median in the presence of extreme outliers and sensor noise.
    """
    np.random.seed(42)
    true_value = 0.5
    
    # Initialize sensors with Gaussian noise around true value
    sensors_init = np.random.normal(true_value, 0.1, n_agents)
    
    # Inject anomalies (extreme values far from the mean)
    num_anomalies = int(n_agents * anomaly_rate)
    if num_anomalies > 0:
        anomaly_indices = np.random.choice(n_agents, num_anomalies, replace=False)
        sensors_init[anomaly_indices] = np.random.uniform(2.0, 5.0, num_anomalies)
    
    # Apply sensor damage (additive noise to the reading process)
    damage_mask = np.random.rand(n_agents) < sensor_damage_rate
    sensors_init[damage_mask] += np.random.normal(0, 0.5, np.sum(damage_mask))

    def compute_metric(aggregator):
        current_vals = sensors_init.copy()
        for _ in range(25):
            new_vals = []
            for i in range(n_agents):
                # Local interaction: agents look at neighbors within a value-range window
                diffs = np.abs(current_vals - current_vals[i])
                neighbors_mask = diffs < 1.0
                neighbors = current_vals[neighbors_mask]
                
                if len(neighbors) > 0:
                    target = aggregator(neighbors)
                    new_vals.append(current_vals[i] + 0.4 * (target - current_vals[i]))
                else:
                    new_vals.append(current_vals[i])
            current_vals = np.array(new_vals)
        
        error = np.abs(np.mean(current_vals) - true_value)
        return 1.0 / (1.0 + error)

    success_mean = compute_metric(np.mean)
    success_median = compute_metric(np.median)
    return success_mean, success_median

def run_experiment():
    print(f"experiment: anomaly_detection_v3")
    anomaly_rates = [0.01, 0.05, 0.1, 0.2]
    for ar in anomaly_rates:
        start_time = time.time()
        try:
            s_mean, s_median = run_anomaly_detection_v3(anomaly_rate=ar)
            duration = time.time() - start_time
            print(f"anomaly_rate_{ar:.2f}_mean_success_{s_mean:.4f}_median_success_{s_median:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{ar}_{e}")

if __name__ == "__main__":
    run_experiment()