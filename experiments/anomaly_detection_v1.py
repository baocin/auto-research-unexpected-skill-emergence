import numpy as np
import time

def run_anomaly_detection_experiment(n_agents=100, anomaly_rate=0.05, sensor_damage_rate=0.2):
    """
    Investigates 'Anomaly Detection': Can the population identify and 
    effectively isolate/ignore agents with corrupted (extreme) sensor values?
    """
    np.random.seed(42)
    # True underlying value is 0.5
    true_value = 0.5
    
    # Initialize agent sensors with Gaussian noise around true value
    sensors = np.random.normal(true_value, 0.1, n_agents)
    
    # Inject anomalies (extreme values)
    num_anomalies = int(n_agents * anomaly_reg) if 'num_anomalies' in locals() else int(n_agents * anomaly_rate)
    # Let's just simplify this to avoid the walrus error and logic confusion
    num_anomalies = int(n_agents * anomaly_rate)
    anomaly_indices = np.random.choice(n_agents, num_anomalies, replace=False)
    sensors[anomaly_indices] = np.random.uniform(2.0, 5.0, num_anomalies)
    
    # Apply sensor damage (noise/corruption to the reading process)
    damage_mask = np.random.rand(n_agents) < sensor_damage_rate
    sensors[damage_mask] += np.random.normal(0, 0.5, np.sum(damage_mask))

    # Consensus mechanism: Local Median (Robust to outliers) vs Local Mean (Sensitive)
    def compute_metric(aggregator):
        current_vals = sensors.copy()
        for _ in range(20):
            new_vals = []
            for i in range(n_agents := n_agents):
                # Simulating local neighborhood interaction
                neighbors = current_vals[np.abs(current_vals - current_vals[i]) < 1.0]
                if len(neighbors) > 0:
                    target = aggregator(neighbors)
                    new_vals.append(current_vals[i] + 0.5 * (target - current_vals[i]))
                else:
                    new_vals.append(current_vals[i])
            current_vals = np.array(new_vals)
        
        # Success is the proximity of final consensus to true value
        error = np.abs(np.mean(current_vals) - true_value)
        return 1.0 / (1.0 + error)

    success_mean = compute_metric(np.mean)
    success_median = compute_metric(np.median)
    
    return success_mean, success_median

def run_experiment():
    print(f"experiment: anomaly_detection_v1")
    anomaly_rates = [0.01, 0.05, 0.1, 0.2]
    for ar in anomaly_rates:
        start_time = time.time()
        try:
            s_mean, s_med = run_anomaly_detection_experiment(anomaly_rate=ar)
            duration = time.time() - start_time
            print(f"anomaly_rate_{ar:.2f}_mean_success_{s_mean:.4f}_median_success_{s_med:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{ar}_{e}")

if __name__ == "__main__":
    run_experiment()