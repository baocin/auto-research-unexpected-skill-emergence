import numpy as np
import time

def run_anomaly_experiment(n_agents=100, anomaly_rate=0.2, noise_level=0.1, iterations=30):
    """
    Investigates if agents can 'self-heal' by using a weighted consensus 
    mechanism that de-weights neighbors with high deviations (anomalies).
    """
    np.random.seed(42)
    # Initial values: all agents start near 0
    values = np.random.normal(0, 1, n_agents)
    
    # Introduce anomalies: a percentage of agents have 'broken' sensors 
    num_anomalies = int(n_agents * anomaly_rate)
    anomaly_indices = np.random.choice(n_agents, num_anomalies, replace=
False)
    values[anomaly_indices] += 15.0 # Large anomalies
    
    initial_variance = np.var(values)
    current_values = values.copy()
    
    for _ in range(iterations):
        new_values = current_values.copy()
        snapshot = current_values.copy()
        
        for i in range(n_agents):
            # Calculate weights based on deviation from the local mean
            # We use a 'z-score' like weighting: weight = 1 / (1 + |x - mu|)
            local_mu = np.mean(snapshot)
            diffs = np.abs(snapshot - local_mu)
            weights = 1.0 / (1.0 + diffs)
            
            # Calculate the weighted mean of all agents
            weighted_mean = np.sum(snapshot * weights) / np.sum(weights)
            
            # Update with noise
            noise = np.random.normal(0, noise_level)
            new_values[i] = snapshot[i] + 0.5 * (weighted_mean - snapshot[i]) + noise
            
        current_values = new_values

    final_variance = np.var(current_values)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: anomaly_detection_sensor_damage_weighted")
    # Test different anomaly rates
    rates = [0.05, 0.2, 0.4]
    for r in rates:
        start_time = time.time()
        success = run_anomaly_experiment(anomaly_rate=r)
        end_time = time.time()
        print(f"anomaly_rate_{r:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()