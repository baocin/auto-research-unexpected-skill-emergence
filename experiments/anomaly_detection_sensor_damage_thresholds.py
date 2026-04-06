import numpy as np
import time

def run_anomaly_experiment(n_agents=100, anomaly_rate=0.2, noise_level=0.05, iterations=30):
    """
    Investigates if agents can 'self-heal' by using a weighted consensus 
    mechanism that de-weights neighbors with high deviations (anomalies).
    """
    np.random.seed(42)
    # Initial values: all agents start near 0
    values = np.random.normal(0, 1, n_agents)
    
    # Introduce anomalies: a percentage of agents have 'broken' sensors 
    num_anomalies = int(n_agents * anomaly_rate)
    anomaly_indices = np.random.choice(n_agents, num_anomalies, replace=False)
    values[anomaly_indices] += 15.0 # Large anomalies
    
    initial_variance = np.var(values)
    current_values = values.copy()
    
    for _ in range(iterations):
        snapshot = current_values.copy()
        new_values = []
        # To make it computationally feasible, we use a global weighted approach 
        # simulating local perception of the population mean.
        
        # Calculate weights based on deviation from the population mean
        local_mu = np.mean(snapshot)
        diffs = np.abs(snapshot - local_mu)
        weights = 1.0 / (1.0 + diffs)
        
        weighted_mean = np.sum(snapshot * weights) / np.sum(weights)
        
        for i in range(n_agents):
            # Update rule: Move towards the weighted mean with noise
            noise = np.random.normal(0, noise_level)
            new_val = snapshot[i] + 0.5 * (weighted_mean - snapshot[i]) + noise
            new_values.append(new_val)
            
        current_values = np.array(new_values)

    final_variance = np.var(current_values)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: anomaly_detection_sensor_damage_thresholds")
    # Test different levels of noise in the consensus process to see when it breaks
    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]
    for nl in noise_levels:
        start_time = time.time()
        success = run_anomaly_experiment(noise_level=nl)
        end_time = time.time()
        print(f"noise_{nl:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()