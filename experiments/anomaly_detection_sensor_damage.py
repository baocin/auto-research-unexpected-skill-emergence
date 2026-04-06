import numpy as np
import time

def run_anomaly_experiment(n_agents=100, anomaly_rate=0.1, anomaly_magnitude=10.0, iterations=30):
    """
    Investigates if the consensus algorithm can 'self-heal' by 
    effectively ignoring agents with corrupted (anomalous) sensor data.
    """
    np.random.seed(42)
    # Initial values: all agents start near 0
    values = np.random.normal(0, 1, n_agents)
    
    # Introduce anomalies: a percentage of agents have 'broken' sensors 
    # that report extreme values.
    num_anomalies = int(n_agents * anomaly_rate)
    anomaly_indices = np.random.choice(n_agents, num_anomalies, replace=False)
    values[anomaly_indices] += anomaly_magnitude
    
    initial_variance = np.var(values)
    current_values = values.copy()
    
    for _ in range(iterations):
        new_values = current_values.copy()
        for i in range(n_agents):
            # The 'sensor' logic: agents look at neighbors. 
            # If an agent is 'damaged', it might broadcast wrong info, 
            # but the algorithm's ability to filter depends on the update rule.
            
            # We use a robust mean (Median) vs standard Mean to test filtering.
            # To make it interesting, we will compare two different 'brains':
            # 1. Standard Mean (Sensitive to outliers)
            # 2. Median (Robust to outliers)
            pass

    # Let's rewrite the loop more cleanly to actually perform the comparison
    return 0 # Placeholder for structure

def run_robustness_comparison(n_agents=100, anomaly_rate=0.1, iterations=30):
    """
    Compares Mean-based consensus vs Median-based consensus in the presence of anomalies.
    """
    np.random.seed(42)
    
    def simulate(method='mean'):
        values = np.random.normal(0, 1, n_agents)
        num_anomalies = int(n_args := n_agents * anomaly_rate)
        anomaly_indices = np.random.choice(n_agents, num_anomalies, replace=False)
        values[anomaly_indices] += 10.0 # Large anomalies
        
        initial_variance = np.var(values)
        current_values = values.copy()
        
        for _ in range(iterations):
            snapshot = current_values.copy()
            new_vals = []
            for i in range(n_agents):
                if method == 'mean':
                    target = np.mean(snapshot)
                else: # median
                    target = np.median(snapshot)
                
                # Local update
                new_val = snapshot[i] + 0.5 * (target - snapshot[i])
                new_vals.append(new_val)
            current_values = np.array(new_vals)
        
        final_variance = np.mean((current_values - np.mean(current_values))**2) # Simplified variance
        # We use the spread relative to initial spread
        return 1.0 - (np.var(current_values) / initial_variance)

    results = {}
    for m in ['mean', 'median']:
        start = time.time()
        success = simulate(method=m)
        end = time.time()
        results[f"method_{m}"] = (success, end - start)
    return results

def run_experiment():
    print(f"experiment: anomaly_detection_sensor_damage")
    # Test different levels of anomaly density
    rates = [0.01, 0.1, 0.3, 0.5]
    for r in rates:
        start_time = time.time()
        res = run_robustness_comparison(anomaly_rate=r)
        end_time = time.time()
        # Print results for both methods
        for method, (success, duration) in res.items():
            print(f"rate_{r:.2f}_{method}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()