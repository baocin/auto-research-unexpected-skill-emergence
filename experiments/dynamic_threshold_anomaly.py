import numpy as np
import time

def run_experiment():
    """
    Investigates 'Dynamic Threshold Anomaly Detection':
    Each sensor calculates its own anomaly threshold based on 
    the local standard deviation of its neighbors.
    We measure if this adaptive approach reduces false positives/negatives.
    """
    num_sensors = 100
    num_damaged = 15
    window_size = 4
    
    # 1. True signal: Sine wave
    positions = np.linspace(0, 10, num_sensors)
    true_signal = np.sin(positions)
    
    # 2. Initial readings (Healthy + Noise)
    readings = true_signal + np.random.normal(0, 0.1, num_sensors)
    
    # 3. Inject damage (High variance spikes)
    damaged_indices = np.random.choice(num_sensors, num_damaged, replace=False)
    readings[damaged_indices] += np.random.normal(0, 8.0, num_damaged)
    
    initial_mse = np.mean((readings - true_signal)**2)
    start_time = time.time()
    
    # 4. Adaptive Detection
    detected_anomalies = []
    filtered_readings = np.copy(readings)
    
    for i in range(num_sensors):
        # Windowed neighbors
        start_idx = max(0, i - window_size)
        end_idx = min(num_sensors, i + window_size + 1)
        neighbor_indices = [j for j in range(start_idx, end_idx) if j != i]
        
        if neighbor_indices:
            neighbor_values = readings[neighbor_indices]
            local_std = np.std(neighbor_values) + 1e-6
            local_mean = np.mean(neighbor_values)
            
            # Adaptive Threshold: Z-score relative to local standard deviation
            z_score = abs(readings[i] - local_mean) / local_std
            
            if z_score > 3.0:
                # This is an anomaly; we 'repair' it by setting it to the mean
                detected_anomalies.append(i)
                filtered_readings[i] = local_mean
        
    end_time = time.time()
    
    final_mse = np.mean((filtered_readings - true_signal)**2)
    success_metric = 1.0 - (final_mse / initial_mse) if initial_mse > 0 else 0

    print(f"experiment: dynamic_threshold_anomaly")
    print(f"initial_mse: {initial_mse:.4f}")
    print(f"final_mse: {final_mse:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()