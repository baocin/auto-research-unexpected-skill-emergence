import numpy as np
import time

def run_experiment():
    """
    Tests if local median-based anomaly detection works when 
    the number of neighbors is very small (high sparsity).
    """
    num_sensors = 100
    num_damaged = 10
    # We will test different numbers of 'neighbors' available to each sensor
    neighbor_counts = [2, 5, 10, 20]
    
    true_signal = np.sin(np.linspace(0, 10, num_sensors))
    readings = true_signal + np.random.normal(0, 0.1, num_sensors)
    
    # Inject damage
    damaged_indices = np.random.choice(num_sensors, num_damaged, replace=False)
    readings[damaged_indices] += np.random.normal(0, 10.0, num_damaged)
    
    initial_mse = np.mean((readings - true_signal)**2)
    results = []

    for k in neighbor_counts:
        # For each sensor, we simulate having only 'k' neighbors within a window
        # We do this by randomly selecting k indices from the dataset
        current_readings = readings.copy()
        start_time = time.time()
        
        filtered_readings = np.zeros(num_sensors)
        for i in range(num_sensors):
            # Randomly pick k neighbors for this sensor
            neighbor_indices = np.random.choice(num_sensors, k, replace=False)
            neighbor_values = readings[neighbor_indices]
            
            # Median filter using only these k neighbors
            filtered_readings[i] = np.median(neighbor_values)
        
        end_time = time.time()
        final_mse = np.mean((filtered_readings - true_signal)**2)
        success_metric = 1.0 - (final_mse / initial_mse) if initial_mse > 0 else 0
        
        results.append((k, success_metric, end_time - start_time))

    print(f"experiment: anomaly_sparsity_test")
    for k, success, duration in results:
        print(f"neighbors_{k}_success_{success:.4f}_time_{duration:.4_f}")

if __name__ == "__main__":
    run_experiment()