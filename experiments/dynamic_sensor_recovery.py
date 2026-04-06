import numpy as np
import time

def run_experiment():
    """
    Investigates 'Dynamic Sensor Recovery':
    Sensors provide a signal with noise. Some sensors are 'damaged' (extreme outliers).
    In each iteration, active nodes average their neighbors. 
    Additionally, if a node's value is too far from its local mean (an anomaly),
    it is 'recalibrated' to that local mean.
    We measure if the system can recover the true signal variance over time.
    """
    num_sensors = 100
    num_damaged = 20
    iterations = 30
    window_size = 3
    
    # 1. True signal: Sine wave
    positions = np.linspace(0, 10, num_sensors)
    true_signal = np.sin(positions)
    
    # 2. Initial readings (Healthy + Noise)
    readings = true_signal + np.random.normal(0, 0.1, num_sensors)
    
    # 3. Inject damage
    damaged_indices = np.random.choice(num_sensors, num_damaged, replace=False)
    readings[damaged_indices] += np.random.normal(0, 10.0, num_damaged)
    
    initial_mse = np.mean((readings - true_signal)**2)
    start_time = time.time()
    
    # 4. Simulation loop
    current_readings = readings.copy()
    for _ in range(iterations):
        new_readings = np.copy(current_readings)
        
        # Step A: Local Averaging (Consensus/Smoothing)
        for i in range(num_sensors):
            start_idx = max(0, i - window_size)
            end_idx = min(num_sensors, i + window_size + 1)
            new_readings[i] = np.mean(current_readings[start_idx:end_idx])
        
        # Step B: Anomaly-based Recalibration (The 'Recovery' mechanism)
        # If a sensor is too far from its neighborhood average, reset it.
        for i in range(num_sensors):
            start_idx = max(0, i - window_size)
            end_idx = min(num_sensors, i + window_size + 1)
            local_mean = np.mean(current_readings[start_idx:end_idx])
            
            # Check if the original reading was an outlier
            if abs(current_readings[i] - local_mean) > 2.0:
                new_readings[i] = local_mean
        
        current_readings = new_readings

    end_time = time.time()
    
    final_mse = np.mean((current_readings - true_signal)**2)
    
    # Success is the reduction in MSE relative to initial noisy/damaged state
    success_metric = 1.0 - (final_mse / initial_mse) if initial_mse > 0 else 0

    print(f"experiment: dynamic_sensor_recovery")
    print(f"initial_mse: {initial_mse:.4f}")
    print(f"final_mse: {final_mse:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()