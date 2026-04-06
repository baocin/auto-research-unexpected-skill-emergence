import numpy as np
import time

def run_experiment():
    """
    Investigates 'Anomaly Detection with Damaged Sensors':
    A set of sensor nodes measure a true signal (a sine wave).
    Some sensors are 'healthy' and report the signal + small noise.
    Some sensors are 'damaged' and report high-variance white noise.
    We test if local median filtering can recover the signal strength.
    """
    num_sensors = 200
    num_damaged = 40
    signal_freq = 0.1
    
    # 1. Create sensor positions (on a 1D line)
    positions = np.linspace(0, 10, num_sensors)
    
    # 2. True signal: sine wave
    true_signal = np.sin(signal_freq * positions)
    
    # 3. Sensor readings
    readings = true_signal + np.random.normal(0, 0.1, num_sensors) # Healthy noise
    
    # 4. Inject damage (high variance noise in random sensors)
    damaged_indices = np.random.choice(num_sensors, num_damaged, replace=False)
    readings[damaged_indices] += np.random.normal(0, 5.0, num_damaged)
    
    # 5. Local filtering algorithm: Median filter (robust to outliers/damage)
    start_time = time.time()
    filtered_readings = np.copy(readings)
    window_size = 5
    half_win = window_size // 2
    
    for i in range(num_sensors):
        start_idx = max(0, i - half_win)
        end_idx = min(num_sensors, i + half_win + 1)
        # Median is robust to the 'damaged' outliers
        filtered_readings[i] = np.median(readings[start_idx:end_idx])
    
    end_time = time.time()
    
    # 6. Evaluate effectiveness
    # Success metric: Reduction in MSE (Mean Squared Error) relative to the true signal
    initial_mse = np.mean((readings - true_signal)**2)
    final_mse = np.mean((filtered_readings - true_signal)**2)
    
    success_metric = 1.0 - (final_mse / initial_mse) if initial_mse > 0 else 0

    print(f"experiment: anomaly_damaged_sensors")
    print(f"initial_mse: {initial_mse:.4f}")
    print(f"final_mse: {final_mse:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()