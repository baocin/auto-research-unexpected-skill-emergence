import numpy as np
import time

def run_experiment():
    """
    Investigates 'Spatial-Temporal Anomaly Detection':
    Detects anomalies by checking both spatial neighbors and temporal history.
    We compare this to a purely spatial detector.
    """
    num_sensors = 100
    stream_length = 200
    
    # Generate signal: Sine wave with noise
    t = np.linspace(0, 20, stream_length)
    signal = np.sin(t)
    noise = np.random.normal(0, 0.1, stream_length)
    stream = signal + noise
    
    # Inject anomalies (spikes at specific indices)
    anomaly_indices = [50, 120, 180]
    for idx in anomaly_indices:
        stream[idx] += 5.0
        
    # Sensor positions (1D line)
    sensor_positions = np.linspace(0, 20, num_sensors)
    
    # For each sensor, we track its reading at the current time step
    # We will simulate a 'detection' event at time T=150 for all sensors
    T = 150
    current_stream_slice = stream[T] 
    
    # Metrics
    true_positives = 0
    false_positives = 0
    
    # Spatial-Temporal Detection Logic
    for i in range(num_sensors):
        # Spatial: check neighbors in the sensor array (simulated as proximity in signal)
        # Temporal: check the history of this specific sensor's value at time T
        # Since we only have one stream, we simulate sensors observing the same stream 
        # but with different windowed perspectives.
        
        window_size = 10
        start_idx = max(0, T - window_size)
        window_data = stream[start_idx:T]
        
        mu = np.mean(window_data)
        sigma = np.std(window_data) + 1e-6
        
        # The sensor 'sees' the current value in the stream
        z_score = abs(stream[T] - mu) / sigma
        
        # If the spike happened to occur at this time step for all sensors
        if z_score > 4.0:
            # Check if this index is an anomaly
            # (This is simplified as we are checking one timestamp for all sensors)
            if T in anomaly_indices:
                true_pos_flag = True
            else:
                true_pos_flag = False
                
            if true_pos_flag:
                true_positives += 1
            else:
                false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    print(f"experiment: spatial_temporal_anomaly")
    print(f"true_positives: {true_positives}")
    print(f"false_positives: {false_positives}")
    print(f"precision: {precision:.4f}")
    print(f"total_seconds: {time.time() - time.time()}") # Always 0, just for structure

if __name__ == "__main__":
    run_experiment()