import numpy as np
import time

def run_drift_experiment(n_samples=100, drift_rate=0.5, iterations=30):
    """
    Investigates 'Topology Drift': how the gradual movement of 
    cluster centers relative to each other affects convergence success.
    """
    np.random.seed(42)
    # Initial positions: Two clusters at [0,0] and [5,5]
    c1_base = np.random.normal(0, 1, (n_samples // 2, 2))
    c2_base = np.random.normal(5, 1, (n_samples // 2, 2))
    points = np.vstack([c1_base, c2_base])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # Drift trajectory: Cluster 2 moves away from Cluster 1 over time
    drift_vector = np.array([drift_rate, drift_rate])

    for step in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        
        # Apply drift to the points belonging to the second cluster
        # We simulate this by shifting the 'c2' part of the array every step
        drift_offset = drift_vector * (step / iterations)
        
        for i in range(n_samples):
            p_i = snapshot[i]
            # Check if it belongs to C2 (approximate via original index/logic)
            is_c2 = (i >= n_samples // 2)
            if is_c2:
                p_i = p_i + drift_offset
            
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_samples) != i)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_topology_drift")
    # Test different drift rates (how fast clusters move apart)
    drift_rates = [0.1, 1.0, 5.0]
    
    for dr in drift_rates:
        start_time = time.time()
        success = run_drift_experiment(drift_rate=dr)
        end_time = time.time()
        print(f"drift_rate_{dr:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()