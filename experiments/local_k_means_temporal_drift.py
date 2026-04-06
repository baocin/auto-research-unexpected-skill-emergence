import numpy as np
import time

def run_temporal_drift_kmeans(points, iterations=20, radius=2.5, drift_rate=0.1):
    """
    Investigates 'Temporal Drift K-Means':
    The positions of the clusters themselves slowly drift over time/iterations.
    This tests if local updates can track a moving target distribution.
    """
    current_points = points.copy()
    
    for iter_idx in range(iterations):
        # Apply drift to the cluster centers (simulated by shifting all points)
        drift_vector = np.array([drift_rate, drift_rate])
        current_points += drift_vector * 0.1 
        
        new_points = []
        snapshot_points = current_points.copy()
        
        for i in range(len(current_points)):
            p = current_points[i]
            distances = np.linalg.norm(snapshot_points - p, axis=1)
            neighbors_mask = (distances <= radius) & (distances > 0)
            neighbors = snapshot_points[neighbors_mask]
            
            if len(neighbors) > 0:
                centroid = np.mean(neighbors, axis=0)
                direction = centroid - p
                new_p = p + 0.5 * direction
                new_points.append(new_p)
            else:
                new_points.append(p)
        current_points = np.array(new_points)
        
    return current_points

def run_experiment():
    np.random.seed(42)
    # Initial clusters
    c1 = np.random.randn(30, 2) + np.array([0, 0])
    c2 = np.random.randn(30, 2) + np.array([5, 5])
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    
    # Test different drift rates
    drift_rates = [0.0, 0.1, 0.5]
    
    print(f"experiment: temporal_drift_k_means")
    for dr in drift_rates:
        start_time = time.time()
        final_points = run_temporal_drift_kmeans(data, iterations=20, radius=3.0, drift_rate=dr)
        end_t = time.time()
        
        final_variance = np.var(final_points)
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"drift_{dr:.1f}_success_{success_metric:.4f}_time_{end_t - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()