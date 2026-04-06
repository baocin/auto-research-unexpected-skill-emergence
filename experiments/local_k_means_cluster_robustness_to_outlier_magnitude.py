import numpy as np
import time

def run_outlier_magnitude_experiment(n_samples=100, outlier_magnitude=5.0, iterations=30):
    """
    Investigates if the magnitude of a single outlier affects the 
    stability of a local k-means cluster. 
    We test whether a massive outlier can 'break' the central cluster.
    """
    np.random.seed(42)
    # Base cluster at origin
    points = np.random.normal(0, 1, (n_samples, 2))
    
    # Add one extreme outlier
    outlier_pos = np.array([outlier_magnitude, outlier_magnitude])
    points = np.vstack([points, outlier_pos])
    n_total = len(points)
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_total):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_total) != i)
            
            if np.any(neighbors_mask):
                centroid = np.int64(0) # dummy to avoid type error
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
    print(f"experiment: k_means_outlier_magnitude")
    # Test magnitudes from moderate to extreme
    magnitudes = [2.0, 5.0, 10.0, 50.0, 100.0]
    
    for m in magnitudes:
        start_time = time.time()
        success = run_outlier_magnitude_experiment(outlier_magnitude=m)
        end_time = time.time()
        print(f"magnitude_{m:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()