import numpy as np
import time

def run_outlier_density_experiment(n_base, outlier_ratio, radius=2.5, iterations=20):
    """
    Investigates how the density of outliers affects structural stability.
    """
    # Base cluster
    c1 = np.random.randn(n_base, 2) + np.array([0, 0])
    
    num_outliers = int(n_base * outlier_ratio)
    if num_outliers > 0:
        # Outliers are placed in a distant region
        c_outliers = np.random.uniform(15, 20, size=(num_outliers, 2))
        data = np.vstack([c1, c_outliers])
    else:
        data = c1.copy()
        
    initial_variance = np.var(data)
    current_points = data.copy()
    
    for _ in range(iterations):
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
        
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    n_base = 100
    outlier_ratios = [0.01, 0.1, 0.2, 0.4]
    
    print(f"experiment: outlier_density_impact")
    for r_val in outlier_ratios:
        start_time = time.time()
        success = run_outlier_density_experiment(n_base, r_val)
        end_time = time.time()
        print(f"ratio_{r_val:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()