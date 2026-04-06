import numpy as np
import time

def run_k_means_density_experiment(n_samples=100, density_factor=1.0, iterations=30):
    """
    Investigates how the initial density of points affects the ability 
    of local updates to form distinct clusters.
    Higher density_factor means points are more tightly packed in a smaller area.
    """
    np.random.seed(42)
    # Scale the domain by the density factor (higher factor = smaller domain = higher density)
    domain_size = 10.0 / density_factor
    points = np.random.rand(n_samples, 2) * domain_size
    initial_variance = np.var(points)
    
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            # Fixed interaction radius
            radius = 2.0
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= radius) & (dists > 0)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    # Success metric: reduction in variance (convergence to compact clusters/points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: local_k_means_density_impact")
    # density_factors: higher means more dense (smaller domain)
    density_factors = [0.5, 1.0, 2.0, 4.0]
    
    for df in density_factors:
        start_time = time.time()
        success = run_k_means_density_experiment(density_factor=df)
        end_time = time.time()
        print(f"density_factor_{df:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()