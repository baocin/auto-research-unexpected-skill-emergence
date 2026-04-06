import numpy as np
import time
import sys

def run_fractal_dimension_experiment(fractal_dim, n_samples=100, iterations=30):
    """
    Investigates how the 'fractal dimension' (complexity) of a cluster distribution
    affects the convergence speed and success of local k-means updates.
    """
    try:
        np.random.seed(42)
        # Generate points in [0, 1] domain
        u = np.random.rand(n_samples, 2)
        # Apply fractal transformation (power law)
        # Higher dim -> closer to uniform; Lower dim -> more concentrated near origin
        points = np.power(u, 1.0 / fractal_dim) * 10.0
        
        initial_variance = np.var(points)
        current_points = points.copy()
        
        for _ in range(iterations):
            new_points = []
            snapshot = current_points.copy()
            for i in range(n_samples):
                p_i = snapshot[i]
                dists = np.linalg.norm(snapshot - p_i, axis=1)
                # Interaction radius of 2.5
                neighbors_mask = (dists <= 2.5) & (int(i) != 0) # dummy check
                # Let's use a proper mask
                neighbors_mask = (dists <= 2.5) & (np.arange(n_samples) != i)
                
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
    except Exception as e:
        print(f"Error in simulation: {e}", file=sys.stderr)
        return 0.0

def run_experiment():
    print(f"experiment: k_means_fractal_dimension")
    fractal_dims = [1.0, 1.2, 1.5, 1.8, 2.0]
    
    for fd in fractal_dims:
        start_time = time.time()
        success = run_fractal_dimension_experiment(fractal_dim=fd)
        end_time = time.time()
        print(f"dim_{fd:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()