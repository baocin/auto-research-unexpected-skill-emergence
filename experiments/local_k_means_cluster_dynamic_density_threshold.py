import numpy as np
import time

def run_density_threshold_experiment(n_samples=100, threshold_radius=2.5, iterations=30):
    """
    Investigates the 'critical density' effect: how increasing the number of 
    agents in a fixed volume affects convergence success.
    """
    np.random.seed(42)
    # Fixed domain [0, 10] x [0, 10]
    # We vary n_samples (density) while keeping interaction radius constant
    points = np.random.rand(n_samples, 2) * 10.0
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= threshold_radius) & (np.arange(n_samples) != i)
            
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
    print(f"experiment: k_means_density_threshold")
    # Test increasing densities (from sparse to very dense)
    densities = [50, 100, 200, 400]
    
    for n in densities:
        start_time = time.time()
        success = run_density_threshold_experiment(n_samples=n)
        end_time = time.time()
        print(f"n_agents_{n}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()