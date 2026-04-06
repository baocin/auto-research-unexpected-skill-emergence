import numpy as np
import time

def run_temperature_experiment(n_samples=100, temp=1.0, iterations=30):
    """
    Investigates the effect of 'temperature' (stochasticity) on cluster convergence.
    Higher temperature means more random jitter in each local update.
    """
    np.random.seed(42)
    # Create two well-separated clusters
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(5, 1, (n_samples // 2, 2))
    points = np.vstack([c1, c2])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 2.5) & (dists > 0)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                # Deterministic part: move towards centroid
                deterministic_move = p_i + 0.5 * (centroid - p_i)
                # Stochastic part: temperature-scaled jitter
                jitter = np.random.normal(0, temp, size=(2,))
                new_p = deterministic_move + jitter
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    # Success is the reduction of variance (convergence to clusters)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_temperature_impact")
    # Test temperatures from freezing (0) to high heat (2.0)
    temperatures = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    for t in temperatures:
        start_time = time.time()
        success = run_temperature_experiment(temp=t)
        end_time = time.time()
        print(f"temp_{t:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()