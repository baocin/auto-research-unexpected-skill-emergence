import numpy as np
import time

def run_density_experiment(density_factor, n_agents=100, iterations=30):
    np.random.seed(42)
    # High density factor means smaller spread (more concentrated)
    spread = max(0.01, 1.0 - density_factor)
    points = np.random.uniform(0, spread, (n_agents, 2))
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 0.5) & (np.arange(n_agents) != i)
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
    print(f"experiment: k_means_density_connectivity")
    # density_factor: high value means agents are very close together initially
    densities = [0.9, 0.7, 0.5, 0.1]
    for d in densities:
        start_time = time.time()
        success = run_density_experiment(density_factor=d)
        end_time = time.time()
        print(f"density_{d:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()