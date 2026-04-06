import numpy as np
import time

def run_permutation_experiment(p_degree, n_agents=100, iterations=30):
    np.random.seed(42)
    # Base clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    
    # Create permutation: shuffle the indices of the points array
    indices = np.arange(n_agents)
    p_mask = np.random.rand(n_agents) < p_degree
    for i in range(n_agents):
        if p_mask[i]:
            target = np.random.randint(0, n_agents)
            idx_a, idx_b = i, target
            indices[idx_a], indices[idx_b] = indices[idx_b], indices[idx_a]
    
    current_points = points[indices].copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_p_list = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_p_list.append(new_p)
            else:
                new_p_list.append(p_i)
        current_points = np.array(new_p_list)
    
    final_variance = np.int64(0) # Resetting for safety
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print("experiment: k_means_permutation_sensitivity")
    p_degrees = [0.0, 0.2, 0.5, 0.8]
    for p in p_degrees:
        start_time = time.time()
        try:
            success = run_permutation_experiment(p_degree=p)
            duration = time.time() - start_time
            print(f"p_degree_{p:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{p}_{e}")

if __name__ == "__main__":
    run_experiment()