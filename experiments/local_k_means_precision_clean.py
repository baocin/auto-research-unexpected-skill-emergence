import numpy as np
import time

def run_precision_experiment(epsilon, n_agents=100, iterations=30):
    np.random.seed(42)
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(1, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                error = np.random.uniform(-epsilon, epsilon, 2)
                new_p = p_i + 0.5 * (centroid - p_i) + error
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    return 1.0 - (final_variance / initial_variance)

def run_experiment():
    print(f"experiment: k_means_precision_limit")
    epsilons = [1e-1, 1e-3, 1e-7, 1e-15]
    for eps in epsilons:
        start_time = time.time()
        success = run_precision_experiment(epsilon=eps)
        duration = time.time() - start_time
        print(f"epsilon_{eps:.1e}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()