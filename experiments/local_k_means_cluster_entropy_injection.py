import numpy as np
import time

def run_entropy_experiment(n_agents=100, entropy_rate=0.05, iterations=30):
    """
    Investigates 'Entropy Injection': The effect of randomly perturbing 
    agent positions during the clustering process.
    This simulates a highly dynamic or unstable environment.
    """
    np.random.seed(42)
    # Two clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        # Inject entropy: random displacement of agents
        if entropy_rate > 0:
            entropy_mask = np.random.rand(*current_points.shape) < entropy_rate
            current_points[entropy_mask] += np.random.normal(0, 2.0, current_points[entropy_mask].shape)

        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
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
    print(f"experiment: k_means_entropy_injection")
    # Test different entropy rates (probability of an agent being displaced)
    entropy_rates = [0.0, 0.05, 
                     0.1, 0.2]
    for er in entropy_rates:
        start_time = time.time()
        success = run_entropy_experiment(entropy_rate=er)
        end_time = time.time()
        print(f"entropy_{er:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()