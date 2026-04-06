import numpy as np
import time

def run_sparsity_experiment(sparsity_rate, n_agents=100, iterations=30):
    """
    Investigates 'Feature Sparsity': How the presence of 'empty' or 
    zero-value features in agent data affects clustering stability.
    """
    np.random.seed(42)
    # Two clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # Sparsity: Randomly zero out components of the feature vectors
    # We apply this at each step to simulate sensor uncertainty/dropout
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        
        # Apply sparsity mask (dropout)
        mask = np.random.rand(*snapshot.shape) > sparsity_rate
        masked_snapshot = snapshot * mask
        
        for i in range(n_agents):
            p_i = masked_snapshot[i]
            dists = np.linalg.norm(masked_snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                centroid = np.mean(masked_snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_feature_sparsity")
    sparsity_rates = [0.0, 0.3, 0.6, 0.9]
    for sr in sparsity_rates:
        start_time = time.time()
        success = run_sparsity_experiment(sparsity_rate=sr)
        end_time = time.time()
        print(f"sparsity_{sr:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()