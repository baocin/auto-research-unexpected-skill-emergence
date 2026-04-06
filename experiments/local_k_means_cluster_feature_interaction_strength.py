import numpy as np
import time

def run_interaction_strength_experiment(n_agents=100, interaction_strength=0.5, iterations=30):
    """
    Investigates 'Interaction Strength': How the weight of the consensus 
    step (the 'learning rate' or 'step size') affects convergence stability.
    We vary the strength from 0.01 (very slow) to 0.99 (very aggressive).
    """
    np.random.seed(42)
    # Two clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
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
                # The 'strength' controls how much the agent moves toward the centroid
                new_p = p_i + interaction_strength * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_interaction_strength")
    # Test different strengths of the update step
    strengths = [0.01, 0.1, 0.5, 0.9]
    for s in strengths:
        start_time = time.time()
        try:
            success = run_interaction_strength_experiment(interaction_strength=s)
            duration = time.time() - start_time
            print(f"strength_{s:.2f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{s}_{e}")

if __name__ == "__main__":
    run_experiment()