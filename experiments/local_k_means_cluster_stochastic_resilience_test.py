import numpy as np
import time

def run_resilience_experiment(n_agents=100, noise_intensity=0.5, iterations=30):
    """
    Investigates 'Stochastic Resilience': How the system's ability to 
    converge is affected by increasing levels of white noise in the 
    positional updates (simulating environmental turbulence).
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
        
        # Add white noise to the snapshot (environmental turbulence)
        noise = np.random.normal(0, noise_intensity, snapshot.shape)
        noisy_snapshot = snapshot + noise
        
        for i in range(n_agents):
            p_i = snapshot[i] # Use original position for reference
            dists = np.linalg.norm(noisy_snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                # Average the noisy positions of neighbors
                centroid = np.mean(noisy_snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_stochastic_resilience")
    # Test increasing levels of environmental noise
    noise_levels = [0.0, 0.2, 0.5, 1.0]
    for nl in noise_levels:
        start_time = time.time()
        try:
            success = run_resilience_experiment(noise_intensity=nl)
            duration = time.time() - start_time
            print(f"noise_{nl:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{nl}_{e}")

if __name__ == "__main__":
    run_experiment()