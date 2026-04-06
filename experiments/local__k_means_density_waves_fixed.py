import numpy as np
import time

def run_clean_experiment(amplitude, n_agents=100, iterations=30):
    np.random.seed(42)
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for step in range(iterations):
        wave_factor = 1.0 + amplitude * np.sin(step * 0.5)
        snapshot = current_points.copy()
        new_points = []
        
        # We must ensure wave_factor is not zero to avoid division by zero
        if abs(wave_factor) < 1e-6:
            wave_factor = 1e-6

        transformed_snapshot = snapshot * wave_factor
        
        for i in range(n_agents):
            p_i = transformed_snapshot[i]
            dists = np.linalg.norm(transformed_snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                centroid = np.mean(transformed_snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        
        current_points = np.array(new_points) / wave_factor

    final_variance = np.int64(0) # Placeholder to avoid error
    # Actually, let's just use the real variance calculation
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print("experiment: k_means_density_waves")
    amplitudes = [0.0, 0.5, 1.0, 2.0]
    for amp in amplitudes:
        start_time = time.time()
        try:
            success = run_clean_experiment(amplitude=amp)
            duration = time_val := time.time() - start_time
            print(f"amplitude_{amp:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{amp}_{e}")

if __name__ == "__main__":
    run_experiment()