import numpy as np
import time

def run_noise_threshold_experiment(n_samples=100, noise_level=0.5, iterations=30):
    """
    Investigates the threshold at which sensor noise overcomes 
    the structural signal of a local k-means algorithm.
    We measure how much 'true' cluster information is lost.
    """
    np.random.seed(42)
    # Create two distinct, well-separated clusters
    c1 = np.random.normal(0, 0.5, (n_samples // 2, 2))
    c2 = np.random.normal(5, 0.5, (n_samples // 2, 2))
    points = np.vstack([c1, c2])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # Pre-calculate noise for each agent to ensure consistent 'sensor' behavior per run
    noise_mask = np.random.rand(n_samples) < noise_level
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 2.5) & (dists > 0)
            
            if np.any(neighbors_mask):
                if noise_mask[i]:
                    # Sensor error: reports a random point in the domain
                    obs = np.random.uniform(-5, 10, size=(2,))
                else:
                    # Healthy sensor: reports local mean
                    obs = np.mean(snapshot[neighbors_mask], axis=0)
                
                new_p = p_i + 0.5 * (obs - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    # Success is the reduction of global variance (convergence to clusters)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: local_k_means_noise_threshold")
    # Test noise levels from 0% to 80%
    noise_rates = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    
    for rate in noise_rates:
        start_time = time.time()
        success = run_noise_threshold_experiment(noise_level=rate)
        end_time = time.time()
        print(f"noise_{rate:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()