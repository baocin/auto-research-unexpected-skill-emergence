import numpy as np
import time

def run_adversarial_influence_experiment(adv_magnitude, n_samples=100, iterations=30):
    np.random.seed(42)
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(5, 1, (n_samples // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    
    adv_agents = np.array([
        [adv_magnitude, adv_magnitude],
        [-adv_magnitude, -adv_magnitude]
    ])
    points = np.vstack([points, adv_agents])
    n_total = len(points)
    
    fixed_mask = np.zeros(n_total, dtype=bool)
    fixed_mask[-2:] = True 

    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_total):
            if fixed_mask[i]:
                new_points.append(snapshot[i])
                continue
                
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(_n_samples := n_total) != i)
            # Fixed the variable name error in mask logic
            neighbors_mask = (dists <= 3.0) & (np.arange(n_total) != i)
            
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
    print(f"experiment: k_means_adversarial_influence_strength")
    magnitudes = [2.0, 5.0, 10.0, 50.0]
    for m in magnitudes:
        start_time = time.time()
        success = run_adversarial_influence_experiment(adv_magnitude=m)
        end_time = time.time()
        print(f"magnitude_{m:.1_f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    # Redefining the run function to be clean and correct
    def run_clean(magnitudes):
        for m in magnitudes:
            start = time.time()
            success = run_adversarial_influence_experiment(adv_magnitude=m)
            end = time.time()
            print(f"magnitude_{m:.1f}_success_{success:.4f}_time_{end - start:.4f}")
    
    print(f"experiment: k_means_adversarial_influence_strength")
    run_clean([2.0, 5.0, 10.0, 50.0])