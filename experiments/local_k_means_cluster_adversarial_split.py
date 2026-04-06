import numpy as np
import time

def run_adversarial_kmeans(n_samples=100, adversarial_ratio=0.2, iterations=30):
    """
    Investigates k-means where a subset of agents are 'adversarial' 
    by being frozen at specific coordinates (e.g., [5,5] and [-5,-5]).
    This tests if local clustering can maintain cluster identity 
    when certain nodes are anchored to contradictory positions.
    """
    np.random.seed(42)
    # Base clusters in a central region
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(2, 1, (n_samples // 2, 2))
    points = np.vstack([c1, c2])
    
    # Add adversarial anchors
    num_adv = int(n_samples * adversarial_ratio)
    adv_indices = []
    for i in range(num_adv):
        # Alternate between two distant poles
        pos = np.array([5.0, 5.0]) if i % 2 == 0 else np.array([-5.0, -5.0])
        points = np.vstack([points, pos])
        adv_indices.append(len(points) - 1)
    
    n_total = len(points)
    initial_variance = np.var(points)
    fixed_mask = np.zeros(n_total, dtype=bool)
    for idx in adv_indices:
        fixed_mask[idx] = True

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
            neighbors_mask = (dists <= 2.5) & (dists > 0)
            
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
    print(f"experiment: adversarial_k_means_anchors")
    # Test how increasing the number of fixed anchors disrupts cluster formation
    adversarial_ratios = [0.0, 0.1, 0.2, 0.4]
    
    for ratio in adversarial_ratios:
        start_time = time.time()
        success = run_adversarial_kmeans(adversarial_ratio=ratio)
        end_time = time.time()
        print(f"adv_ratio_{ratio:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()