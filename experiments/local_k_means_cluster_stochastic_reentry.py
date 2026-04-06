import numpy as np
import time

def run_reentry_experiment(n_samples=100, reentry_prob=0.2, iterations=30):
    """
    Investigates the 'Reentry' effect: a mechanism where agents occasionally 
    'jump' between two distant clusters. This tests if stochastic jumps can 
    act as a bridge to overcome topological isolation in local algorithms.
    """
    np.random.seed(42)
    # Two well-separated clusters
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(8, 1, (n_samples // 2, 2))
    points = np.vstack([c_p for c_p in [c1, c2]])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            
            # Stochastic Reentry: with probability p, the agent 'jumps' 
            # to a random position in the other cluster's domain
            if np.random.rand() < reentry_prob:
                # Jump to a point near the opposite cluster center
                target_center = np.array([8.0, 8.0]) if p_i[0] < 4 else np.array([0.0, 0.0])
                p_jump = target_center + np.random.normal(0, 1, size=(2,))
                new_points.append(p_jump)
            else:
                # Standard local update rule
                dists = np.linalg_norm_calc(snapshot, p_i) # Using a helper to avoid errors
                neighbors_mask = (dists <= 3.0) & (np.arange(n_samples) != i)
                
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

def np_linalg_norm_calc(snapshot, p_i):
    return np.linalg.norm(snapshot - p_i, axis=1)

# Re-implementing the logic cleanly in one function to avoid NameErrors
def run_clean_simulation(reentry_prob, n_samples=100, iterations=30):
    np.random.seed(4_2)
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(8, 1, (n_samples // 2, 2))
    points = np.vstack([c1, c2])
    init_v = np.var(points)
    curr = points.copy()
    
    for _ in range(iterations):
        snap = curr.copy()
        new_pts = []
        for i in range(n_samples):
            p_i = snap[i]
            # Reentry logic
            if np.random.rand() < reentry_prob:
                target = np.array([8.0, 8.0]) if p_i[0] < 4 else np.array([0.0, 0.0])
                new_pts.append(target + np.random.normal(0, 0.5, size=(2,)))
            else:
                dists = np.linalg.norm(snap - p_i, axis=1)
                mask = (dists <= 3.0) & (np.arange(n_samples) != i)
                if np.any(mask):
                    new_pts.append(p_i + 0.5 * (np.mean(snap[mask], axis=0) - p_i))
                else:
                    new_pts.append(p_i)
        curr = np.array(new_pts)
    return 1.0 - (np.var(curr)/init_v)

def run_experiment():
    print(f"experiment: k_means_stochastic_reentry")
    probs = [0.0, 0.05, 0.1, 0.2, 0.4]
    for p in probs:
        start = time.time()
        success = run_clean_simulation(p)
        end = time.time()
        print(f"reentry_prob_{p:.2f}_success_{success:.4f}_time_{end - start:.4f}")

if __name__ == "__main__":
    run_experiment()