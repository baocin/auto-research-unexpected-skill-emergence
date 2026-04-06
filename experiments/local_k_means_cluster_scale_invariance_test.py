import numpy as np
import time

def run_scale_invariance_experiment(scale_factor, n_agents=100, iterations=30):
    """
    Investigates 'Scale Invariance': Does the algorithm's ability to converge 
    remain constant when we scale the entire universe (positions and radius) 
    by the same factor? If scale-invariant, success should be identical.
    """
    np.random.seed(42)
    # Base configuration: Two clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c_array for c_array in [c1, c2]])
    
    # Apply scale factor to positions
    points = points * scale_factor
    
    # The interaction radius must also scale to maintain relative topology
    interaction_radius = 3.0 * scale_factor
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            # Distance calculation in the scaled space
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= interaction_radius) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    # Success is the reduction of variance relative to the scaled initial variance
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_scale_invariance")
    # Test different scales of the universe
    scales = [0.1, 1.0, 10.0, 100.0]
    for s in scales:
        start_time = time.time()
        success = run_scale_int_experiment(scale_factor=s)
        end_time = time.time()
        print(f"scale_{s:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

# Fix the function name mismatch for execution
def run_experiment_final():
    print(f"experiment: k_means_scale_invariance")
    scales = [0.1, 1.0, 10.0, 100.0]
    for s in scales:
        start_time = time.time()
        # Call the correct function name defined above
        success = run_scale_invariance_experiment(scale_factor=s)
        end_duration = time.time() - start_time
        print(f"scale_{s:.1f}_success_{success:.4f}_time_{start_time - start_time:.4f}")

# Let's just use a single, clean execution path to avoid any confusion
if __name__ == "__main__":
    print(f"experiment: k_means_scale_invariance")
    scales = [0.1, 1.0, 10.0, 100.0]
    for s in scales:
        start = time.time()
        # The real logic is inside the function defined at top
        import sys
        # Re-importing or re-defining to ensure scope
        def run_s(scale):
            np.random.seed(42)
            n = 100
            c1 = np.random.normal(0, 1, (n // 2, 2))
            c2 = np.random.normal(5, 1, (n // 2, 2))
            pts = np.vstack([c1, c2]) * scale
            iv = np.var(pts)
            curr = pts.copy()
            r = 3.0 * scale
            for _ in range(30):
                snap = curr.copy()
                new_p = []
                for i in range(n):
                    dists = np.linalg.norm(snap - snap[i], axis=1)
                    mask = (dists <= r) & (np.arange(n) != i)
                    if np.any(mask):
                        cen = np.mean(snap[mask], axis=0)
                        new_p.append(snap[i] + 0.5 * (cen - snap[i]))
                    else:
                        new_p.append(snap[i])
                curr = np.array(new_p)
            return 1.0 - (np.var(curr)/iv)
        
        res = run_s(s)
        dur = time.time() - start
        print(f"scale_{s:.1f}_success_{res:.4f}_time_{dur:.4f}")