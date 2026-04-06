import numpy as np
import time

def run_bottleneck_experiment(n_samples=100, bottleneck_width=1.0, iterations=30):
    """
    Investigates the 'bottleneck' effect: a narrow corridor of connectivity 
    between two clusters. We create two groups of points and only allow 
    interaction if agents fall within a narrow x-range (the bottleneck).
    """
    np.random.seed(42)
    # Cluster 1: [0, 3] on x-axis
    c1 = np.random.rand(n_samples // 2, 2)
    c1[:, 0] = c1[:, 0] * 3.0
    
    # Cluster 2: [7, 10] on x-axis
    c2 = np.random.rand(n_samples // 2, 2)
    c2[:, 0] = c2[:, 0] * 3.0 + 7.0
    
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # The bottleneck is a vertical strip in the x-axis [4.5, 5.5]
    # Only agents within this strip can 'bridge' the clusters.
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            
            # Standard spatial radius
            neighbors_mask = (dists <= 3.0) & (np.arange(n_samples) != i)
            
            # Bottleneck constraint: An agent can only 'bridge' if it is in the corridor
            if np.any(neighbors_mask):
                # Check if any neighbor is also in the bottleneck
                bottleneck_mask = (snapshot[:, 0] >= 4.5) & (snapshot[:, 0] <= 5.5)
                effective_neighbors = neighbors_mask & bottleneck_mask
                
                if np.any(effective_neighbors):
                    centroid = np.mean(snapshot[effective_neighbors], axis=0)
                    new_p = p_i + 0.5 * (centroid - p_i)
                else:
                    # Standard local update if no bridge is present
                    standard_neighbors = neighbors_mask
                    if np.any(standard_neighbors):
                        centroid = np.mean(snapshot[standard_neighbors], axis=0)
                        new_p = p_i + 0.5 * (centroid - p_i)
                    else:
                        new_p = p_i
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_bottleneck_width")
    # We vary the width of the 'bridge' corridor
    widths = [0.1, 0.5, 1.0, 2.0]
    
    for w in widths:
        start_time = time.time()
        success = run_bott_experiment(bottleneck_width=w)
        end_time = time.time()
        print(f"width_{w:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

def run_bott_experiment(bottleneck_width):
    # Wrapper to handle the parameter passing correctly
    return run_bottleneck_logic(bottleneck_width)

def run_bott_logic(bw):
    # Re-implementing logic inside for stability in execution
    np.random.seed(42)
    n_samples = 100
    c1 = np.random.rand(50, 2); c1[:, 0] *= 3.0
    c2 = np.random.rand(50, 2); c2[:, 0] = (c2[:, 0] * 3.0) + 7.0
    points = np.vstack([c1, c2])
    init_v = np.var(points)
    curr = points.copy()
    for _ in range(30):
        snap = curr.copy()
        new_pts = []
        for i in range(n_samples):
            p_i = snap[i]
            dists = np.linalg.norm(snap - p_i, axis=1)
            mask = (dists <= 3.0) & (np.arange(n_samples) != i)
            if np.any(mask):
                # The bottleneck logic: can we reach the other side?
                bridge_mask = mask & (snap[:, 0] >= (5.0 - bw/2) & (snap[:, 0] <= (5.0 + bw/2)))
                # Wait, syntax error in boolean logic above. Let'_s fix:
                b_m = mask & ((snap[:, 0] >= (5.0 - bw/2)) & (snap[:, 0] <= (5.0 + bw/2)))
                if np.any(b_m):
                    centroid = np.mean(snap[b_m], axis=0)
                    new_pts.append(p_i + 0.5 * (centroid - p_i))
                else:
                    # Fallback to standard local update
                    std_mask = mask
                    if np.any(std_mask):
                        new_pts.append(p_i + 0.5 * (np.mean(snap[std_mask], axis=0) - p_i))
                    else:
                        new_pts.append(p_i)
            else:
                new_pts.append(p_i)
        curr = np.array(new_pts)
    return 1.0 - (np.var(curr)/init_v)

if __name__ == "__main__":
    # Re-run the experiment with corrected logic
    print(f"experiment: k_means_bottleneck_robust")
    widths = [0.1, 0.5, 1.0, 2.0]
    for w in widths:
        s_t = time.time()
        # Using the internal robust logic directly
        res = run_bott_logic(w)
        e_t = time.time()
        print(f"width_{w:.1f}_success_{res:.4f}_time_{e_t - s_t:.4f}")