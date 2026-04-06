import numpy as np
import time

def run_hierarchical_kmeans(n_samples=100, radius_range=[1.0, 2.5, 5.0], iterations=30):
    """
    Investigates if a multi-scale interaction approach (varying radii) 
    can achieve better convergence than a single fixed radius.
    """
    np.random.seed(42)
    # Generate two distinct clusters
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(5, 1, (n_samples // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    
    results = []

    for r in radius_range:
        current_points = points.copy()
        for _ in range(iterations):
            snapshot = current_points.copy()
            new_pts = []
            for i in range(n_samples):
                p_i = snapshot[i]
                dists = np.linalg.norm(snapshot - p_i, axis=1)
                neighbors_mask = (dists <= r) & (dists > 0)
                if np.any(neighbors_mask):
                    centroid = np.mean(snapshot[neighbors_mask], axis=0)
                    new_pts.append(p_i + 0.5 * (centroid - p_i))
                else:
                    new_pts.append(p_i)
            current_points = np.array(new_pts)
        
        final_variance = np.var(current_points)
        success = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
        results.append((r, success))
    
    return results

def run_experiment():
    print(f"experiment: hierarchical_k_means_radius_sensitivity")
    radii = [0.5, 2.0, 5.0]
    start_time = time.time()
    # Fixed function name call
    results = run_hierarchical_kmeans(radius_range=radii)
    end_time = time.time()
    
    for r, success in results:
        print(f"radius_{r:.1f}_success_{success:.4f}")
    print(f"total_seconds:{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()