import numpy as np
import time

def run_heterogeneous_density_experiment(n_samples=100, cluster_separation=5.0, iterations=30):
    """
    Investigates how the density difference between two clusters 
    affects convergence success. One cluster is dense, the other sparse.
    """
    np.random.seed(42)
    # Cluster 1: Dense (many points in a small area)
    c1_size = int(n_samples * 0.8)
    c1 = np.random.normal(0, 0.5, (c1_size, 2))
    
    # Cluster 2: Sparse (few points spread over a larger area)
    c2_size = n_samples - c1_size
    c2 = np.random.normal(cluster_separation, 2.0, (c2_size, 2))
    
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (dists > 0)
            
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
    print(f"experiment: k_means_heterogeneous_density")
    # Test varying the 'imbalance' of the clusters
    # We vary the size ratio of the dense vs sparse cluster
    ratios = [0.5, 0.7, 0.8, 0.9]
    n_total = 100
    
    for r in ratios:
        start_time = time.time()
        # Re-implementing logic inside to control n_samples precisely
        np.random.seed(42)
        c1_size = int(n_total * r)
        c2_size = n_total - c1_size
        c1 = np.random.normal(0, 0.5, (c1_size, 2))
        c2 = np.random.normal(5.0, 2.0, (c2_size, 2))
        points = np.vstack([c1, c2])
        
        # Run simulation
        current_p = points.copy()
        for _ in range(30):
            snap = current_p.copy()
            new_pts = []
            for i in range(n_total):
                dists = np.linalg.norm(snap - snap[i], axis=1)
                mask = (dists <= 3.0) & (np.arange(n_total) != i)
                if np.any(mask):
                    new_pts.append(snap[i] + 0.5 * (np.mean(snap[mask], axis=0) - snap[i]))
                else:
                    new_pts.append(snap[i])
            current_p = np.array(new_pts)
        
        final_v = np.var(current_p)
        init_v = np.var(points)
        success = 1.0 - (final_v / init_v) if init_v > 0 else 0
        end_time = time.time()
        print(f"ratio_{r:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()