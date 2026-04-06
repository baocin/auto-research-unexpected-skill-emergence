import numpy as np
import time

def run_som_vs_kmeans(n_samples=100, iterations=30, learning_rate=0.5):
    """
    Compares a standard local k-means update with a simplified 
    Self-Organizing Map (SOM) style topological update.
    The SOM version uses a fixed topology to pull neighbors towards a feature.
    """
    np.random.seed(42)
    # Create two clusters
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(5, 1, (n_samples // 2, 2))
    points = np.vstack([c_p for c_p in [c1, c2]])
    initial_variance = np.var(points)
    
    # --- K-Means Local Update ---
    km_points = points.copy()
    for _ in range(iterations):
        snap = km_points.copy()
        new_pts = []
        for i in range(n_samples):
            p_i = snap[i]
            dists = np.linalg.norm(snap - p_i, axis=1)
            mask = (dists <= 3.0) & (np.arange(n_samples) != i)
            if np.any(mask):
                centroid = np.mean(snap[mask], axis=0)
                new_pts.append(p_i + 0.5 * (centroid - p_i))
            else:
                new_pts.append(p_i)
        km_points = np.array(new_pts)
    km_success = 1.0 - (np.var(km_points) / initial_variance)

    # --- SOM-style Topological Update ---
    # We treat the points as a fixed topology (ring) and update them
    som_points = points.copy()
    for _ in range(iterations):
        snap = som_points.copy()
        new_pts = []
        for i in range(n_samples):
            p_i = snap[i]
            # SOM rule: pull neighbors towards p_i based on a fixed learning rate
            # Here we use the ring topology neighbors to simulate 'feature' pulling
            prev_idx = (i - 1) % n_samples
            next_idx = (i + 1) % n_samples
            target = (snap[prev_idx] + snap[next_idx]) / 2.0
            new_p = p_i + learning_rate * (target - p_i)
            new_pts.append(new_p)
        som_points = np.array(new_pts)
    som_success = 1.0 - (np.var(som_points) / initial_variance)

    return km_success, som_success

def run_experiment():
    print(f"experiment: k_means_vs_som_topology")
    # Test different learning rates for the SOM-like update
    learning_rates = [0.1, 0.5, 0.9]
    
    for lr in learning_rates:
        start_time = time.time()
        km_s, som_s = run_som_vs_kmeans(learning_rate=lr)
        end_time = time.time()
        print(f"lr_{lr:.1f}_km_success_{km_s:.4f}_som_success_{som_s:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_som_vs_kmeans = run_som_vs_kmeans # redundant but safe
    run_experiment()