import numpy as np
import time

def run_local_kmeans(points, iterations=10, radius=2.0):
    new_points = points.copy()
    for _ in range(iterations):
        updates = []
        for i, p in enumerate(points):
            distances = np.linalg.norm(points - p, axis=1)
            neighbors_mask = distances <= radius
            neighbors = points[neighbors_mask]
            if len(neighbors) > 1:
                centroid = np.mean(neighbors, axis=0)
                new_p = p + 0.5 * (centroid - p)
                updates.append(new_p)
            else:
                updates.append(p)
        new_points = np.array(updates)
        points = new_points.copy()
    return points

def run_experiment():
    # Test different scales of N (number of points) and R (radius)
    scales = [20, 50, 100]
    radii = [1.0, 2.0, 5.0]
    results = []

    for n in scales:
        # Create a cluster-based dataset for each scale
        cluster1 = np.random.randn(n // 2, 2) + np.array([0, 0])
        cluster2 = np.random.randn(n // 2, 2) + np.array([5, 5])
        data = np.vstack([cluster1, cluster2])
        initial_var = np.var(data)

        for r in radii:
            start_time = time.time()
            final_points = run_local_kmeans(data, iterations=10, radius=r)
            end_time = time.time()
            
            final_var = np.var(final_points)
            success_metric = 1.0 - (final_var / initial_var) if initial_var > 0 else 0
            
            results.append((n, r, success_metric, end_time - start_time))

    print("experiment: scaling_analysis")
    for n, r, score, duration in results:
        print(f"N_{n}_R_{r}_success_{score:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()