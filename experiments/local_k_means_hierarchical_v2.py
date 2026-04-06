import numpy as np
import time

def run_hierarchical_kmeans(points, iterations=15, radius=2.5):
    current_points = points.copy()
    for _ in range(iterations):
        # Stage 1: Smoothing
        smoothed_points = np.copy(current_points)
        for i in range(len(current_points)):
            dists = np.linalg.norm(current_points - current_points[i], axis=1)
            mask = (dists <= radius) & (dists > 0)
            if np.any(mask):
                smoothed_points[i] = np.mean(current_points[mask], axis=0)
        
        # Stage 2: Movement
        new_points = []
        for i in range(len(current_points)):
            p = current_points[i]
            dists = np.linalg.norm(smoothed_points - p, axis=1)
            mask = (dists <= radius) & (dists > 0)
            neighbors = smoothed_points[mask]
            if len(neighbors) > 0:
                centroid = np.mean(neighbors, axis=0)
                new_p = p + 0.5 * (centroid - p)
                new_points.append(new_p)
            else:
                new_points.append(p)
        current_points = np.array(new_points)
    return current_points

def run_experiment():
    np.random.seed(42)
    c1 = np.random.randn(30, 2) + np.array([0, 0])
    c2 = np.random.randn(30, 2) + np.array([6, 6])
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    radii = [1.5, 3.0, 7.0]
    print(f"experiment: hierarchical_k_means_v2")
    for r in radii:
        start_time = time.time()
        final_points = run_hierarchical_kmeans(data, radius=r)
        end_time = time.time()
        final_variance = np.var(final_points)
        success = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
        print(f"radius_{r:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()