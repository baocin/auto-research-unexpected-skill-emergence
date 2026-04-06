import numpy as np
import time

def run_hierarchical_kmeans(points, iterations=15, radius=2.5):
    """
    Investigates a 'Hierarchical' local k-means approach:
    In each iteration, nodes first perform a local smoothing (averaging),
    and then use that smoothed value to compute the next movement step.
    This tests if a two-stage local update can stabilize convergence 
    in high-variance environments.
    """
    current_points = points.copy()
    
    for _ in range(iterations):
        # Stage 1: Local Smoothing (pre-processing the neighborhood)
        smoothed_points = np.copy(current_points)
        for i in range(len(current_points)):
            distances = np.linalg.norm(current_points - current_points[i], axis=1)
            neighbors_mask = (distances <= radius) & (distances > 0)
            if np.any(neighbors_mask):
                smoothed_points[i] = np.mean(current_points[neighbors_mask], axis=0)
        
        # Stage 2: Movement based on smoothed neighborhood
        new_points = []
        snapshot_smoothed = smoothed_points.copy()
        
        for i in range(len(current_points)):
            p = current_points[i]
            distances = np.linalg.norm(snapshot_smoothed - p, axis=1)
            neighbors_mask = (distances <= radius) & (distances > 0)
            neighbors = snapshot_smoothed[neighbors_mask]
            
            if len(neighbors) > 0:
                centroid = np.mean(neighbors, axis=0)
                direction = centroid - p
                new_p = p + 0.5 * direction
                new_points.append(new_p)
            else:
                new_points.append(p)
        current_points = np.array(new_points)
        
    return current_points

def run_experiment():
    np.random.seed(42)
    # Two clusters with a clear gap
    c1 = np.random.randn(30, 2) + np.array([0, 0])
    c2 = np.random.randn(30, 2) + np.array([6, 6])
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    
    # Test different radii
    radii = [1.5, 3.0, 7.0]
    
    print(f"experiment: hierarchical_local_k_means")
    for r in radii:
        start_time = time.time()
        final_points = run_hierarchical_kmeans(data, radius=r)
        end_time = time.time()
        
        final_variance = np.var(final_points)
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"radius_{r:.1f}_success_{success_metric:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()