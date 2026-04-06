import numpy as np
import time

def run_density_gradient_kmeans(points, iterations=20, radius=2.5):
    """
    Investigates k-means where the density of points is not uniform 
    but follows a gradient across the space. This tests if local 
    updates can resolve structure when the 'context' changes 
    linearly across the dataset.
    """
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot_points = current_points.copy()
        
        for i in range(len(current_points)):
            p = current_points[i]
            distances = np.linalg.norm(snapshot_points - p, axis=1)
            neighbors_mask = (distances <= radius) & (distances > 0)
            neighbors = snapshot_points[neighbors_mask]
            
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
    # Create a density gradient: more points on the left, fewer on the right
    num_points_left = 60
    num_points_right = 20
    
    c1 = np.random.randn(num_points_left, 2) + np.array([0, 0])
    c2 = np.random.randn(num_points_right, 2) + np.array([6, 6])
    
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    
    # Test different radii to see how they interact with the gradient
    radii = [1.0, 3.0, 7.0]
    
    print(f"experiment: density_gradient_k_means")
    for r in radii:
        start_time = time.time()
        final_points = run_density_gradient_kmeans(data, radius=r)
        end_time = time.time()
        
        final_variance = np.var(final_points)
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"radius_{r:.1f}_success_{success_metric:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()