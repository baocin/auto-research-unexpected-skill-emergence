import numpy as np
import time

def run_heterogeneous_kmeans(points, iterations=20, radius=2.5):
    """
    Investigates k-means where clusters have significantly different densities.
    This tests if the local algorithm can resolve structure when one cluster 
    is much 'tighter' than the other.
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
    np.random.seed(42)
    # Cluster 1: Very dense
    c1 = np.random.randn(40, 2) * 0.5 + np.array([0, 0])
    # Cluster 2: Very sparse/spread out
    c2 = np.random.randn(40, 2) * 3.0 + np.array([8, 8])
    data = np.vstack([c1, c2])
    
    initial_variance = np.var(data)
    
    # Test across different radii to see if the dense cluster is 'swallowed'
    radii = [1.0, 3.0, 6.0]
    
    print(f"experiment: heterogeneous_density_kmeans")
    for r in radii:
        start_time = time.time()
        final_points = run_heterogeneous_kmeans(data, radius=r)
        end_time = time.time()
        
        final_variance = np.var(final_points)
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"radius_{r:.1f}_success_{success_metric:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()