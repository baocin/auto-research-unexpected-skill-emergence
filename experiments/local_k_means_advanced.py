import numpy as np
import time

def run_advanced_kmeans(points, iterations=20, radius=2.5, temperature=1.0):
    """
    An advanced version of local k-means using a 'temperature' parameter 
    to control the stochasticity of point movement (simulated annealing approach).
    High temperature allows for more exploration; low temperature promotes convergence.
    """
    current_points = points.copy()
    
    for iter_idx in range(iterations):
        new_points = []
        snapshot_points = current_points.copy()
        # Cool the temperature over time
        T = temperature * (1 - iter_idx / iterations)
        
        for i in range(len(current_points)):
            p = current_points[i]
            distances = np.linalg.norm(snapshot_points - p, axis=1)
            neighbors_mask = (distances <= radius) & (distances > 0)
            neighbors = snapshot_points[neighbors_mask]
            
            if len(neighbors) > 0:
                centroid = np.mean(neighbors, axis=0)
                direction = centroid - p
                
                # Add stochastic noise scaled by temperature
                noise = np.random.normal(0, T, size=2)
                new_p = p + 0.5 * direction + noise
                new_points.append(new_p)
            else:
                new_points.append(p)
        current_points = np.array(new_points)
        
    return current_points

def run_experiment():
    # Generate two distinct clusters
    np.random.seed(7)
    c1 = np.random.randn(30, 2) + np.array([0, 0])
    c2 = np.random.randn(30, 2) + np.array([5, 5])
    data = np.vstack([c1, c2])
    
    initial_variance = np.var(data)
    
    # Test different temperatures
    temperatures = [0.01, 0.5, 2.0]
    
    print(f"experiment: advanced_local_k_means")
    for T in temperatures:
        start_time = time.time()
        final_points = run_advanced_kmeans(data, iterations=25, radius=3.0, temperature=T)
        end_time = time.time()
        
        final_variance = np.var(final_points)
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"temp_{T:.2f}_success_{success_metric:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()