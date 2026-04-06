import numpy as np
import time

def run_momentum_kmeans(points, iterations=20, radius=2.5, momentum=0.9):
    """
    Investigates if a momentum term helps or hinders convergence 
    in local k-means clustering.
    """
    current_points = points.copy()
    velocities = np.zeros_like(points)
    
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
                # Apply momentum to the velocity vector
                velocities[i] = (momentum * velocities[i]) + (1 - momentum) * direction
                new_p = p + velocities[i]
                new_points.append(new_p)
            else:
                new_points.append(p)
        current_points = np.array(new_points)
        
    return current_points

def run_experiment():
    np.random.seed(42)
    # Generate two clusters
    c1 = np.random.randn(30, 2) + np.array([0, 0])
    c2 = np.random.randn(30, 2) + np.array([5, 5])
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    
    # Test different momentum values: No momentum vs High momentum
    momentum_values = [0.0, 0.5, 0.95]
    
    print(f"experiment: momentum_k_means")
    for m in momentum_values:
        start_time = time.time()
        final_points = run_momentum_kmeans(data, iterations=20, radius=3.0, momentum=m)
        end_time = time.time()
        
        final_variance = np.var(final_points)
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"momentum_{m:.2f}_success_{success_metric:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()