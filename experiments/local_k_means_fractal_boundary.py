import numpy as np
import time

def run_fractal_boundary_kmeans(points, iterations=20, radius=2.5):
    """
    Investigates k-means where the boundary between clusters is not a simple line
    but a complex, high-frequency 'fractal' interface (sinusoidal perturbation).
    This tests if local updates can resolve structure in highly interleaved distributions.
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
    # Create two clusters that are highly interleaved/fractal at the boundary
    n_samples = 100
    t = np.linspace(0, 10, n_samples)
    # Cluster 1: points along a sine wave
    c1 = np.column_stack([t, np.sin(t)]) + np.random.normal(0, 0.2, (n_samples, 2))
    # Cluster 2: points along a shifted/perturbed sine wave
    c2 = np.column_stack([t, np.sin(t) + 3.0]) + np.random.normal(0, 0.2, (n_samples, 2))
    
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    
    # Test different radii to see if they can 'untangle' the interleaved waves
    radii = [0.5, 2.0, 5.0]
    
    print(f"experiment: fractal_boundary_k_means")
    for r in radii:
        start_time = time.time()
        final_points = run_fract_kmeans_helper(data, r)
        end_time = time.time()
        
        final_variance = np.var(final_points)
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"radius_{r:.1f}_success_{success_metric:.4f}_time_{end_time - start_time:.4f}")

def fract_kmeans_helper(data, r):
    # Re-using the logic to avoid complexity in the main function
    from experiments.local_k_means_hierarchical import run_hierarchical_kmeans
    # We use a simplified version of the logic directly here for stability
    return run_fract_logic(data, r)

def run_fract_logic(points, radius):
    current_points = points.copy()
    for _ in range(20):
        new_points = []
        snapshot = current_points.copy()
        for i in range(len(current_points)):
            p = current_points[i]
            dists = np.linalg.norm(snapshot - p, axis=1)
            mask = (dists <= radius) & (dists > 0)
            neighbors = snapshot[mask]
            if len(neighbors) > 0:
                new_p = p + 0.5 * (np.mean(neighbors, axis=0) - p)
                new_points.append(new_p)
            else:
                new_points.append(p)
        current_points = np.array(new_points)
    return current_points

# Redefining the experiment runner to be self-contained and robust
def run_experiment_final():
    np.random.seed(42)
    n_samples = 100
    t = np.linspace(0, 10, n_samples)
    c1 = np.column_stack([t, np.sin(t)]) + np.random.normal(0, 0.2, (n_samples, 2))
    c2 = np.column_stack([t, np.sin(t) + 3.0]) + np.random.normal(0, 0.2, (n_samples, 2))
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    radii = [0.5, 2.0, 5.0]
    print(f"experiment: fractal_boundary_k_means")
    for r in radii:
        start_time = time.time()
        # Direct implementation for robustness
        current_points = data.copy()
        for _ in range(20):
            snapshot = current_points.copy()
            new_p_list = []
            for i in range(len(current_points)):
                dists = np.linalg_norm_calc(snapshot, current_points[i])
                mask = (dists <= r) & (dists > 0)
                if np.any(mask):
                    new_p_list.append(current_points[i] + 0.5 * (np.mean(snapshot[mask], axis=0) - current_points[i]))
                else:
                    new_p_list.append(int_points_val := current_points[i])
            current_points = np.array(new_p_list)
        end_time = time.time()
        final_variance = np.var(current_points)
        success = 1.0 - (final_variance / initial_variance)
        print(f"radius_{r:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

# The above was getting messy, let's just write a clean, single-function script.