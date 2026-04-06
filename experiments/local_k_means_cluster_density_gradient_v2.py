import numpy as np
import time

def run_density_gradient_experiment(n_samples=100, gradient_strength=1.0, iterations=30):
    """
    Investigates how a spatial density gradient affects convergence success.
    Uses a more robust sampling method to ensure the gradient is clearly present.
    """
    np.random.seed(42)
    # Define x-range
    x_coords = np.linspace(0, 10, n_samples)
    
    # Calculate weights based on exponential decay from center (5.0)
    # Higher strength means more concentration at the start of the range
    weights = np.exp(-gradient_strength * (x_coords - 5.0))
    weights /= np.sum(weights)
    
    # Sample x-coordinates using the calculated weights
    x_samples = np.random.choice(x_coords, size=n_samples, p=weights)
    # Uniformly distributed y-coordinates
    y_samples = np.random.rand(n_samples) * 10
    points = np.column_stack([x_samples, y_samples])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            # Interaction radius of 2.5
            neighbors_mask = (dists <= 2.5) & (np.arange(n_samples) != i)
            
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
    print(f"experiment: k_means_density_gradient_v2")
    # Test different gradient strengths
    gradient_strengths = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    for gs in gradient_strengths:
        start_time = time.time()
        success = run_density_gradient_experiment(gradient_strength=gs)
        end_time = time.time()
        print(f"grad_strength_{gs:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()