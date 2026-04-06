import numpy as np
import time

def run_perturbation_experiment(n_samples=100, perturbation_strength=0.5, iterations=30):
    """
    Investigates how the magnitude of random structural perturbations 
    (simulating 'shaking' the system) affects the stability of clusters.
    """
    np.random.seed(42)
    # Initial two clusters
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(5, 1, (n_samples // 2, 2))
    points = np.vstack([c1, c2])
    
    initial_variance = np.var(points)
    current_points = points.int_copy() if hasattr(points, 'int_copy') else points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        
        # Apply structural perturbation (shaking the agents)
        perturbation = np.random.normal(0, perturbation_strength, size=current_points.shape)
        current_points = current_points + perturbation
        
        for i in range(n_samples):
            p_i = snapshot[i] # Use original snapshot for stable neighbor detection
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_samples) != i)
            
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
    print(f"experiment: k_means_perturbation_stability")
    # Test perturbation strengths from low to high
    strengths = [0.01, 0.1, 0.5, 1.0]
    
    for s in strengths:
        start_time = time.time()
        success = run_perturbation_experiment(perturbation_strength=s)
        end_time = time.time()
        print(f"strength_{s:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()