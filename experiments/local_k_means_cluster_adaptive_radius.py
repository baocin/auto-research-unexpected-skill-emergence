import numpy as np
import time

def run_adaptive_radius_experiment(n_samples=100, decay_rate=0.9, iterations=30):
    """
    Investigates the impact of a decaying interaction radius (simulating 
    cooling or shrinking connectivity) on clustering success.
    As the radius shrinks, does it help 'lock in' clusters or prevent convergence?
    """
    np.random.seed(42)
    # Initial distribution: two spread out clusters
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(5, 1, (n_samples // 2, 2))
    points = np.vstack([c1, c2])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    current_radius = 5.0  # Start with a large radius to allow connectivity
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= current_radius) & (dists > 0)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        
        current_points = np.array(new_points)
        # Shrink the radius over time
        current_radius *= decay_rate
        
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: adaptive_radius_decay")
    # Test different decay rates
    decay_rates = [0.5, 0.8, 0.95, 0.99]
    
    for dr in decay_rates:
        start_time = time.time()
        success = run_adaptive_radius_experiment(decay_rate=dr)
        end_time = time.time()
        print(f"decay_{dr:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()