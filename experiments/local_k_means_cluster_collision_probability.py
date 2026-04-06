import numpy as np
import time

def run_collision_experiment(n_samples=100, collision_radius=0.5, iterations=30):
    """
    Investigates how the 'collision' probability (the likelihood of two points 
    being within a certain radius) affects the stability and convergence of clusters.
    We measure if higher density/collision frequency leads to faster collapse or instability.
    """
    np.random.seed(42)
    # Generate random points in [0, 10] x [0, 10]
    points = np.random.rand(n_samples, 2) * 10
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # We simulate a 'collision' by slightly perturbing points that are too close
    # This tests if the algorithm can handle high-frequency local density fluctuations.
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= collision_radius) & (dists > 0)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                # Add a 'collision jitter' to simulate high-frequency density noise
                jitter = np.random.normal(0, 0.01, size=(2,))
                new_points.append(new_p + jitter)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_collision_jitter")
    # Test different collision radii (representing density/proximity thresholds)
    radii = [0.1, 0.5, 1.0, 2.0]
    
    for r in radii:
        start_time = time.time()
        success = run_collision_experiment(collision_radius=r)
        end_time = time.time()
        print(f"collision_radius_{r:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()