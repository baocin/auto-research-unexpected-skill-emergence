import numpy as np
import time

def run_collision_edge_case(n_samples=100, collision_radius=0.5, iterations=30):
    """
    Investigates the 'near-collision' effect: when agents are extremely close, 
    does their interaction strength (and thus convergence) fluctuate?
    We test how sensitive the algorithm is to precision errors at a micro-scale.
    """
    np.random.seed(42)
    # Generate points in [0, 10] domain
    points = np.random.rand(n_samples, 2) * 10
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            # Calculate distances
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            # Use the collision radius for neighbor detection
            neighbors_mask = (dists <= collision_radius) & (dists > 1e-12)
            
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
    print(f"experiment: k_means_epsilon_collision")
    epsilons = [1e-12, 1e-9, 1e-6, 1e-3]
    
    for eps in epsilons:
        start_time = time.time()
        success = run_collision_edge_case(collision_radius=eps)
        end_time = time.time()
        print(f"epsilon_{eps:.1e}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()