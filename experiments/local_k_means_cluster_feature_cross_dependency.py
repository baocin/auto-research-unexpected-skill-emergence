import numpy as np
import time

def run_cross_dependency_experiment(n_agents=100, dependency_strength=0.5, iterations=30):
    """
    Investigates 'Feature Cross-Dependency': How the correlation between 
    the x and y dimensions affects the stability of convergence.
    This tests if the algorithm's ability to converge is sensitive to 
    the dimensionality of information (from independent axes to highly correlated).
    """
    np.random.seed(42)
    # Base clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # Create dependency: Rotate and scale the space based on dependency strength
    # We use a rotation matrix where the angle is controlled by dependency_strength
    theta = dependency_strength * (np.pi / 2)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Apply the dependency transformation to all points
    current_points = current_points @ rotation_matrix
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg_norm_helper(snapshot, p_i)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
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

def np_linalg_norm_helper(snapshot, p_i):
    return np.linalg.norm(snapshot - p_i, axis=1)

# Redefining for execution robustness
def run_experiment():
    print(f"experiment: k_means_feature_cross_dependency")
    # Dependency strengths: 0 (independent) to 1 (fully dependent/collinear)
    strengths = [0.0, 0.2, 0.5, 0.8]
    for s in strengths:
        start_time = time.time()
        try:
            # Re-implementing the logic inside for extreme robustness
            np.random.seed(42)
            n = 100
            c1 = np.random.normal(0, 1, (n // 2, 2))
            c2 = np.random.normal(5, 1, (n // 2, 2))
            pts = np.vstack([c1, c2])
            iv = np.var(pts)
            curr = pts.copy()
            
            theta = s * (np.pi / 2)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            pts_transformed = pts @ rot
            curr = pts_transformed.copy()

            for _ in range(30):
                snap = curr.copy()
                new_p = []
                for i in range(n):
                    dists = np.linalg.norm(snap - snap[i], axis=1)
                    mask = (dists <= 3.0) & (np.arange(n) != i)
                    if np.any(mask):
                        cen = np.mean(snap[mask], axis=0)
                        new_p.append(snap[i] + 0.5 * (cen - snap[i]))
                    else:
                        new_p.append(snap[i])
                curr = np.array(new_p)
            
            fv = np.var(curr)
            success = 1.0 - (fv/iv)
            duration = time.time() - start_time
            print(f"strength_{s:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{s}_{e}")

if __name__ == "__main__":
    run_experiment()