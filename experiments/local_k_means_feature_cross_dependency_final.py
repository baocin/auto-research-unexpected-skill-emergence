import numpy as np
import time

def run_experiment():
    print("experiment: k_means_feature_cross_dependency")
    strengths = [0.0, 0.2, 0.5, 0.8]
    
    for s in strengths:
        start_time = time.time()
        try:
            # Setup data
            np.random.seed(42)
            n = 100
            c1 = np.random.normal(0, 1, (n // 2, 2))
            c2 = np.random.normal(5, 1, (n // 2, 2))
            pts = np.vstack([c1, c2])
            initial_variance = np.var(pts)
            
            # Apply rotation (dependency/correlation)
            theta = s * (np.pi / 2)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            current_points = pts @ rotation_matrix
            
            # Iterative clustering process
            iterations = 30
            for _ in range(iterations):
                snapshot = current_points.copy()
                new_p_list = []
                for i in range(n):
                    p_i = snapshot[i]
                    # Distance calculation in the rotated space
                    dists = np.linalg.norm(snapshot - p_i, axis=1)
                    neighbors_mask = (dists <= 3.0) & (np.arange(n) != i)
                    
                    if np.any(neighbors_mask):
                        centroid = np.mean(snapshot[neighbors_mask], axis=0)
                        new_p = p_i + 0.5 * (centroid - p_i)
                        new_p_list.append(new_p)
                    else:
                        new_p_list.append(p_i)
                current_points = np.array(new_p_list)
            
            final_variance = np.var(current_points)
            success = 1.0 - (final_variance / initial_variance)
            duration = time.time() - start_time
            print(f"strength_{s:.1f}_success_{success:.4f}_time_{duration:.4f}")
            
        except Exception as e:
            print(f"error_at_{s}_{e}")

if __name__ == "__main__":
    run_experiment()