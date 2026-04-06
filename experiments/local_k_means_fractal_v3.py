import numpy as np
import time

def run_experiment():
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
        current_points = data.copy()
        start_time = time.time()
        for _ in range(20):
            snapshot = current_points.copy()
            new_p_list = []
            for i in range(len(current_points)):
                p = current_points[i]
                dists = np.linalg.norm(snapshot - p, axis=1)
                mask = (dists <= r) & (dists > 0)
                if np.any(mask):
                    new_p_list.append(p + 0.5 * (np.mean(snapshot[mask], axis=0) - p))
                else:
                    new_p_list.append(p)
            current_points = np.array(new_p_list)
        end_time = time.time()
        final_variance = np.var(current_points)
        success = 1.0 - (final_variance / initial_variance)
        print(f"radius_{r:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()