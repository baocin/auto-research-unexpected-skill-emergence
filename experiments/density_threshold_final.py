import numpy as np
import time

def run_sim(n):
    np.random.seed(42)
    # Create points in [0, 10] domain
    pts = np.random.rand(n, 2) * 10.0
    init_v = np.var(pts)
    curr = pts.copy()
    
    for _ in range(30):
        snap = curr.copy()
        new_p = []
        for i in range(n):
            p_i = snap[i]
            dists = np.linalg.norm(snap - p_i, axis=1)
            mask = (dists <= 2.5) & (np.arange(n) != i)
            if np.any(mask):
                centroid = np.mean(snap[mask], axis=0)
                new_p.append(p_i + 0.5 * (centroid - p_i))
            else:
                new_p.append(p_i)
        curr = np.array(new_p)
    
    final_v = np.var(curr)
    return 1.0 - (final_v / init_v)

if __name__ == "__main__":
    print("experiment: density_threshold_clean")
    densities = [50, 100, 200, 400]
    for d in densities:
        start = time.time()
        success = run_sim(d)
        end = time.time()
        print(f"n_agents_{d}_success_{success:.4f}_time_{end - start:.4f}")