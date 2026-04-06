import numpy as np
import time

def run_simulation(bw, n_samples=100, iterations=30):
    np.random.seed(42)
    c1 = np.random.rand(n_samples // 2, 2); c1[:, 0] *= 3.0
    c2 = np.random.rand(n_samples // 2, 2); c2[:, 0] = (c2[:, 0] * 3.0) + 7.0
    points = np.vstack([c1, c2])
    init_v = np.var(points)
    curr = points.copy()
    
    for _ in range(iterations):
        snap = curr.copy()
        new_pts = []
        for i in range(n_samples):
            p_i = snap[i]
            dists = np.linalg.norm(snap - p_i, axis=1)
            mask = (dists <= 3.0) & (np.arange(n_samples) != i)
            b_mask = mask & ((snap[:, 0] >= (5.0 - bw/2)) & (snap[:, 0] <= (5.0 + bw/2)))
            
            if np.any(b_mask):
                new_pts.append(p_i + 0.5 * (np.mean(snap[b_mask], axis=0) - p_i))
            elif np.any(mask):
                new_pts.append(p_i + 0.5 * (np.mean(snap[mask], axis=0) - p_i))
            else:
                new_pts.append(p_i)
        curr = np.array(new_pts)
    return 1.0 - (np.var(curr)/init_v)

if __name__ == "__main__":
    print("experiment: k_means_bottleneck_fixed")
    widths = [0.1, 0.5, 1.0, 2.0]
    for w in widths:
        start = time.time()
        success = run_simulation(bw=w)
        end = time.time()
        print(f"width_{w:.1f}_success_{success:.4f}_time_{end - start:.4f}")