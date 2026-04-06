import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def run_experiment():
    n_scales = [500, 1000, 2000]
    k = 3
    num_trials = 5
    noise_std = 1.5
    cluster_sep = 5.0
    d = 10
    results = []

    for n in n_scales:
        batch_size = max(int(n * 0.1), 20)
        p_full_list = []
        p_mini_list = []
        t_full_list = []
        t_mini_list = []
        
        for _ in range(num_trials):
            # Generate data
            centers = np.zeros((k, d))
            for i in range(k):
                centers[i, 0] = i * cluster_sep
            
            points = []
            true_labels = []
            for i in range(k):
                count = n // k + (1 if i < n % k else 0)
                pts = centers[i] + np.random.normal(0, noise_std, (count, d))
                points.append(pts)
                true_labels.extend([i] * count)
            data = np.vstack(points)
            true_labels = np.array(true_labels)

            # --- Full Batch K-means ---
            s_f = time.time()
            c_f = centers.copy()
            for _ in range(15):
                dist_f = np.linalg.norm(data[:, np.newaxis, :] - c_f[np.newaxis, :, :], axis=2)
                labels_f = np.argmin(dist_f, axis=1)
                new_c = []
                for i in range(k):
                    mask = (labels_f == i)
                    if np.any(mask):
                        new_c.append(np.mean(data[mask], axis=0))
                    else:
                        new_c.append(c_f[i])
                c_f = np.array(new_c)
            t_full_list.append(time.time() - s_f)
            
            # Evaluation for Full Batch
            cost_f = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    count = np.sum((labels_f == i) & (true_labels == j))
                    cost_f[i, j] = -float(count)
            r_idx, c_idx = linear_sum_assignment(cost_f)
            mapping = {r: c for r, c in zip(r_idx, c_idx)}
            correct_f = sum(1 for idx, l in enumerate(labels_f) if mapping.get(l, -1) == true_labels[idx])
            p_full_list.append(correct_f / n)

            # --- Mini Batch K-means ---
            s_m = time.time()
            c_m = centers.copy()
            for _ in range(15):
                idx = np.random.choice(n, batch_size, replace=False)
                batch_data = data[idx]
                dist_m = np.linalg.norm(batch_data[:, np.newaxis, :] - c_m[np.newaxis, :, :], axis=2)
                labels_m = np.argmin(dist_m, axis=1)
                new_c = []
                for i in range(k):
                    mask = (labels_m == i)
                    if np.any(mask):
                        new_c.append(np.mean(batch_data[mask], axis=0))
                    else:
                        new_c.append(c_m[i])
                c_m = np.array(new_c)
            t_mini_list.append(time.time() - s_m)

            # Evaluation for Mini Batch (using full data labels)
            dist_eval = np.linalg.norm(data[:, np.newaxis, :] - c_m[np.newaxis, :, :], axis=2)
            labels_eval = np.argmin(dist_eval, axis=1)
            cost_m = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    count = np.sum((labels_eval == i) & (true_labels == j))
                    cost_m[i, j] = -float(count)
            r_idx_m, c_idx_m = linear_sum_assignment(cost_m)
            mapping_m = {r: c for r, c in zip(r_idx_m, c_idx_m)}
            correct_m = sum(1 for idx, l in enumerate(labels_eval) if mapping_m.get(l, -1) == true_labels[idx])
            p_mini_list.append(correct_m / n)

        results.append({
            'n': n, 
            'precision_full': np.mean(p_full_list),
            'precision_mini': np.mean(p_mini_list),
            'time_full': np.mean(t_full_list),
            'time_mini': np.mean(t_mini_list)
        })

    print(f"experiment: k_means_stress_test_v1")
    for r in results:
        print(f"n:{r['n']} p_f:{r['precision_full']:.4f} p_m:{r['precision_mini']:.4f} t_f:{r['time_full']:.6f} t_m:{r['time_mini']:.6f}")

if __name__ == "__main__":
    run_experiment()