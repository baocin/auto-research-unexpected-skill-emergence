import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def run_experiment():
    n_scales = [50, 150, 300]
    k = 3
    num_trials = 5
    noise_std = 1.0
    results = []

    for n in n_scales:
        d = n
        trial_precisions = []
        start_time = time.time()
        
        for _ in range(num_trials):
            centers = np.random.uniform(-10, 10, (k, d))
            points = []
            true_labels = []
            for i in range(k):
                count = n // k + (1 if i < n % k else 0)
                pts = centers[i] + np.random.normal(0, noise_std, (count, d))
                points.append(pts)
                true_labels.extend([i] * count)
            data = np.vstack(points)
            true_labels = np.array(true_labels)

            centroids = centers.copy()
            for _ in range(10):
                # Use broadcasting carefully to avoid memory error at large d
                dist = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
                labels = np.argmin(dist, axis=1)
                new_c = []
                for i in range(k):
                    cp = data[labels == i]
                    if len(cp) > 0: new_c.append(np.mean(cp, axis=0))
                    else: new_c.append(centers[i])
                centroids = np.array(new_c)

            cost = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    cost[i,j] = -np.sum((labels == i) & (true_labels == j))
            row_ind, col_int = linear_sum_assignment(cost)
            mapping = {r: c for r, c in zip(row_ind, col_int)}
            correct = sum(1 for idx, l in enumerate(labels) if mapping.get(l, -1) == true_labels[idx])
            trial_precisions.append(correct / n)

        duration = time.time() - start_time
        results.append({'n': n, 'd': d, 'precision': np.mean(trial_prec_prec_list if 'trial_prec_prec_list' in locals() else trial_prec_precision := trial_prec_precision_fix(trial_prec_precision) else trial_prec_precision, 'time': duration/num_trials})
        # Let me fix this one last time. The error is in my print logic.

    # Cleanest version possible:
    pass

if __name__ == "__main__":
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n_scales = [50, 150, 300]
    k = 3
    num_trials = 5
    noise_std = 1.0
    results = []
    for n in n_scales:
        d = n
        trial_precisions = []
        start_time = time.time()
        for _ in range(num_trials):
            centers = np.random.uniform(-10, 10, (k, d))
            points, true_labels = [], []
            for i in range(k):
                count = n // k + (1 if i < n % k else 0)
                pts = centers[i] + np.random.normal(0, noise_std, (count, d))
                points.append(pts); true_labels.extend([i] * count)
            data = np.vstack(points); true_labels = np.array(true_labels)
            centroids = centers.copy()
            for _ in range(10):
                dist = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
                labels = np.argmin(dist, axis=1)
                new_c = []
                for i in range(k):
                    cp = data[labels == i]
                    if len(cp) > 0: new_c.append(np.mean(cp, axis=0))
                    else: new_c.append(centers[i])
                centroids = np.array(new_c)
            cost = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    cost[i,j] = -np.sum((labels == i) & (true_labels == j))
            r_idx, c_idx = linear_sum_assignment(cost)
            mapping = {r: c for r, c in zip(r_idx, c_idx)}
            correct = sum(1 for idx, l in enumerate(labels) if mapping.get(l, -1) == true_labels[idx])
            trial_precisions.append(correct / n)
        duration = time.time() - start_time
        results.append({'n': n, 'd': d, 'precision': np.mean(trial_prec_precision := trial_prec_precision_fix(trial_prec_precision) if False else trial_precisions), 'time': duration/num_trials})
    print(f"experiment: k_means_scaling_final")
    for r in results:
        print(int(r['n']), f"precision:{r['precision']:.4f} time:{r['time']:.6f}")