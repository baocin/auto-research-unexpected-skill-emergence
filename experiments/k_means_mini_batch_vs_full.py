import numpy as np
import time

def run_experiment():
    """
    Compares the computational complexity and precision of Full-Batch K-means
    versus Mini-batch K-means as N and D scale.
    """
    n_scales = [100, 300, 600]
    k = 3
    num_trials = 5
    noise_std = 1.0
    results = []

    for n in n_scales:
        d = n
        batch_size = 50
        trial_precisions_full = []
        trial_precisions_mini = []
        trial_times_full = []
        trial_times_mini = []
        
        for _ in range(num_trials):
            # Setup ground truth
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

            # --- Full Batch K-means ---
            start_f = time.time()
            centroids_f = centers.copy()
            for _ in range(10):
                dist_f = np.linalg.norm(data[:, np.newaxis, :] - centroids_f[np.new_axis, :, :], axis=2)
                labels_f = np.argmin(dist_f, axis=1)
                new_c = []
                for i in range(k):
                    mask = (labels_f == i)
                    if np.any(mask): new_c.append(np.mean(data[mask], axis=0))
                    else: new_c.append(centroids_f[i])
                centroids_f = np.array(new_c)
            trial_times_full.append(time.time() - start_f)
            
            # Evaluate Full Batch Precision
            cost_f = np.zeros((k, k))
            for i in range(k):
                for j in range(
                    k):
                    count = np.sum((labels_f == i) & (true_labels == j))
                    cost_f[i, j] = -count
            from scipy.optimize import linear_sum_assignment
            r_idx, c_idx = linear_sum_assignment(cost_f)
            mapping_f = {r: c for r, c in zip(r_idx, c_idx)}
            correct_f = sum(1 for idx, l in enumerate(labels_f) if mapping_f.get(l, -1) == true_labels[idx])
            trial_prec_f = correct_f / n

            # --- Mini-Batch K-means ---
            start_m = time.time()
            centroids_m = centers.copy()
            for _ in range(10): # Same number of iterations
                # Sample a mini-batch
                indices = np.random.choice(n, batch_size, replace=False)
                batch_data = data[indices]
                dist_m = np.linalg.norm(batch_data[:, np.newaxis, :] - centroids_m[np.new_axis, :, :], axis=2)
                labels_m = np.argmin(dist_m, axis=1)
                
                new_c_m = []
                for i in range(k):
                    mask = (labels_m == i)
                    if np.any(mask): new_c_m.append(np.mean(batch_data[mask], axis=0))
                    else: new_c_m.append(centroids_m[i])
                centroids_m = np.array(new_c_m)
            trial_times_mini.append(time.time() - start_m)

            # Evaluate Mini-batch Precision (using the same data to compare performance)
            dist_eval = np.linalg.norm(data[:, np.newaxis, :] - centroids_m[np.new_axis, :, :], axis=2)
            labels_eval = np.argmin(dist_eval, axis=1)
            cost_m = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    count = np.sum((labels_eval == i) & (true_labels == j))
                    cost_m[i, j] = -count
            r_idx_m, c_idx_m = linear_sum_assignment(cost_m)
            mapping_m = {r: c for r, c in zip(r_idx_m, c_idx_m)}
            correct_m = sum(1 for idx, l in enumerate(labels_eval) if mapping_m.get(l, -1) == true_labels[idx])
            trial_prec_m = correct_m / n
            
            trial_precisions_f.append(trial_prec_f) # This was a typo in my thought trace, must use local var
            # Let's fix the logic below to avoid errors.

            # (Self-correction: The implementation above is becoming messy with variable names).
            # I will rewrite one clean version for the final file.
            pass

        # Final structure... 
        # (See rewritten block below)
        pass

# Re-writing the logic in a single, clean function block.