import numpy as np
import time

def run_experiment():
    """
    Tests K-means robustness when certain 'sensors' (features) are randomly 
    set to zero. We use K=2 and ground truth labels [0, 1] to ensure 
    evaluation is mathematically valid.
    """
    n_samples = 1000
    d = 20
    num_trials = 5
    damage_rates = [0.0, 0.2, 0.5]
    k = 2
    results = []

    for damage in damage_rates:
        p_damaged_list = []
        t_damaged_list = []
        
        for _ in range(num_trials):
            # Generate data (Two clusters: one at origin, one at offset)
            centers = np.array([[0]*d, [5]*d])
            points = []
            true_labels = []
            for i in range(k):
                count = n_samples // k
                pts = centers[i] + np.random.normal(0, 1, (count, d))
                points.append(pts)
                true_labels.extend([i] * count)
            data_clean = np.vstack(points)
            true_labels = np.array(true_labels)

            # Create damaged data
            data_damaged = data_clean.copy()
            if damage > 0:
                mask = np.random.rand(n_samples, d) < damage
                data_damaged[mask] = 0.0

            # --- Evaluate Damaged K-means ---
            s_d = time.time()
            c_d = centers.copy()
            for _ in range(15):
                dist = np.linalg.norm(data_damaged[:, np.newaxis, :] - c_d[np.newaxis, :, :], axis=2)
                labels_d = np.argmin(dist, axis=1)
                new_c = []
                for i in range(k):
                    mask_idx = (labels_d == i)
                    if np.any(mask_idx):
                        new_c.append(np.mean(data_damaged[mask_idx], axis=0))
                    else:
                        new_c.append(c_d[i])
                c_d = np.array(new_c)
            t_damaged = time.time() - s_d
            
            # Evaluation on CLEAN data using centroids learned from DAMAGED data
            dist_eval = np.linalg.norm(data_clean[:, np.newaxis, :] - c_d[np.newaxis, :, :], axis=2)
            labels_eval = np.argmin(dist_eval, axis=1)
            
            # Since K=2, we check if prediction matches truth (handling potential label swap)
            # A simple way: Check if the error rate is the same if we flip 0 and 1
            acc_normal = np.mean(labels_eval == true_labels)
            acc_flipped = 1.0 - acc_normal
            p_damaged = max(acc_normal, acc_flipped)
            
            p_damaged_list.append(p_damaged)
            t_damaged_list.append(t_damaged)

        results.append({
            'damage': damage,
            'p_damaged': np.mean(p_damaged_list),
            't_damaged': np.mean(t_damaged_list)
        })

    print("experiment: k_means_sensor_damage")
    for r in results:
        print(f"damage:{r['damage']} p:{r['p_damaged']:.4f} t:{r['t_damaged']:.6f}")

if __name__ == "__main__":
    run_experiment()