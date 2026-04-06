import numpy as np
import time

def run_experiment():
    n_samples = 500
    p_damage = 0.2
    anomaly_magnitude = 10.0
    noise_std = 1.0
    num_trials = 20

    results = []

    for trial in range(num_trials):
        true_stream = np.random.normal(0, noise_std, n_samples)
        true_stream[250] = anomaly_magnitude
        
        damaged_stream = true_stream.copy()
        damage_mask = np.random.rand(n_samples) < p_damage
        damaged_stream[damage_mask] = 0.0
        
        # Imputation: Replace zeros with the rolling median to neutralize damage
        imputed_stream = damaged_stream.copy()
        window_size = 20
        for i in range(window_size, n_samples):
            window = damaged_stream[i-window_size:i]
            # Use the median of the window to fill the 'damaged' holes
            imputed_stream[i] = np.median(window)

        window_size = 20
        detections = []
        
        for i in range(window_size, n_samples):
            window = imputed_stream[i-window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 1e-5:
                z_score = abs(imputed_stream[i] - mean) / std
                if z_score > 3.0:
                    detections.append(i)

        tp = 1 if 250 in detections else 0
        fp = len([d for d in detections if d != 250])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp 
        
        results.append({
            'tp': tp, 'fp': fp, 'precision': precision, 'recall': recall
        })

    avg_prec = np.mean([r['precision'] for r in results])
    avg_rec = np.mean([r['recall'] for r in results])
    
    print(f"experiment: anomaly_detection_impute_v1")
    print(f"n:{n_samples} p_damage:{p_damage} avg_precision:{avg_prec:.4f} avg_recall:{avg_rec:.4f}")

if __name__ == "__main__":
    run_experiment()