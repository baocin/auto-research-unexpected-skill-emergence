import numpy as np
import time

def run_dropout_experiment(n_agents=100, dropout_rate=0.2, iterations=30):
    """
    Investigates 'Sensor Dropout Frequency': How the frequency of information 
    loss (simulated by masking features) affects clustering stability.
    This tests if the system is resilient to intermittent data loss.
    """
    np.random.seed(42)
    # Two clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c_array for c_array in [c1, c2]])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        
        # Dropout: Randomly mask features of the snapshot for this step
        mask = np.random.rand(*snapshot.shape) > dropout_rate
        masked_snapshot = snapshot * mask
        
        for i in range(n_agents):
            p_i = masked_snapshot[i]
            dists = np.linalg.norm(masked_snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                centroid = np.mean(masked_snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_sensor_dropout")
    # Test different dropout rates (from 0% to 80% information loss)
    dropout_rates = [0.0, 0.1, 0.3, 0.5, 0.8]
    for dr in dropout_rates:
        start_time = time.time()
        try:
            success = run_dropout_experiment(dropout_rate=dr)
            duration = time.time() - start_time
            print(f"dropout_{dr:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{dr}_{e}")

if __name__ == "__main__":
    run_experiment()