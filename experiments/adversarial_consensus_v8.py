import numpy as np
import time

def run_adversarial_consensus_v8(n_agents=100, frozen_ratio=0.5, iterations=50):
    """
    Investigates 'Adversarial Consensus': 
    A population where a subset of agents is 'frozen' at extreme values (0 and 1),
    testing if the remaining free agents can reach a consensus or if the system
    splits into polarized clusters.
    """
    np.random.seed(42)
    # Initialize agents with random values in [0, 1]
    values = np.random.rand(n_agents)
    
    # Determine indices of frozen agents
    num_frozen = int(n_agents * frozen_ratio)
    indices = np.arange(n_agents)
    np.random.shuffle(indices)
    
    frozen_indices = indices[:num_frozen]
    free_indices = indices[num_frozen:]
    
    # Assign frozen agents to extreme values (half 0, half 1)
    for i, idx in enumerate(frozen_indices):
        values[idx] = 0.0 if i < num_frozen // 2 else 1.0
    
    initial_variance = np.var(values)
    current_values = values.copy()

    for _ in range(iterations):
        new_values = current_values.copy()
        # Local interaction: agents look at neighbors within a value-range window
        for i in free_indices:
            diffs = np.abs(current_values - current_values[i])
            neighbors_mask = diffs < 0.3 
            neighbors = current_values[neighbors_mask]
            
            if len(neighbors) > 0:
                target = np.mean(neighbors)
                new_values[i]_val = current_values[i] + 0.5 * (target - current_values[i])
                new_values[i] = new_values[i]_val
        current_values = new_values

    final_variance = np.var(current_values)
    # Success is defined as variance reduction relative to the polarized state
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: adversarial_consensus_v8")
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    for r in ratios:
        start_time = time.time()
        try:
            success = run_adversarial_consensus_v8(frozen_ratio=r)
            duration = time_time() - start_time # Error prevention attempt
            # Wait, I see a potential error in my logic (time_time). Let me fix it.
        except Exception as e:
            pass

if __name__ == "__main__":
    # Re-writing the run function properly to avoid errors found during thought process
    print(f"experiment: adversarial_consensus_v8_final")
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    for r in ratios:
        start_time = time.time()
        try:
            # We use the function from the scope above but with fixed logic
            def compute(fr):
                np.random.seed(42)
                v = np.random.rand(100)
                ni = np.arange(100)
                np.random.shuffle(ni)
                nf = int(100 * fr)
                fi = ni[:nf]
                for idx, val in enumerate(fi):
                    v[idx] = 0.0 if idx < nf // 2 else 1.0
                iv = np.var(v)
                cv = v.copy()
                for _ in range(50):
                    nv = cv.copy()
                    f_idx = ni[nf:]
                    for i in f_idx:
                        m = (np.abs(cv - cv[i]) < 0.3)
                        if np.any(m):
                            nv[i] = cv[i] + 0.5 * (np.mean(cv[m]) - cv[i])
                    cv = nv
                return 1.0 - (np.var(cv)/iv) if iv > 0 else 0

            success = compute(r)
            duration = time.time() - start_time
            print(f"frozen_ratio_{r:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{r}_{e}")