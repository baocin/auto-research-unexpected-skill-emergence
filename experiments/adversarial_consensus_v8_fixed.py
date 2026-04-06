import numpy as np
import time

def compute_adversarial_success(frozen_ratio):
    """Core logic for computing success rate under adversarial constraints."""
    np.random.seed(42)
    n_agents = 100
    v = np.random.rand(n_agents)
    ni = np.arange(n_agents)
    np.random.shuffle(ni)
    nf = int(n_agents * frozen_ratio)
    fi = ni[:nf]
    free_indices = ni[nf:]
    
    # Assign frozen agents to extreme values (0 and 1)
    for idx, pos in enumerate(fi):
        v[pos] = 0.0 if idx < nf // 2 else 1.0
        
    initial_variance = np.var(v)
    cv = v.copy()

    for _ in range(50):
        nv = cv.copy()
        for i in free_indices:
            # Local interaction window based on value similarity
            mask = (np.abs(cv - cv[i]) < 0.3)
            neighbors = cv[mask]
            if len(neighbors) > 0:
                target = np.mean(neighbors)
                nv[i] = cv[i] + 0.5 * (target - cv[i])
        cv = nv

    final_variance = np.var(cv)
    return 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

def run_experiment():
    print(f"experiment: adversarial_consensus_v8_fixed")
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    for r in ratios:
        start_time = time.time()
        try:
            success = compute_adversarial_success(r)
            duration = time.time() - start_time
            print(f"frozen_ratio_{r:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{r}_{e}")

if __name__ == "__main__":
    run_experiment()