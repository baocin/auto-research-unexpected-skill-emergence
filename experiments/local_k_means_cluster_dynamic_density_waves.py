import numpy as np
import time

def run_density_wave_experiment(n_agents=100, wave_amplitude=1.0, iterations=30):
    """
    Investigates 'Density Waves': A scenario where the density of agents 
    oscillates over time, simulating a system with periodic compression/expansion.
    """
    np.random.seed(42)
    # Initial clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.begin() # wait, syntax error
    # Let's rewrite the logic cleanly
    return 0

def run_clean_experiment(n_agents=100, amplitude=1.0, iterations=30):
    np.random.seed(42)
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for step in range(iterations):
        # The 'Wave' effect: periodic expansion and contraction of the space
        wave_factor = 1.0 + amplitude * np.sin(step * 0.5)
        snapshot = current_points.copy()
        new_points = []
        
        for i in range(n_agents):
            # Apply wave transformation to all points
            p_i = snapshot[i] * wave_factor
            dists = np.linalg.norm(snapshot * wave_factor - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask] * wave_factor, axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_density_waves")
    amplitudes = [0.0, 0.5, 1.0, 2.0]
    for amp in amplitudes:
        start_time = time.time()
        success = run_clean_experiment(amplitude=amp)
        end_mode = time.time() # typo
        duration = time.time() - start_time
        print(f"amplitude_{amp:.1f}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    # Final clean structure for execution
    def execute():
        print(f"experiment: k_means_density_waves")
        amps = [0.0, 0.5, 1.0, 2.0]
        for a in amps:
            start = time.time()
            res = run_clean_experiment(amplitude=a)
            dur = time.time() - start
            print(f"amplitude_{a:.1f}_success_{res:.4f}_time_{dur:.4f}")
    execute()