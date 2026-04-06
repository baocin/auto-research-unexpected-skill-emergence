import numpy as np
import time
import random

class ConsensusAgent:
    def __init__(self, id, initial_value, is_frozen=False):
        self.id = id
        self.value = initial_value
        self.is_frozen = is_frozen
        self.neighbors = []

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def step(self, noise_level=0.05):
        if self.is_frozen:
            return
        
        neighbor_values = []
        for n in self.neighbors:
            noise = np.random.normal(0, noise_level)
            neighbor_values.append(n.value + noise)
        
        if neighbor_values:
            target = np.mean(neighbor_values)
            self.value += 0.5 * (target - self.value)

def run_experiment():
    num_agents = 40
    num_trials = 15
    # Probability of a connection being broken (partitioning risk)
    p_broken_link = 0.3 
    
    initial_values = [0.0] * (num_agents // 2) + [1.0] * (num_agents // 2)
    frozen_status = [False] * (num_agents // 2) + [True] * (num_agents // 2)

    results = []

    for trial in range(num_trials):
        # Create agents for this trial
        agents = []
        for i in range(num_agents):
            agents.append(ConsensusAgent(i, initial_values[i], frozen_status[i]))

        # Build Ring Topology with broken links
        for i in range(num_agents):
            neighbors = []
            # Check clockwise and counter-clockwise neighbors
            for offset in [-1, 1]:
                neighbor_idx = (i + offset) % num_agents
                if random.random() > p_broken_link:
                    neighbors.append(agents[neighbor_idx])
            agents[i].set_neighbors(neighbors)

        # Reset values
        for i in range(num_agents):
            agents[i].value = initial_values[i]
            
        start_time = time.time()
        max_steps = 150
        
        for _ in range(max_steps):
            prev_values = np.array([a.value for a in agents])
            for a in agents:
                a.step(noise_level=0.02)
            new_values = np.array([a.value for a in agents])
            if np.allclose(prev_values, new_values, atol=1e-4):
                break
        
        duration = time.time() - start_time
        final_variance = np.var([a.value for a in agents])
        # Success: low variance AND we didn't just end up stuck in two clusters (high variance)
        # Actually, success in consensus means LOW variance. 
        # If partitioning happens, variance will be HIGH (~0.25).
        success = 1.0 if final_variance < 0.1 else 0.0
        
        results.append({
            'trial': trial,
            'variance': final_variance,
            'success': success,
            'time': duration
        })

    print(f"experiment: adv_consensus_v2_partitioning")
    avg_var = np.mean([r['variance'] for r in results])
    avg_succ = np.mean([r['success'] for r in results])
    print(f"n:{num_agents} p_broken:{p_broken_link} avg_var:{avg_var:.4f} success_rate:{avg_succ:.4f} time:{np.mean([r['time'] for r in results]):.4f}")

if __name__ == "__main__":
    import os
    print(f"Executing: {os.path.abspath(__file__)}")
    run_experiment()