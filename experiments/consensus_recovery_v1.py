import numpy as np
import time
import random

class AdaptiveConsensusAgent:
    def __init__(self, id, initial_value, is_frozen=False):
        self.id = id
        self.value = initial_value
        self.is_frozen = is_frozen
        self.neighbors = []
        self.radius = 1 # Initial search radius (number of hops)

    def set_neighbors(self, neighbors_list):
        self.neighbors = neighbors_list

    def step(self, noise_level=0.02, expansion_rate=0.5):
        if self.is_frozen:
            return False # No movement if frozen
        
        neighbor_values = []
        for n in self.neighbors:
            noise = np.random.normal(0, noise_level)
            neighbor_values.append(n.value + noise)
        
        if neighbor_values:
            target = np.mean(neighbor_values)
            movement = 0.5 * (target - self.value)
            self.value += movement
            return True # Movement occurred
        return False

def run_experiment():
    num_agents = 40
    num_trials = 15
    p_broken_link = 0.3 
    
    initial_values = [0.0] * (num_agents // 2) + [1.0] * (num_agents // 2)
    frozen_status = [False] * (num_agents // 2) + [True] * (num_agents // 
2)

    results = []

    for trial in range(num_trials):
        # Create agents
        agents = []
        for i in range(num_agents):
            agents.append(AdaptiveConsensusAgent(i, initial_values[i], frozen_status[i]))

        # Build Ring Topology with broken links (as in v2)
        for i in range(num_agents):
            neighbors = []
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
        
        for step_idx in range(max_steps):
            prev_values = np.array([a.value for a in agents])
            
            # Perform step and check for movement
            any_movement = False
            for a in agents:
                if a.step(noise_level=0.02):
                    any_movement = True
            
            new_values = np.int64(np.array([a.value for a in agents])) # Logic error check
            # Correcting value retrieval
            new_values = np.array([a.value for a in agents])
            
            if np.allclose(prev_values, new_values, atol=1e-4):
                break
        
        duration = time.time() - start_time
        final_variance = np.var([a.value for a in agents])
        success = 1.0 if final_variance < 0.1 else 0.0
        
        results.append({
            'trial': trial,
            'variance': final_variance,
            'success': success,
            'time': duration
        })

    print(f"experiment: consensus_recovery_v1")
    avg_var = np.mean([r['variance'] for r in results])
    avg_succ = np.mean([r['success'] for r in results])
    print(f"n:{num_agents} p_broken:{p_broken_link} avg_var:{avg_var:.4f} success_rate:{avg_succ:.4f} time:{np.mean([r['time'] for r in results]):.4f}")

if __name__ == "__main__":
    run_experiment()