import numpy as np
import time

class ReroutingEnv:
    def __init__(self, size=6, broken_rate=0.2):
        self.size = size
        self.grid = np.arange(size * size).reshape((size, size))
        indices = np.random.permutation(size * size)
        self.grid = self.grid.flatten()[indices].reshape((size, size))
        self.broken_mask = np.random.rand(size, size) < broken_rate
        self.target_grid = np.arange(size * size).reshape((size, size))

    def get_error(self):
        return np.sum(np.abs(self.grid - self.target_grid))

    def step_with_reroute(self):
        """
        Attempts to move a value not just towards its target, but 
        actively 'flowing' around broken cells using local neighbor checks.
        """
        swaps = 0
        # We perform multiple passes to simulate "flow"
        for _ in range(3): 
            step_swaps = 0
            for r in range(self.size):
                for c in range(self.size):
                    if self.broken_mask[r,c]: continue
                    
                    # Neighbors: Right, Down, Left, Up
                    neighbors = [(r, c+1), (r+1, c), (r, c-1), (r-1, c)]
                    
                    for nr, nc in neighbors:
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if not self.broken_mask[nr, nc]:
                                # If the current value is 'greater' than its neighbor 
                                # (meaning it belongs elsewhere/further along), swap it.
                                if self.grid[r, c] > self.grid[nr, nc]:
                                    self.grid[r, c], self.grid[nr, nc] = \
                                        self.grid[nr, nc], self.grid[r, c]
                                    step_swaps += 1
            swaps += step_swaps
        return swaps

def run_experiment():
    sizes = [6]
    broken_rates = [0.0, 0.2, 0.4]
    results = []

    for s in sizes:
        for br in broken_rates:
            env = ReroutingEnv(size=s, broken_rate=br)
            start_error = env.get_error()
            
            start_time = time.time()
            total_swaps = 0
            for _ in range(100): # Max steps
                s_swaps = env.step_with_reroute()
                if s_swaps == 0: break
                total_swaps += s_swaps
            end_time = time.time()
            
            final_error = env.get_error()
            # Efficiency metric: Error reduction ratio
            reduction = (start_error - final_error) / start_error if start_error > 0 else 1.0

            results.append({
                'size': s, 'br': br, 
                'reduction': reduction, 
                'swaps': total_swaps, 
                'time': end_time - start_time
            })

    print(f"experiment: grid_rerouting_v1")
    for r in results:
        print(f"n:{r['size']} br:{r['br']} red:{r['reduction']:.4f} swaps:{r['swaps']} time:{r['time']:.4f}")

if __name__ == "__main__":
    run_experiment()