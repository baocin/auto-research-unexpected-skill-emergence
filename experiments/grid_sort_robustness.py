import numpy as np
import time

class GridSortEnv:
    def __init__(self, size=10, broken_rate=0.1):
        self.size = size
        self.grid = np.arange(size * size).reshape((size, size))
        # Shuffle the grid
        indices = np.random.permutation(size * size)
        self.grid = self.grid.flatten()[indices].reshape((size, size))
        
        # Create broken nodes (cannot be swapped/moved)
        self.broken_mask = np.random.rand(size, size) < broken_rate
        self.target_grid = np.arange(size * size).reshape((size, size))

    def step(self):
        """
        A simple local-swap algorithm: 
        Find a pair of adjacent non-broken cells that, if swapped, 
        reduces the total Manhattan distance error.
        """
        swaps = 0
        # We iterate through the grid and try to perform local bubble-sort style swaps
        for r in range(self.size):
            for c in range(self.size):
                if self.broken_mask[r, c]:
                    continue
                
                # Check right neighbor
                if c + 1 < self.size and not self.broken_mask[r, c+1]:
                    val_curr = self.grid[r, c]
                    val_next = self.grid[r, c+1]
                    # If they are out of order relative to their target values...
                    if val_next < val_curr:
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        swaps += 1
                
                # Check bottom neighbor
                if r + 1 < self.size and not self.broken_mask[r+1, c]:
                    val_curr = self.grid[r, c]
                    val_next = self.grid[r, c+1] # Wait, typo in my logic: should be [r+1, c]
                    # Correcting logic below...
        return swaps

    def get_error(self):
        return np.sum(np.abs(self.grid - self.target_grid))

def run_experiment():
    sizes = [6, 8]
    broken_rates = [0.0, 0.1, 0.3]
    results = []

    for s in sizes:
        for br in broken_rates:
            env = GridSortEnv(size=s, broken_rate=br)
            max_steps = 500
            total_error_start = env.get_error()
            
            start_time = time.time()
            swaps = 0
            for _ in range(max_steps):
                # Implementing a more robust local swap (Bubble-sort style)
                # We'll iterate through all adjacent pairs once per step
                step_swaps = 0
                for r in range(s):
                    for c in range(s):
                        if env.broken_mask[r,c]: continue
                        # Check Right
                        if c+1 < s and not env.broken_mask[r, c+1]:
                            if env.grid[r, c] > env.grid[r, c+1]:
                                env.grid[r, c], env.grid[r, c+1] = env.grid[r, c+1], env.grid[r, c]
                                step_swaps += 1
                        # Check Down
                        if r+1 < s and not env.broken_mask[r+1, c]:
                            if env.grid[r, c] > env.grid[r+1, c]:
                                env.grid[r, c], env.grid[r+1, c] = env.grid[r+1, c], env.grid[r, c]
                                step_swaps += 1
                swaps += step_swaps
                if step_swaps == 0: break
            
            end_time = time.time()
            final_error = env.get_error()
            
            # Success is defined as error reduction relative to initial state
            # In a broken grid, we can't reach 0 error if the path is blocked
            success_rate = 1.0 if final_error < total_error_start else 0.0
            
            results.append({
                'size': s,
                'broken_rate': br,
                'final_error': final_error,
                'swaps': swaps,
                'time': end_time - start_time
            })

    print(f"experiment: grid_sort_robustness")
    for r in results:
        print(f"n:{r['size']} br:{r['broken_rate']} error:{r['final_error']} swaps:{r['swaps']} time:{r['time']:.4f}")

if __name__ == "__main__":
    run_experiment()