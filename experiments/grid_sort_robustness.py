import numpy as np
import time

def run_experiment():
    """
    Tests local sorting on a 2D grid. Each cell looks at its neighbors 
    (up, down, left, right) and attempts to propagate values.
    We measure the 'disorder' (variance of local differences) after N steps.
    """
    grid_size = 10
    num_trials = 5
    iterations = 50
    results = []

    for _ in range(num_trials):
        # Initialize grid with random values
        grid = np.random.rand(grid_size, grid_size)
        
        s_time = time.time()
        for _ in range(iterations):
            new_grid = grid.copy()
            for r in range(grid_size):
                for c in range(grid_size):
                    # Check neighbors (4-connectivity)
                    neighbor_coords = []
                    if r > 0: neighbor_coords.append((r-1, c))
                    if r < grid_size - 1: neighbor_coords.append((r+1, c))
                    if c > 0: neighbor_coords.append((r, c-1))
                    if c < grid_size - 1: neighbor_coords.append((r, c+1))
                    
                    # The "Sorting" rule: a cell tries to move its value towards the local mean
                    # This is a diffusion process that mimics sorting/smoothing
                    neighbor_vals = [grid[nr, nc] for nr, nc in neighbor_coords]
                    if neighbor_vals:
                        new_grid[r, c] = 0.8 * grid[r, c] + 0.2 * np.mean(neighbor_vals)
            
            grid = new_grid

        t_elapsed = time.time() - s_time
        # Metric: Standard deviation of the final grid (measure of remaining 'disorder')
        disorder = np.std(grid)
        
        results.append({
            'disorder': disorder,
            'time': t_elapsed
        })

    print("experiment: grid_sort_robustness")
    for r in results:
        print(f"disorder:{r['disorder']:.6f} t:{r['time']:.6f}")

if __name__ == "__main__":
    run_experiment()