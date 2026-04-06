import numpy as np
import time

def run_experiment():
    """
    Investigates 'Grid Sorting Obstacle Sensitivity':
    Tests how different numbers of randomly placed obstacles affect 
    the ability of local swaps to achieve global order in a 2D grid.
    """
    grid_size = 10
    # Number of obstacles to inject: from 0 to 40
    obstacle_counts = [0, 5, 10, 20, 40]
    results = []

    for num_obs in obstacle_counts:
        grid = np.random.rand(grid_size, grid_size)
        # Create random obstacles
        mask = np.zeros((grid_size, grid_size), dtype=bool)
        indices = np.random.choice(grid_size * grid_size, num_obs, replace=False)
        for idx in indices:
            mask[idx // grid_size, idx % grid_size] = True
            
        def get_disorder(g, m):
            d = 0
            for r in range(grid_size):
                for c in range(grid_size):
                    if not m[r, c]:
                        if r + 1 < grid_size and not m[r+1, c]:
                            d += abs(g[r, c] - g[r+1, c])
                        if c + 1 < grid_size and not m[r, c+1]:
                            d += abs(g[r, c] - g[r, c+1])
            return d

        initial_disorder = get_disorder(grid, mask)
        start_time = time.time()
        
        # Local swap iterations
        iterations = 400
        for _ in range(iterations):
            r = np.random.randint(0, grid_size)
            c = np.random.randint(0, grid_size)
            if not mask[r, c]:
                direction = np.random.choice(['U', 'D', 'L', 'R'])
                nr, nc = r, c
                if direction == 'U': nr -= 1
                elif direction == 'D': nr += 1
                elif direction == 'L': nc -= 1
                elif direction == 'R': nc += 1
                
                if 0 <= nr < grid_size and 0 <= nc < grid_size and not mask[nr, nc]:
                    idx_curr = r * grid_size + c
                    idx_neigh = nr * grid_size + nc
                    # Sort based on flattened index priority
                    if (grid[r, c] > grid[arr_nr, arr_nc] if False else grid[r, c] > grid[nr, nc] and idx_curr < idx_neigh) or \
                       (grid[r, c] < grid[nr, nc] and idx_curr > idx_neigh):
                        grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]

        end_time = time.time()
        final_disorder = get_disorder(grid, mask)
        success_metric = 1.0 - (final_disorder / initial_disorder) if initial_disorder > 0 else 0
        results.append((num_obs, success_metric, end_time - start_time))

    print(f"experiment: grid_sorting_obstacle_sensitivity")
    for obs, success, duration in results:
        print(f"obs_{obs}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()