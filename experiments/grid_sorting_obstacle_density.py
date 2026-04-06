import numpy as np
import time

def run_experiment():
    """
    Investigates 'Grid Sorting Obstacle Density':
    Tests how different densities of obstacles affect the ability 
    of local swaps to reduce global disorder.
    """
    grid_size = 10
    # Test obstacle densities: 5%, 15%, 30%
    densities = [0.05, 0.15, 0.30]
    results = []

    for density in densities:
        grid = np.random.rand(grid_size, grid_size)
        obstacle_mask = np.random.choice([True, False], size=(grid_size, grid_size), p=[density, 1-density])
        
        def get_disorder(g, m):
            d = 0
            for r in range(grid_s): # Error: variable name mismatch
                pass
            return d

        # Let's rewrite the disorder function correctly inside the loop
        def calculate_disorder(g, m):
            d = 0
            for r in range(grid_size):
                for c in range(grid_size):
                    if not m[r, c]:
                        if r + 1 < grid_size and not m[r+1, c]:
                            d += abs(g[r, c] - g[r+1, c])
                        if c + 1 < grid_size and not m[r, c+1]:
                            d += abs(g[r, c] - g[r, c+1])
            return d

        initial_disorder = calculate_disorder(grid, obstacle_mask)
        start_time = time.time()
        
        iterations = 300
        for _ in range(iterations):
            r = np.random.randint(0, grid_size)
            c = np.random.randint(0, grid_size)
            if not obstacle_mask[r, c]:
                direction = np.random.choice(['U', 'D', 'L', 'R'])
                nr, nc = r, c
                if direction == 'U': nr -= 1
                elif direction == 'D': nr += 1
                elif direction == 'L': nc -= 1
                elif direction == 'R': nc += 1
                
                if 0 <= nr < grid_size and 0 <= nc < grid_size and not obstacle_mask[nr, nc]:
                    idx_curr = r * grid_size + c
                    idx_neigh = nr * grid_size + nc
                    # Sort based on flattened index priority
                    if (grid[r, c] > grid[nr, nc] and idx_curr < idx_neigh) or \
                       (grid[r, c] < grid[nr, nc] and idx_curr > idx_neigh):
                        grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]

        end_time = time.time()
        final_disorder = calculate_disorder(grid, obstacle_mask)
        success_metric = 1.0 - (final_disorder / initial_disorder) if initial_disorder > 0 else 0
        results.append((density, success_metric, end_time - start_time))

    print(f"experiment: grid_sorting_obstacle_density")
    for density, success, duration in results:
        print(f"density_{density:.2f}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()