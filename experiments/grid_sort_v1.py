import numpy as np
import time

def run_2d_grid_sort(n_side=10, iterations=50):
    """
    Investigates '2D Grid Sorting': Can agents in a 2D grid, 
    using only local value-based swaps/interactions, achieve a 
    globally sorted state (e.g., values increasing with x and y coordinates)?
    """
    n_agents = n_side * n_side
    # Initialize agents with random values in [0, 1]
    values = np.random.rand(n_agents)
    
    # Map indices to 2D grid coordinates
    coords = []
    for i in range(n_side):
        for j in range(n_side):
            coords.append((i, j))
    coords = np.array(coords)

    # Target: values should match the order of coordinates (lexicographical)
    target_values = np.linspace(0, 1, n_agents)
    
    start_time = time.time()
    for _ in range(iterations):
        new_values = values.copy()
        # Each agent looks at its immediate Von Neumann neighbors (Up, Down, Left, Right)
        for i in range(n_agents):
            curr_coord = coords[i]
            neighbors_idx = []
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                neighbor_coord = curr_coord + [dx, dy]
                if 0 <= neighbor_coord[0] < n_side and 0 <= neighbor_coord[1] < n_side:
                    # Find index of this neighbor
                    idx = (neighbor_coord[0] * n_side) + neighbor_coord[1]
                    neighbors_idx.append(idx)
            
            if neighbors_idx:
                # Local rule: if neighbor has a value that 'should' be elsewhere, try to swap/average
                # Here we simulate a simple diffusion-based sorting (heat equation style)
                neighbor_vals = values[neighbors_idx]
                target_val = target_values[i]
                
                # If current value is far from target, move towards neighbor average
                avg_neighbor = np.mean(neighbor_vals)
                new_values[i] = values[i] + 0.5 * (avg_neighbor - values[i])
        
        values = new_values

    # Success metric: Correlation between agent value and its grid position rank
    # We check how well the final values match the target spatial distribution
    final_error = np.mean(np.abs(values - target_values))
    success = 1.0 / (1.0 + final_error)
    
    duration = time.time() - start_time
    return success, duration

def run_experiment():
    print(f"experiment: grid_sort_v1")
    sizes = [5, 8, 10]
    for s in sizes:
        try:
            success, duration = run_2d_grid_sort(n_side=s)
            print(f"size_{s}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{s}_{e}")

if __name__ == "__main__":
    run_experiment()