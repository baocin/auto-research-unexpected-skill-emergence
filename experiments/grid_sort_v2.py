import numpy as np
import time

def run_2d_grid_sort_v2(n_side=10, iterations=50):
    """
    Investigates '2D Grid Sorting' using a discrete swap-based approach 
    instead of diffusion to see if local swaps can achieve global order.
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
        # For each agent, try to swap with a neighbor if it helps sorting
        for i in range(n_agents):
            curr_coord = coords[i]
            neighbors_idx = []
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                neighbor_coord = curr_coord + [dx, dy]
                if 0 <= neighbor_coord[0] < n_side and 0 <= neighbor_coord[1] < n_side:
                    idx = (neighbor_coord[0] * n_side) + neighbor_coord[1]
                    neighbors_idx.append(idx)
            
            for n_idx in neighbors_idx:
                # If the neighbor has a value that 'belongs' at index i, swap them
                # This is a decentralized bubble-sort variant
                if values[n_idx] < values[i]:
                    values[i], values[n_idx] = values[n_idx], values[i]

    # Success metric: Correlation between agent value and its target rank
    final_error = np.mean(np.abs(values - target_values))
    success = 1.0 / (1.0 + final_error)
    
    duration = time.time() - start_time
    return success, duration

def run_experiment():
    print(f"experiment: grid_sort_v2")
    sizes = [4, 6, 8]
    for s in sizes:
        try:
            success, duration = run_2d_grid_sort_v2(n_side=s)
            print(f"size_{s}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{s}_{e}")

if __name__ == "__main__":
    run_experiment()