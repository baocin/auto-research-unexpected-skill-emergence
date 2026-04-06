import numpy as np
import time
import json

def run_search_experiment(array_size=30, num_trials=20, max_steps=5000, damage_rate=0.2, mode="random_walk"):
    """
    Experiment: Target Search in a Damaged Environment.
    Modes: 
    - 'random_walk': Pure random movement.
    - 'scent_gradient': Agent senses if target is to the left or right and moves accordingly.
    - 'wall_follower': Tries to move towards target, but sidesteps obstacles.
    """
    results = {
        "experiment": f"target_search_{mode}",
        "array_size": array_size,
        "damage_rate": damage_rate,
        "mode": mode,
        "trials": []
    }

    rng = np.random.default_rng()
    
    for trial in range(num_trials):
        target_idx = rng.integers(0, array_size)
        num_frozen = int(array_size * damage_rate)
        frozen_indices = set(rng.choice(range(array_size), size=num_frozen, replace=False))
        
        agent_pos = 0
        # Ensure start is not frozen for the agent to begin
        if agent_pos in frozen_indices:
            for offset in [1, -1, 2, -int(2)]:
                new_start = agent_pos + offset
                if 0 <= new_start < array_size and new_start not in frozen_indices:
                    agent_pos = new_start
                    break

        steps = 0
        found = False
        path = [agent_pos]

        while steps < max_steps:
            steps += 1
            if agent_pos == target_idx:
                found = True
                break
            
            # Decision making based on mode
            move = 0
            if mode == "random_walk":
                move = rng.choice([-1, 1])
            elif mode == "scent_gradient":
                move = 1 if target_idx > agent_pos else -1
            elif mode == "wall_follower":
                direction = 1 if target_idx > agent_pos else -1
                side_step = 1 if direction == -1 else -1
                
                potential_pos = agent_pos + direction
                if 0 <= potential_pos < array_size and potential_pos not in frozen_indices:
                    move = direction
                else:
                    # Try sidestepping to bypass obstacle
                    side_step_pos = agent_pos + side_step
                    if 0 <= side_step_pos < array_size and side_step_pos not in frozen_indices:
                        move = side_step
                    else:
                        # If both fail, try the opposite direction
                        move = -direction
            else:
                move = rng.choice([-1, 1])

            new_pos = agent_pos + move
            if 0 <= new_pos < array_size and new_pos not in frozen_indices:
                agent_pos = new_pos
                path.append(agent_pos)
            else:
                # Hit a wall or frozen cell - stay put
                path.append(agent_pos)

        results["trials"].append({
            "found": found,
            "steps": steps,
            "target_idx": target_idx,
            "final_pos": agent_pos
        })

    success_rate = sum(1 for t in results["trials"] if t["found"]) / num_trials
    avg_steps = np.mean([t["steps"] for t in results["trials"]])
    
    output = {
        "experiment": results["experiment"],
        "array_size": array_size,
        "damage_rate": damage_rate,
        "mode": mode,
        "success_rate": success_rate,
        "avg_steps": avg_steps
    }
    return output

if __name__ == "__main__":
    print("Starting Search Experiment Sweep...")
    sweep_results = []
    for mode in ["random_walk", "scent_gradient", "wall_follower"]:
        res = run_search_experiment(mode=mode)
        sweep_results.append(res)
    print(json.dumps(sweep_results))