import os
import re

def analyze():
    print("experiment: meta_analysis_robustness")
    
    # Patterns to look for in logs
    patterns = {
        'k_means': r"n:(\d+) p_f:([\d.]+) p_m:([\d.]+) t_f:([\d.]+) t_m:([\d.]+)",
        'anomaly': r"sep:([\d.]+) f1:([\d.]+) t:([\d.]+)"
    }
    
    all_results = []

    # Scan the experiments directory for logs or known experiment files
    # For this agent, I will manually parse the results I just generated 
    # since I don't have a central database yet.
    
    # Data from current session:
    k_means_data = [
        {'n': 500, 'pf': 0.9408, 'pm': 0.9240},
        {'n': 1000, 'pf': 0.9336, 'pm': 0.9278},
        {'n': 2000, 'pf': 0.9403, 'pm': 0.9367}
    ]
    
    anomaly_data = [
        {'sep': 2.0, 'f1': 0.0880},
        {'sep': 3.0, ' ' : 0.1840}, # Note: I'll use the actual values from grep
        {'sep': 4.0, 'f1': 0.2900},
        {'sep': 5.0, 'f1': 0.4160},
        {'sep': 6.0, 'f1': 0.6020}
    ]
    # Actually, I'll just re-parse the actual grep outputs if they were saved.
    # Since I can't easily 're-read' previous terminal outputs without files,
    # I will look for any .log files in the current dir.

    for filename in os.listdir('.'):
        if filename.endswith('.log'):
            with open(filename, 'r') as f:
                content = f.read()
                matches = re.findall(patterns['k_means'], content)
                for m in matches:
                    all_results.append(f"K-Means Scale {m[0]}: pf={m[1]} pm={m[2]}")
                
                matches_anom = re.findall(patterns['anomaly'], content)
                for m in matches_anom:
                    all_results.append(f"Anomaly Sep {m[0]}: f1={m[1]}")

    if not all_results:
        # Fallback for this specific execution if logs weren't captured in one file
        print("No log files found to parse.")
    else:
        for res in all_results:
            print(res)

if __name__ == "__main__":
    analyze()