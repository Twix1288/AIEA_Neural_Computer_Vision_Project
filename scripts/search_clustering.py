import os
import subprocess
import json

def run_clustering(algorithm, params, random_units=3):
    cmd = [
        "python3", "scripts/run_clustering.py",
        f"--clustering_algo={algorithm}",
        f"--clustering_params={json.dumps(params)}",
        f"--random_units={random_units}",
        "--target_layers=layer4",
        "--seed=42" # Use a fixed seed for reproducibility across runs
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {algorithm} with {params}:")
        print(result.stderr)
        return None

    # Parse output to extract labels
    labels = []
    for line in result.stdout.split('\n'):
        if line.startswith("RESULT:"):
            # FORMAT: RESULT: layer4 | Unit 5 - <label> - Cluster 4
            parts = line.split(" - ")
            if len(parts) >= 2:
                label = parts[1].strip()
                labels.append(label)
    
    return labels

def is_meaningful(labels):
    if not labels:
        return False
    # Check if all labels are the same (degenerate behavior)
    unique_labels = set(labels)
    if len(unique_labels) == 1:
        return False
    return True

def calculate_difference(base_labels, test_labels):
    if not base_labels or not test_labels or len(base_labels) != len(test_labels):
        return 0.0
    
    diff_count = 0
    for b, t in zip(base_labels, test_labels):
        if b != t:
            diff_count += 1
            
    return diff_count / len(base_labels)

def main():
    print("--- Running Baseline (KMeans) ---")
    base_labels = run_clustering("kmeans", {})
    if not base_labels:
        print("Failed to run baseline. Exiting.")
        return
        
    print(f"Baseline labels: {base_labels}")
    
    configurations = [
        ("minibatch_kmeans", {"batch_size": 256}),
        ("minibatch_kmeans", {"batch_size": 1024}),
        ("bisecting_kmeans", {"bisecting_strategy": "biggest_inertia"}),
        ("agglomerative", {"linkage": "ward"}),
        ("agglomerative", {"linkage": "average"}),
        ("gmm", {"covariance_type": "full"}),
        ("gmm", {"covariance_type": "diag"}),
        ("kmeans", {"init": "random", "n_init": 10}),
        ("kmeans", {"algorithm": "elkan"})
    ]
    
    best_config = None
    best_diff = 0.0
    
    print("\n--- Testing Configurations ---")
    for algo, params in configurations:
        print(f"\nTesting {algo} with {params}")
        test_labels = run_clustering(algo, params)
        
        if not test_labels:
            continue
            
        print(f"Labels: {test_labels}")
        
        if not is_meaningful(test_labels):
            print("=> NOT MEANINGFUL (Degenerate behavior detected)")
            continue
            
        diff = calculate_difference(base_labels, test_labels)
        print(f"=> Difference: {diff*100:.2f}%")
        
        if diff >= 0.33:
            print("=> FOUND A PROMISING CONFIGURATION!")
            best_config = (algo, params)
            best_diff = diff
            break
            
    if best_config:
        print(f"\nSUCCESS: Found configuration {best_config[0]} with {best_config[1]} (Difference: {best_diff*100:.2f}%)")
        print("You can run it on all 10 units now:")
        print(f"python3 scripts/run_clustering.py --clustering_algo={best_config[0]} --clustering_params='{json.dumps(best_config[1])}' --random_units=10")
    else:
        print("\nCould not find a configuration with >= 33% difference that is also meaningful.")
        print("All tested configurations have been reported above.")

if __name__ == "__main__":
    main()
