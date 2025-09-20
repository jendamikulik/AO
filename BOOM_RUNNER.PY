# runner.py
import subprocess
import os
import argparse
import json
from datetime import datetime

# Path to your BOOM.PY script
BOOM_SCRIPT = "BOOM.PY"

# Base parameters for the spectral tester
BASE_PARAMS = {
    'C': 500,  # Number of clauses (adjusted per file)
    'cR': 15.0,
    'rho_lock': 0.50,
    'zeta0': 0.40,
    'deg': 4,
    'seed': 42,
    'tau': 0.80
}

def run_test(mode, filename, file_params):
    """Run BOOM.PY for a given mode and file."""
    params = BASE_PARAMS.copy()
    params.update(file_params)
    
    cmd = [
        "python", BOOM_SCRIPT,
        f"--mode={mode}",
        f"--C={params['C']}",
        f"--cR={params['cR']}",
        f"--rho_lock={params['rho_lock']}",
        f"--zeta0={params['zeta0']}",
        f"--deg={params['deg']}",
        f"--seed={params['seed']}",
        f"--tau={params['tau']}"
    ]
    
    print(f"\n--- Running test for {filename} (Mode: {mode}) ---")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Parse the output to get key metrics
        mu = float(result.stdout.split("mu: ")[1].splitlines()[0])
        decision = result.stdout.split("Decision")[1].split(":")[1].strip().splitlines()[0].strip()
        
        return {
            "file": filename,
            "mode": mode,
            "mu": mu,
            "decision": decision
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running test for {filename}: {e}")
        return None
    
def get_file_info(filepath):
    """Extract C from DIMACS file header."""
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('p cnf'):
                parts = line.split()
                return {"C": int(parts[3])}
    return None

def main():
    parser = argparse.ArgumentParser(description="AO Framework SATLIB Runner")
    parser.add_argument("--path", type=str, default="./satlib", help="Path to SATLIB directory")
    args = parser.parse_args()

    results = []
    
    # Process UF (Satisfiable) files
    uf_path = os.path.join(args.path, "uf20")
    if os.path.exists(uf_path):
        print(f"--- Processing UF (SAT) files in {uf_path} ---")
        for filename in os.listdir(uf_path):
            if filename.endswith(".cnf"):
                filepath = os.path.join(uf_path, filename)
                info = get_file_info(filepath)
                if info:
                    results.append(run_test("sat", filename, info))

    # Process UUF (Unsatisfiable) files
    uuf_path = os.path.join(args.path, "uuf50")
    if os.path.exists(uuf_path):
        print(f"\n--- Processing UUF (UNSAT) files in {uuf_path} ---")
        for filename in os.listdir(uuf_path):
            if filename.endswith(".cnf"):
                filepath = os.path.join(uuf_path, filename)
                info = get_file_info(filepath)
                if info:
                    results.append(run_test("unsat", filename, info))
    
    # Generate summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n--- Test run complete. Full report saved to {report_file} ---")
    
if __name__ == "__main__":
    main()