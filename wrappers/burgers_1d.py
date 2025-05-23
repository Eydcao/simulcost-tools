import os
import subprocess
import h5py
import numpy as np
import json


def run_sim_burgers_1d(profile, cfl, k, w):
    """Run the burgers_1d simulation with the given parameters if not already simulated."""
    dir_path = f"sim_res/burgers_1d/{profile}_cfl_{cfl}_k_{k}_w_{w}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    print(f"Running new simulation with parameters: cfl={cfl}, k={k}, w={w}")
    cmd = f"python runners/burgers_1d.py --config-name={profile} cfl={cfl} k={k} w={w}"
    subprocess.run(cmd, shell=True, check=True)

    # Load the cost from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_burgers_1d(profile, cfl, k, w):
    """Load all time frames for a given parameter set."""
    dir_path = f"sim_res/burgers_1d/{profile}_cfl_{cfl}_k_{k}_w_{w}/"
    results = []
    X = None

    # Find number of frames by counting res_*.h5 files
    count = 0
    while os.path.exists(os.path.join(dir_path, f"res_{count}.h5")):
        file_path = os.path.join(dir_path, f"res_{count}.h5")
        with h5py.File(file_path, "r") as f:
            results.append(np.array(f["u"]))
            if X is None:
                X = np.array(f["x"])
        count += 1

    return np.array(results), X


def compare_res_burgers_1d(profile1, cfl1, k1, w1, profile2, cfl2, k2, w2, tolerance):
    """Compare two sets of results by comparing the final frames."""
    res1, x1 = get_res_burgers_1d(profile1, cfl1, k1, w1)
    res2, x2 = get_res_burgers_1d(profile2, cfl2, k2, w2)

    # Compare the final frames
    sol1 = res1[-1]
    sol2 = res2[-1]

    # Calculate differences directly
    diff = np.abs(sol1 - sol2)

    # Calculate L1 and L∞ norms of the difference
    l1_norm = np.mean(diff)
    linf_norm = np.max(diff)

    print(f"L1 norm of difference: {l1_norm}")
    print(f"Linfinity norm of difference: {linf_norm}")

    # Check if the solution has converged based on the tolerance
    converged = l1_norm < tolerance

    return converged, l1_norm
