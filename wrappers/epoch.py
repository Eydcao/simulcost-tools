import os
import subprocess
import h5py
import numpy as np
import json
from scipy.interpolate import RegularGridInterpolator

def _find_runner_path():
    """Automatically find the correct path to epoch.py runner."""
    # Get current working directory
    cwd = os.getcwd()

    # List of possible runner paths relative to different working directories
    possible_paths = []

    # If working from project root (SimulCost-Bench/)
    if cwd.endswith('SimulCost-Bench'):
        possible_paths.extend([
            "costsci_tools/runners/epoch.py",
            "runners/epoch.py"
        ])
    # If working from costsci_tools/ subdirectory
    elif cwd.endswith('costsci_tools') or 'costsci_tools' in cwd:
        possible_paths.extend([
            "runners/epoch.py",
            "../runners/epoch.py",
            "costsci_tools/runners/epoch.py"
        ])

    # Add generic fallback paths
    possible_paths.extend([
        "runners/epoch.py",
        "costsci_tools/runners/epoch.py",
        "./runners/epoch.py",
        "../runners/epoch.py",
        "../../runners/epoch.py"
    ])

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in possible_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    for path in unique_paths:
        if os.path.exists(path):
            return path

    # If none found, raise an error with helpful information
    raise FileNotFoundError(
        f"Could not find epoch.py runner in any expected location.\n"
        f"Current working directory: {cwd}\n"
        f"Searched paths: {unique_paths}\n"
        f"Please ensure the runner exists or update the search paths."
    )


def runEpoch(profile, nx, dt_mult, nPart, field_order, particle_order):
    dir_path = (
        f"sim_res/epoch/{profile}_nx_{nx}_dtmult_{dt_mult}_part_{nPart}_fieldO_{field_order}_partO_{particle_order}/"
    )
    meta_file = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                return meta["cost"]

    # Run the simulation if not already done
    runner_path = _find_runner_path()
    cmd = f"python {runner_path}  --config-name={profile} nx={nx} dt_mult={dt_mult} part_cell={nPart} field_order={field_order} particle_order={particle_order}"
    subprocess.run(cmd, shell=True, check=True)

    # Load the cost from the meta.json file
    with open(meta_file, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_epoch(profile, nx, dt_mult, nPart, field_order, particle_order):

    dir_path = (
        f"sim_res/epoch/{profile}_nx_{nx}_dtmult_{dt_mult}_part_{nPart}_fieldO_{field_order}_partO_{particle_order}/"
    )

    meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), dir_path)
    results_file = os.path.join(meta_path, "res.h5")
    meta_file = os.path.join(meta_path, "meta.json")

    if not os.path.exists(results_file):
        runEpoch(profile, nx, dt_mult, nPart, field_order, particle_order)

    results = []
    with h5py.File(results_file, "r") as f:

        results = {
            "Density_Time": f["Density_Time"][:],
            "Ey": f["Ey"][:],
            "Ey_Time": f["Ey_Time"][:],
            "Density": f["n_elec"][:],
            "X_Grid": f["x_grid"][:],
        }
    with open(meta_file, "r") as f:
        meta = json.load(f)
        a0 = meta["a0"]
        lambda_laser = meta["lambda"]

    m_cgs = 9.10938356 * 10 ** (-28)
    q_cgs = -4.8032 * 10 ** (-10)
    c_cgs = 2.99792458 * 10 ** (10)
    k_cgs = lambda lambda_cgs: 2 * np.pi / lambda_cgs
    w_cgs = lambda lambda_cgs: c_cgs * k_cgs(lambda_cgs)

    E0_cgs = lambda a0, lambda_cgs: a0 * m_cgs * c_cgs * w_cgs(lambda_cgs) / abs(q_cgs)
    E0_si = lambda a0, lambda_cgs: E0_cgs(a0, lambda_cgs) / (1 / 3 * 10 ** (-4))

    return results, E0_si(a0, lambda_laser * 10**-4)


# return results


def compare_res_epoch(profile1, nx1, dt_mult1, nPart1, fO1, pO1, profile2, nx2, dt_mult2, nPart2, fO2, pO2, tolerance):

    res1, E01 = get_res_epoch(profile1, nx1, dt_mult1, nPart1, fO1, pO1)
    res2, E02 = get_res_epoch(profile2, nx2, dt_mult2, nPart2, fO2, pO2)

    x1 = res1["X_Grid"]
    t1 = res1["Ey_Time"]

    x2 = res2["X_Grid"]
    t2 = res2["Ey_Time"]

    EY1 = res1["Ey"]
    EY2 = res2["Ey"]

    # interpolate data1 onto 2nd grid if necessary
    if x1.size != x2.size:
        XG2, tG2 = np.meshgrid(x2, t2)

        interp = RegularGridInterpolator((x1, t1), EY1, bounds_error=False, fill_value=None)
        pts = np.column_stack((XG2.ravel(), tG2.ravel()))

        EY1 = np.transpose(interp(pts).reshape(XG2.shape))

    L2Ey = np.sqrt((1 / EY1.size) * np.sum((EY1 / E01 - EY2 / E01) ** 2))

    print(f"L2 Ey: {L2Ey}")

    return L2Ey < tolerance, L2Ey


if __name__ == "__main__":
    # test method
    compare_res_epoch("p1", 4000, 0.96, 10, 2, 3, "p1", 3200, 0.96, 10, 2, 3, 0.2)
