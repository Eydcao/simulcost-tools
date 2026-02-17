import os
import subprocess
import h5py
import numpy as np
import json
from scipy.interpolate import RegularGridInterpolator
import sys
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Get base directory for simulation results from environment variable
# If not set, use current directory (maintains backward compatibility)
SIM_RES_BASE_DIR = os.getenv("SIM_RES_BASE_DIR", None)
if SIM_RES_BASE_DIR:
  print(f"✅ Using custom simulation results directory: {SIM_RES_BASE_DIR}")

def _get_sim_path(relative_path):
  """Construct simulation path, using absolute path if SIM_RES_BASE_DIR is set."""
  if SIM_RES_BASE_DIR:
      return os.path.join(SIM_RES_BASE_DIR, relative_path)
  return relative_path

def _find_runner_path():
  """Automatically find the correct path to cgyro.py runner."""
  # Get current working directory
  cwd = os.getcwd()

  # List of possible runner paths relative to different working directories
  possible_paths = []

  # If working from project root (SimulCost-Bench/)
  if cwd.endswith('SimulCost-Bench'):
      possible_paths.extend([
          "costsci_tools/runners/cgyro.py",
          "runners/cgyro.py"
      ])
  # If working from costsci_tools/ subdirectory
  elif cwd.endswith('costsci_tools') or 'costsci_tools' in cwd:
      possible_paths.extend([
          "runners/cgyro.py",
          "../runners/cgyro.py",
          "costsci_tools/runners/cgyro.py"
      ])

  # Add generic fallback paths
  possible_paths.extend([
      "runners/cgyro.py",
      "costsci_tools/runners/cgyro.py",
      "./runners/cgyro.py",
      "../runners/cgyro.py",
      "../../runners/cgyro.py"
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
      f"Could not find cgyro.py runner in any expected location.\n"
      f"Current working directory: {cwd}\n"
      f"Searched paths: {unique_paths}\n"
      f"Please ensure the runner exists or update the search paths."
  )

def runCgyro(profile, n_radial, n_theta, error_tol, freq_tol, delta_t):
  dir_path = _get_sim_path(
      f"sim_res/cgyro/{profile}_n_radial_{n_radial}_n_theta_{n_theta}_freq_tol_{freq_tol}_delta_t_{delta_t}"
  )
  meta_file = os.path.join(dir_path, "meta.json")

  # Check if the simulation has already been run
  if os.path.exists(meta_file):
      with open(meta_file, "r") as f:
          meta = json.load(f)
          if "cost" in meta:
              print(f"Using existing simulation results from {dir_path}")
              return meta["cost"], meta["converged"]

  # Run the simulation if not already done
  print(f"Running new simulation with parameters: n_radial={n_radial}, n_theta={n_theta}, error_tol={error_tol}, freq_tol={freq_tol}, delta_t={delta_t}")
  runner_path = _find_runner_path()
  if SIM_RES_BASE_DIR:
      dump_dir = os.path.join(SIM_RES_BASE_DIR, f"sim_res/cgyro/{profile}")
      cmd = f"{sys.executable} {runner_path} --config-name={profile} n_radial={n_radial} n_theta={n_theta} error_tol={error_tol} freq_tol={freq_tol} delta_t={delta_t} dump_dir={dump_dir}"
  else:
      cmd = f"{sys.executable} {runner_path} --config-name={profile} n_radial={n_radial} n_theta={n_theta} error_tol={error_tol} freq_tol={freq_tol} delta_t={delta_t}"
  subprocess.run(cmd, shell=True, check=True)

  # Load the cost from the meta.json file
  with open(meta_file, "r") as f:
      meta = json.load(f)
      cost = meta["cost"]
      converged = meta["converged"]

  return cost, converged

def get_res_cgyro(profile, n_radial, n_theta, error_tol, freq_tol, delta_t):
  dir_path = _get_sim_path(
      f"sim_res/cgyro/{profile}_n_radial_{n_radial}_n_theta_{n_theta}_freq_tol_{freq_tol}_delta_t_{delta_t}"
  )

  results_file = os.path.join(dir_path, "res.h5")

  if not os.path.exists(results_file):
      print(f"No results found for parameters: n_radial={n_radial}, n_theta={n_theta}, error_tol={error_tol}, freq_tol={freq_tol}, delta_t={delta_t}. Triggering simulation.")
      cost, converged = runCgyro(profile, n_radial, n_theta, error_tol, freq_tol, delta_t)
  else:
      meta_file = os.path.join(dir_path, "meta.json")
      # Load the cost from the meta.json file
      with open(meta_file, "r") as f:
         meta = json.load(f)
         cost = meta["cost"]
         converged = meta["converged"]

  results = []
  with h5py.File(results_file, "r") as f:
     
      results = {
          "growth_rate": f["growth_rate"][:],
          "mode_frequency": f["mode_frequency"][:],
          "eigenvalues": f["eigenvalues"][:],
          "particle": f["particle"][:],
          "heat": f["heat"][:],
          "momentum": f["momentum"][:],
      }

  return results, cost, converged

def check_convergence_cgyro(profile, n_radial, n_theta, error_tol, freq_tol, delta_t):
   res, cost, converged = get_res_cgyro(profile, n_radial, n_theta, error_tol, freq_tol, delta_t)

   # Simple convergence check within each result checking in out.cgyro.info (there should be an output flag)
   # Use this as first pass, if it does NOT converge, run manual check after to verify (see below)
   if converged:
       return res
  
   eigenvalues = res['eigenvalues'].squeeze()
   # Check convergence within each result using the average of the last K (30%) timesteps,
   # and compare abs difference to last timestep and verify that it is within acceptable error tolerance
   # could also use distribution comparison to capture oscilattory behavior
   MANUAL_CONVERGENCE_LENGTH_FACTOR = 0.30
   K = int(eigenvalues.shape[0] * MANUAL_CONVERGENCE_LENGTH_FACTOR)
   last_k_eig = eigenvalues[ (-K)-1 : -2 ]
   last_k_avg_eig = np.average(last_k_eig, axis=0)
   last_k_avg_real = np.real(last_k_avg_eig)
   last_k_avg_imag = np.imag(last_k_avg_eig)

   last_eig = eigenvalues[-1]
   last_real = np.real(last_eig)
   last_imag = np.imag(last_eig)

   diff_real = np.abs(last_k_avg_real - last_real)
   diff_imag = np.abs(last_k_avg_imag - last_imag)
   if (diff_real < error_tol) and (diff_imag < error_tol):
        return res
   else:
        return None

   # WARNING: ALL MANUAL CHECKS SHOULD VERIFY REAL AND IMAG INDEPENDENTLY, NEED BOTH TO CONVERGE
   #          - (For relatively stable) imag should oscilatte around 0, real should oscillate around point
   #          - (Competing modes case) imag oscilattes around growth rate of 1st root, jumps to 2nd root (probably negative bc diff mode), etc
   #                                  - will likely see 2 attractors in growth rate + freq

   # In case of competing oscilatting modes:
   # -------------------------------------------------
   # Potential K-means-type algorithm:
   # Detect whether eigenvalue jumps from mode 1 to mode 2, based on centerpoint of mode clusters,
   # determine whether at boundary of modes or oscilatting around many different clusters because the case is stable

def compare_res_cgyro(profile1, n_radial1, n_theta1, error_tol1, freq_tol1, delta_t1, profile2, n_radial2, n_theta2, error_tol2, freq_tol2, delta_t2, tolerance):
  res1 = check_convergence_cgyro(profile1, n_radial1, n_theta1, error_tol1, freq_tol1, delta_t1)
  res2 = check_convergence_cgyro(profile2, n_radial2, n_theta2, error_tol2, freq_tol2, delta_t2)

   # if either result is None, one or both of the runs didn't converge,
   # in which case the two runs cannot converge relative to one another
  if res1 == None or res2 == None:
      return False

  eigenvalues1 = res1['eigenvalues'].squeeze()
  eigenvalues2 = res2['eigenvalues'].squeeze()

  last_eig1 = eigenvalues1[-1]
  last_eig2 = eigenvalues2[-1]

  last_real1 = np.real(last_eig1)
  last_real2 = np.real(last_eig2)
  last_imag1 = np.imag(last_eig1)
  last_imag2 = np.imag(last_eig2)

  diff_last_real = np.abs(last_real1 - last_real2)
  diff_last_imag = np.abs(last_imag1 - last_imag2)
  # Need to check real and imaginary components independently
  return (diff_last_real < tolerance) and (diff_last_imag < tolerance)

if __name__ == "__main__":
  converged = compare_res_cgyro("p1", 6, 24, 0.001, 0.001, 1, "p1", 6, 24, 0.001, 0.001, 1, 1e-4)
  print(f'Converged: {converged}')