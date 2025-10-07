import time
from pathlib import Path

import numpy as np
import taichi as ti
import json

from .base_solver import SIMULATOR
from .utils import format_param_for_path
import os
from .unstruct_mpm_utils.helpers import ADV_TYPE

from .unstruct_mpm_utils.SIM_DOMAINs.vibration_bar import SIM_DOMAIN_VIBRATION_BAR
from .unstruct_mpm_utils.SIM_DOMAINs.cantilever import SIM_DOMAIN_CANTILEVER
from .unstruct_mpm_utils.SIM_DOMAINs.disk_collision import SIM_DOMAIN_DISK_COLLISION

CASE_HANDLER = {
    "vibration_bar": SIM_DOMAIN_VIBRATION_BAR,
    "cantilever": SIM_DOMAIN_CANTILEVER,
    "disk_collision": SIM_DOMAIN_DISK_COLLISION,
}

DEVICE_HANDLER = {"cpu": ti.cpu, "gpu": ti.gpu, "cuda": ti.cuda}

class UNSTRUCT_MPM(SIMULATOR):
    """2D Fluid Simulation with configurable parameters"""

    def __init__(self, args):
        """
        Initialize fluid simulation with args object

        Args:
            args: Configuration object with simulation parameters
        """
        # Store parameters from args object
        self._initialize_taichi(args)
        SIM_DOMAIN_CLASS = CASE_HANDLER[args["case"]]
        self.case = args["case"]
        
        self.nx = args["nx"]
        self.n_part = args["n_part"]
        self.cfl = args["cfl"]
        self.radii = 1.0
        
        self.flip_ratio = args["flip_ratio"]
        self.advect_scheme = ADV_TYPE(args["advect_scheme"])
        self.verbose = args["verbose"]
        self.device = args["device"]
        
        self.dump_dir = args["dump_dir"]
        self.output_path = None
        self._initialize_output(self.dump_dir)
        
        self.envs_params = args["envs_params"]
        
        self.sim_domain = SIM_DOMAIN_CLASS(self.nx, self.n_part, self.cfl, self.radii, self.flip_ratio, self.advect_scheme, self.verbose, self.output_path, self.envs_params)
        
        self.cost = 0
        self.is_converged = False

        if self.verbose:
            self._print_config()

    def _initialize_taichi(self, args):
        """Initialize Taichi backend"""
        ti.init(
            arch=DEVICE_HANDLER[args["device"]],
            debug=False,
            default_fp=ti.f64,
            random_seed=1000,
            device_memory_GB=20,
        )

    def _initialize_output(self, dump_dir):
        """Initialize output paths"""
        self.output_path = (
            dump_dir
            + f"_nx{format_param_for_path(self.nx)}_npart{self.n_part}_cfl{format_param_for_path(self.cfl)}"
        )
        self.output_path = Path(self.output_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _print_config(self):
        """Print simulation configuration"""
        print(
            f"CASE: {self.case}\n"
            f"NX: {format_param_for_path(self.nx)}\n"
            f"N_PART: {self.n_part}\n"
            f"CFL: {format_param_for_path(self.cfl)}\n"
            f"RADII: {format_param_for_path(self.radii)}\n"
            f"FLIP RATIO: {format_param_for_path(self.flip_ratio)}\n"
            f"ADVECT SCHEME: {self.advect_scheme.value}\n"
            f"VERBOSE: {self.verbose}\n"
            f"DEVICE: {self.device}\n"
            f"DUMP DIR: {self.dump_dir}"
        )
    
    def _dump_meta(self):
        meta = {
            "cost": self.sim_domain.cost,
            "n_particles": self.sim_domain.ctrl_data.n_particles,
            "sum_each_part_neighbor_communication": self.sim_domain.mpm_field.sum_each_part_neighbor_communication[0],
            "frame_dt": self.sim_domain.ctrl_data.frame_dt,
            "dt": self.sim_domain.ctrl_data.dt,
            "end_frame": self.sim_domain.ctrl_data.end_frame,
            "output_dir": str(self.output_path),
            "n_part": self.n_part,
            "cfl": self.cfl,
            "radii": self.radii,
            "flip_ratio": self.flip_ratio,
            "is_converged": self.is_converged
        }
        with open(os.path.join(self.sim_domain.ctrl_data.cache_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

    
    def run(self):
        self.sim_domain.run()
        self.is_converged = self.sim_domain.is_converged
        self._dump_meta()
        