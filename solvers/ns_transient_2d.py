import time
from pathlib import Path

import numpy as np
import taichi as ti
import h5py

from .ns_transient_2d_utils.fluid_simulator import DyeFluidSimulator, FluidSimulator
from .base_solver import SIMULATOR
import os

ASPECT_RATIO = 0.5  # y/x
MAXIMUM_WALL_TIME = 600  # 10 minutes in seconds


class NSTransient2D(SIMULATOR):
    """2D Fluid Simulation with configurable parameters"""

    def __init__(self, args):
        """
        Initialize fluid simulation with args object

        Args:
            args: Configuration object with simulation parameters
        """
        # Store parameters from args object
        self.n_bc = args.boundary_condition
        self.re = args.reynolds_num
        self.resolution = args.resolution
        self.cfl = args.cfl
        # Calculate time step from CFL: dt = cfl * dx / max_velocity
        # For typical fluid flows, max_velocity is around 1.0
        self.dx = 1 / args.resolution
        self.dt = args.cfl * self.dx  # Simplified CFL condition: dt = cfl * dx
        self.vis_num = args.visualization
        self.no_dye = args.no_dye
        self.scheme = args.advection_scheme
        self.vor_eps = args.vorticity_confinement if args.vorticity_confinement != 0.0 else None
        self.cpu = args.cpu
        self.total_runtime = args.total_runtime
        self.relaxation_factor = args.relaxation_factor
        self.residual_threshold = args.residual_threshold

        # Base solver attributes
        self.verbose = True
        self.current_time = 0.0
        self.record_dt = (
            self.total_runtime / 10 if self.total_runtime is not None else 0.1
        )  # Time interval between recordings
        self.next_record_time = 0.0
        self.end_time = self.total_runtime if self.total_runtime is not None else 1.0
        self.record_frame = 0
        self.num_steps = 0

        # Timeout tracking
        self.start_time = None
        self.max_runtime = MAXIMUM_WALL_TIME
        self.max_wall_time = None  # Disable base class wall time checking - we use our own via early_stop()
        self.wall_time_exceeded = False  # Required by base class run() method for verbose output

        # Pressure solver iteration tracking
        self.total_pressure_iterations = 0

        # Initialize simulation components
        self._initialize_taichi()
        self._initialize_simulator()
        self._initialize_output(args.dump_dir)

        # Print configuration
        self._print_config()

    def _initialize_taichi(self):
        """Initialize Taichi backend"""
        if self.cpu:
            ti.init(arch=ti.cpu)
        else:
            try:
                device_memory_GB = 2.0 if self.resolution > 1000 else 1.0
                ti.init(arch=ti.gpu, device_memory_GB=device_memory_GB)
            except Exception as e:
                print(f"Error initializing Taichi: {e}")
                ti.init(arch=ti.cpu)

    def _initialize_simulator(self):
        """Initialize fluid simulator"""
        if self.no_dye:
            self.fluid_sim = FluidSimulator.create(
                self.n_bc,
                self.resolution,
                self.dt,
                self.dx,
                self.re,
                self.vor_eps,
                self.scheme,
                self.relaxation_factor,
                self.residual_threshold,
            )
        else:
            self.fluid_sim = DyeFluidSimulator.create(
                self.n_bc,
                self.resolution,
                self.dt,
                self.dx,
                self.re,
                self.vor_eps,
                self.scheme,
                self.relaxation_factor,
                self.residual_threshold,
            )

    def _initialize_output(self, dump_dir):
        """Initialize output paths"""
        self.output_path = (
            dump_dir
            + f"_bc{self.n_bc}_res{self.resolution}_re{self.re}_cfl{self.cfl}_relax{self.relaxation_factor}_residual{self.residual_threshold}_runtime{self.total_runtime}"
        )
        self.output_path = Path(self.output_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _print_config(self):
        """Print simulation configuration"""
        print(
            f"Boundary Condition: {self.n_bc}\n"
            f"CFL: {self.cfl}\n"
            f"dt: {self.dt}\n"
            f"Re: {self.re}\n"
            f"Resolution: {self.resolution}\n"
            f"Scheme: {self.scheme}\n"
            f"Vorticity confinement: {self.vor_eps}\n"
            f"Relaxation factor: {self.relaxation_factor}\n"
            f"Residual threshold: {self.residual_threshold}"
        )

    def save_simulation_data(self, frame_number):
        """Save simulation data to H5 file for a single frame"""

        # Get solver for direct field access
        solver = self.fluid_sim._solver

        # Get raw field data
        v_field = solver.v.current.to_numpy()
        p_field = solver.p.current.to_numpy()

        # Prepare field data
        vx = v_field[:, :, 0]  # x-velocity
        vy = v_field[:, :, 1]  # y-velocity
        pressure = p_field[:, :, 0] if p_field.ndim == 3 else p_field

        # Calculate vorticity (curl of velocity)
        dx = solver.dx
        vorticity = np.zeros_like(pressure)
        for i in range(1, v_field.shape[0] - 1):
            for j in range(1, v_field.shape[1] - 1):
                # ∂v_y/∂x - ∂v_x/∂y
                dvydx = (v_field[i + 1, j, 1] - v_field[i - 1, j, 1]) / (2 * dx)
                dvxdy = (v_field[i, j + 1, 0] - v_field[i, j - 1, 0]) / (2 * dx)
                vorticity[i, j] = dvydx - dvxdy

        # Get dye data if available
        dye = None
        if not self.no_dye and hasattr(solver, "dye"):
            dye = solver.dye.current.to_numpy()

        # Create filename for this frame
        data_path = self.output_path / "data"
        data_path.mkdir(exist_ok=True)
        h5_filename = data_path / f"res_{frame_number:06d}.h5"

        # Save to H5 file
        with h5py.File(h5_filename, "w") as f:
            # Save field data
            f.create_dataset("vx", data=vx, compression="gzip", compression_opts=9)
            f.create_dataset("vy", data=vy, compression="gzip", compression_opts=9)
            f.create_dataset("pressure", data=pressure, compression="gzip", compression_opts=9)
            f.create_dataset("vorticity", data=vorticity, compression="gzip", compression_opts=9)

            # Save dye if available
            if dye is not None:
                f.create_dataset("dye", data=dye, compression="gzip", compression_opts=9)

            # Save metadata
            f.attrs["time"] = self.current_time
            f.attrs["step"] = self.num_steps
            f.attrs["dt"] = self.dt
            f.attrs["cfl"] = self.cfl
            f.attrs["resolution"] = solver._resolution
            f.attrs["dx"] = solver.dx
            f.attrs["re"] = solver.Re

    def pre_process(self):
        """Initialize simulation before running"""
        self.start_time = time.time()
        self.converged = True
        print(f"Maximum wall time limit: {self.max_runtime}s ({self.max_runtime/60:.1f} minutes)")

    def cal_dt(self):
        """Calculate and return the base timestep"""
        return self.dt

    def post_process(self):
        """Post-process simulation results and save metadata"""
        import json

        # Calculate cost: 2 * resolution^2 * (total_steps + sum_iter_pressure_solver)
        self.total_pressure_iterations = self.fluid_sim.get_total_pressure_iterations()
        cost = 2 * (self.resolution**2) * (self.num_steps + self.total_pressure_iterations)

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time if self.start_time else 0.0

        # Prepare metadata
        meta = {
            "cost": cost,
            "num_steps": self.num_steps,
            "total_pressure_iterations": self.total_pressure_iterations,
            "total_runtime": self.current_time,
            "wall_time": elapsed_time,
            "converged": self.converged,
            "parameters": {
                "boundary_condition": self.n_bc,
                "resolution": self.resolution,
                "reynolds_num": self.re,
                "cfl": self.cfl,
                "time_step": self.dt,
                "advection_scheme": self.scheme,
                "vorticity_confinement": self.vor_eps,
                "relaxation_factor": self.relaxation_factor,
                "residual_threshold": self.residual_threshold,
                "no_dye": self.no_dye,
                "cpu": self.cpu,
            },
        }

        # Save metadata
        meta_file = self.output_path / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=4)

        print(f"Post-processing completed:")
        print(f"  Cost: {cost}")
        print(f"  Total steps: {self.num_steps}")
        print(f"  Total pressure iterations: {self.total_pressure_iterations}")
        print(f"  Total runtime: {self.current_time:.3f}s")
        print(f"  Wall time: {elapsed_time:.1f}s")
        print(f"  Converged: {self.converged}")
        print(f"  Metadata saved to: {meta_file}")

        # Print info about saved data files
        print(f"\nTemporal simulation data saved to: {self.output_path}/data/")
        print(f"Total time steps saved: {self.record_frame}")
        print(f"Each frame saved as: res_XXXXXX.h5 with all fields (vx, vy, pressure, vorticity, dye)")

    def dump(self):
        """Save simulation state at current_time"""
        # Save H5 data file for this frame
        self.save_simulation_data(self.record_frame)

        # Generate and save visualization image
        if self.vis_num == 0:
            img = self.fluid_sim.get_norm_field()
        elif self.vis_num == 1:
            img = self.fluid_sim.get_pressure_field()
        elif self.vis_num == 2:
            img = self.fluid_sim.get_vorticity_field()
        elif self.vis_num == 3:
            img = self.fluid_sim.get_dye_field()
        else:
            raise NotImplementedError()

        self.output_path.mkdir(exist_ok=True)
        ti.tools.imwrite(img, str(self.output_path / f"step_{self.num_steps:06}.png"))

    def step(self, dt):
        """Perform a simulation step"""
        self.fluid_sim.step()

    def early_stop(self):
        """Check if the simulation should stop early"""
        if self.start_time is None:
            return False

        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_runtime:
            print(f"\nSimulation timeout after {elapsed_time:.1f}s ({self.max_runtime}s limit)")
            self.converged = False
            return True

        return False
