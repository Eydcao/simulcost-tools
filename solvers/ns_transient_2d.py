import time
from pathlib import Path

import numpy as np
import taichi as ti
import h5py

from .ns_transient_2d_utils.fluid_simulator import DyeFluidSimulator, FluidSimulator
import os

ASPECT_RATIO = 0.5 # y/x
MAXIMUM_WALL_TIME = 1200 # 20 minutes in seconds

class NSTransient2D:
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
        self.record_dt = 0.1  # Time interval between recordings
        self.next_record_time = 0.0
        self.end_time = self.total_runtime if self.total_runtime is not None else 1.0
        self.record_frame = 0
        self.num_steps = 0
        
        # Timeout tracking
        self.start_time = None
        self.max_runtime = MAXIMUM_WALL_TIME
        
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
                self.n_bc, self.resolution, self.dt, self.dx, self.re, 
                self.vor_eps, self.scheme, self.relaxation_factor, self.residual_threshold
            )
        else:
            self.fluid_sim = DyeFluidSimulator.create(
                self.n_bc, self.resolution, self.dt, self.dx, self.re, 
                self.vor_eps, self.scheme, self.relaxation_factor, self.residual_threshold
            )
    
    def _initialize_output(self, dump_dir):
        """Initialize output paths"""
        self.output_path = (dump_dir + f"_bc{self.n_bc}_res{self.resolution}_re{self.re}_cfl{self.cfl}_relax{self.relaxation_factor}_residual{self.residual_threshold}_runtime{self.total_runtime}")
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
    
    def save_simulation_data(self, step, h5_file=None, save_all_fields=False):
        """Save simulation data to H5 file"""
        
        # Get solver for direct field access
        solver = self.fluid_sim._solver
        
        # Get raw field data
        v_field = solver.v.current.to_numpy()
        p_field = solver.p.current.to_numpy()
        
        # Prepare field data
        vx = v_field[:, :, 0]  # x-velocity
        vy = v_field[:, :, 1]  # y-velocity
        pressure = p_field[:, :, 0] if p_field.ndim == 3 else p_field
        
        # Calculate additional fields only if needed
        vorticity = None
        dye = None
        if save_all_fields:
            # Calculate vorticity (curl of velocity)
            dx = solver.dx
            vorticity = np.zeros_like(pressure)
            for i in range(1, v_field.shape[0]-1):
                for j in range(1, v_field.shape[1]-1):
                    # ∂v_y/∂x - ∂v_x/∂y
                    dvydx = (v_field[i+1, j, 1] - v_field[i-1, j, 1]) / (2 * dx)
                    dvxdy = (v_field[i, j+1, 0] - v_field[i, j-1, 0]) / (2 * dx)
                    vorticity[i, j] = dvydx - dvxdy
            
            # Get dye data if available
            if not self.no_dye and hasattr(solver, 'dye'):
                dye = solver.dye.current.to_numpy()
        
        # Save to H5 file
        if h5_file is None:
            # Create new H5 file
            data_path = self.output_path / "data"
            data_path.mkdir(exist_ok=True)
            h5_filename = data_path / f"simulation_data.h5"
            h5_file = h5py.File(h5_filename, 'w')
            
            # Save metadata
            metadata = h5_file.create_group('metadata')
            metadata.attrs['dt'] = self.dt
            metadata.attrs['cfl'] = self.cfl
            metadata.attrs['resolution'] = solver._resolution
            metadata.attrs['dx'] = solver.dx
            metadata.attrs['re'] = solver.Re
            
            # Create datasets for temporal data
            fields = h5_file.create_group('fields')
            fields.create_dataset('steps', data=[step], maxshape=(None,), dtype=int)
            fields.create_dataset('times', data=[step * self.dt], maxshape=(None,), dtype=float)
            fields.create_dataset('vx', data=[vx], maxshape=(None, vx.shape[0], vx.shape[1]), compression='gzip', compression_opts=9)
            fields.create_dataset('vy', data=[vy], maxshape=(None, vy.shape[0], vy.shape[1]), compression='gzip', compression_opts=9)
            fields.create_dataset('pressure', data=[pressure], maxshape=(None, pressure.shape[0], pressure.shape[1]), compression='gzip', compression_opts=9)
            
            # Add additional fields if saving all
            if save_all_fields:
                fields.create_dataset('vorticity', data=[vorticity], maxshape=(None, vorticity.shape[0], vorticity.shape[1]), compression='gzip', compression_opts=9)
                if dye is not None:
                    fields.create_dataset('dye', data=[dye], maxshape=(None, dye.shape[0], dye.shape[1], dye.shape[2]), compression='gzip', compression_opts=9)
            
            print(f"Created H5 file: {h5_filename}")
            return h5_file
        else:
            # Append to existing H5 file
            fields = h5_file['fields']
            
            # Resize datasets
            current_size = fields['steps'].shape[0]
            new_size = current_size + 1
            
            fields['steps'].resize((new_size,))
            fields['times'].resize((new_size,))
            fields['vx'].resize((new_size, vx.shape[0], vx.shape[1]))
            fields['vy'].resize((new_size, vy.shape[0], vy.shape[1]))
            fields['pressure'].resize((new_size, pressure.shape[0], pressure.shape[1]))
            
            # Add new data
            fields['steps'][current_size] = step
            fields['times'][current_size] = step * self.dt
            fields['vx'][current_size] = vx
            fields['vy'][current_size] = vy
            fields['pressure'][current_size] = pressure
            
            # Add additional fields if saving all
            if save_all_fields:
                if 'vorticity' in fields:
                    fields['vorticity'].resize((new_size, vorticity.shape[0], vorticity.shape[1]))
                    fields['vorticity'][current_size] = vorticity
                else:
                    fields.create_dataset('vorticity', data=[vorticity], maxshape=(None, vorticity.shape[0], vorticity.shape[1]), compression='gzip', compression_opts=9)
                
                if dye is not None:
                    if 'dye' in fields:
                        fields['dye'].resize((new_size, dye.shape[0], dye.shape[1], dye.shape[2]))
                        fields['dye'][current_size] = dye
                    else:
                        fields.create_dataset('dye', data=[dye], maxshape=(None, dye.shape[0], dye.shape[1], dye.shape[2]), compression='gzip', compression_opts=9)
            
            return h5_file
    
    def run(self):
        """Run the fluid simulation following base solver format"""
        self.pre_process()

        # Initial recording at time 0
        self.dump()
        self.call_back()
        self.record_frame += 1
        self.next_record_time = min(self.current_time + self.record_dt, self.end_time)

        while self.current_time < self.end_time - 1e-10:
            # Calculate base timestep
            base_dt = self.cal_dt()

            # Adjust timestep for recording if needed
            dt, should_record = self.adjust_dt_for_recording(base_dt)

            if self.verbose:
                print(f"Time: {self.current_time:.6f}, dt: {dt:.6e}")

            # Perform simulation step
            self.step(dt)
            self.current_time += dt
            self.num_steps += 1

            # Handle recording if we've reached a recording time
            if should_record:
                self.dump()
                self.call_back()
                self.record_frame += 1
                self.next_record_time = min(self.current_time + self.record_dt, self.end_time)

            if self.early_stop():
                if self.verbose:
                    print(f"Early stopping at time {self.current_time:.6f}")
                break

        if self.verbose:
            print(f"Simulation completed at time {self.current_time:.6f}, total steps: {self.num_steps}")
        self.post_process()
        return self.converged
    
    def adjust_dt_for_recording(self, dt):
        """
        Adjust timestep to align with recording times.
        Returns adjusted timestep and whether we should record after this step.
        """
        time_remaining = self.next_record_time - self.current_time

        # If we've passed the recording time (shouldn't normally happen)
        if time_remaining <= 1e-10:  # Small tolerance for floating point comparison
            if self.verbose:
                print(
                    f"Warning: Current time {self.current_time:.6f} has passed the next recording time {self.next_record_time:.6f}."
                )
            return time_remaining, True

        # Calculate how many base timesteps remain until recording
        steps_to_record = time_remaining / dt

        if steps_to_record >= 2:
            # No adjustment needed
            return dt, False
        elif steps_to_record > 1:
            # Halve the timestep to get closer to recording time
            return dt / 2, False
        else:
            # Adjust timestep to exactly reach recording time
            return time_remaining, True
    
    def pre_process(self):
        """Initialize simulation before running"""
        self.start_time = time.time()
        self.h5_file = None
        self.converged = True
        print(f"Maximum wall time limit: {self.max_runtime}s ({self.max_runtime/60:.1f} minutes)")
    
    def cal_dt(self):
        """Calculate and return the base timestep"""
        return self.dt
    
    def call_back(self):
        """Called after each recording"""
        if self.verbose:
            print(f"Recording at time {self.current_time:.6f} (frame {self.record_frame})")
            print(f"Number of steps: {self.num_steps}")
    
    def post_process(self):
        """Post-process simulation results and save metadata"""
        import json
        
        # Calculate cost: 2 * resolution^2 * (total_steps + sum_iter_pressure_solver)
        self.total_pressure_iterations = self.fluid_sim.get_total_pressure_iterations()
        cost = 2 * (self.resolution ** 2) * (self.num_steps + self.total_pressure_iterations)
        
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
                "cpu": self.cpu
            }
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
        
        # Close H5 file if it exists
        if self.h5_file is not None:
            self.h5_file.close()
            print(f"\nTemporal simulation data saved to: {self.output_path}/data/simulation_data.h5")
            print(f"Total time steps saved: {self.record_frame}")
            print(f"Final step includes all fields (vx, vy, pressure, vorticity, dye)")
    
    def dump(self):
        """Save simulation state at current_time"""
        # Save data with all fields for final step, basic fields for intermediate steps
        save_all_fields = (self.current_time >= self.end_time - 1e-10)
        self.h5_file = self.save_simulation_data(self.num_steps, self.h5_file, save_all_fields=save_all_fields)
        
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