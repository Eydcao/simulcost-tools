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
        self.output_path = (dump_dir + f"_bc{self.n_bc}_res{self.resolution}_re{self.re}_cfl{self.cfl}_scheme{self.scheme}_vor{self.vor_eps}_relax{self.relaxation_factor}_residual{self.residual_threshold}_runtime{self.total_runtime}_no_dye{self.no_dye}_cpu{self.cpu}_vis{self.vis_num}")
        self.output_path = Path(self.output_path)
        self.output_dir = f"{self.output_path}/videos"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
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
        """Run the fluid simulation"""
        video_manager = ti.tools.VideoManager(output_dir=self.output_dir, framerate=30, automatic_build=False)
        
        step = 0
        h5_file = None  # H5 file for accumulating data
        converged = True  # Track convergence status
        
        # Start timer for total runtime tracking
        start_time = time.time()
        max_runtime = 1200  # 20 minutes in seconds
        
        # Calculate total steps if runtime is specified
        total_steps = None
        if self.total_runtime is not None:
            total_steps = int(self.total_runtime / self.dt)
            print(f"Running for {total_steps} steps (runtime: {self.total_runtime:.3f}s, dt: {self.dt:.6f})")
        
        print(f"Maximum wall time limit: {max_runtime}s ({max_runtime/60:.1f} minutes)")
        
        # Main simulation loop
        while (total_steps is None or step < total_steps):
            # Check if we've exceeded the maximum runtime
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime:
                print(f"\nSimulation timeout after {elapsed_time:.1f}s ({max_runtime}s limit)")
                converged = False
                break
            
            # Generate visualization frames every 5 steps
            if step % 5 == 0:
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

                # Save frame to video
                video_manager.write_frame(img)
                
                # Save screenshot every 100 steps
                if step % 100 == 0:
                    self.output_path.mkdir(exist_ok=True)
                    ti.tools.imwrite(img, str(self.output_path / f"step_{step:06}.png"))
                
            # Save data every 0.01 time units
            if step * self.dt % 0.01 == 0:
                h5_file = self.save_simulation_data(step, h5_file, save_all_fields=False)

            # Advance simulation
            self.fluid_sim.step()
            
            step += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step}/{total_steps if total_steps else '∞'}, Time: {step * self.dt:.3f}s")
        
        # Print completion message
        elapsed_time = time.time() - start_time
        if total_steps is not None and step >= total_steps:
            print(f"\nSimulation completed after {step} steps (runtime: {step * self.dt:.3f}s, wall time: {elapsed_time:.1f}s)")
        elif not converged:
            print(f"\nSimulation timed out after {step} steps (runtime: {step * self.dt:.3f}s, wall time: {elapsed_time:.1f}s)")
        else:
            print(f"\nSimulation completed after {step} steps (runtime: {step * self.dt:.3f}s, wall time: {elapsed_time:.1f}s)")

        # Save final state with all fields
        if h5_file is not None:
            # Save final step with all fields
            self.save_simulation_data(step, h5_file, save_all_fields=True)
            h5_file.close()
            print(f"\nTemporal simulation data saved to: {self.output_path}/data/simulation_data.h5")
            print(f"Total time steps saved: {step // 5 + 1}")
            print(f"Final step includes all fields (vx, vy, pressure, vorticity, dye)")

        # Generate video with error handling
        try:
            video_manager.make_video(mp4=True)
        except Exception as e:
            print(f"Warning: Video generation failed: {e}")
            print("Continuing with post-processing...")
        
        # Post-process simulation results (always executed)
        is_converged = self.post_process(step, converged, elapsed_time)
        return is_converged
    
    def post_process(self, final_step, converged=True, elapsed_time=0.0):
        """Post-process simulation results and save metadata"""
        import json
        
        # Calculate cost: 2 * resolution^2 * total_steps * avg_pressure_iterations
        # This accounts for velocity and pressure updates per step, with variable pressure solver iterations
        # Note: Since we now use residual checking, the actual number of iterations varies
        cost = 2 * (self.resolution ** 2) * final_step
        
        # Prepare metadata
        meta = {
            "cost": cost,
            "num_steps": final_step,
            "total_runtime": final_step * self.dt,
            "wall_time": elapsed_time,
            "converged": converged,
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
        print(f"  Total steps: {final_step}")
        print(f"  Total runtime: {final_step * self.dt:.3f}s")
        print(f"  Wall time: {elapsed_time:.1f}s")
        print(f"  Converged: {converged}")
        print(f"  Metadata saved to: {meta_file}")
        return converged