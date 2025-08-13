import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import os
import json
import h5py
import warnings

OUTER_ITERATIONS = 10000
EPS = 1e-12
ITER_V = 50

class NSChannel2D():
    def __init__(self,
                 # -------- Constants --------
                 outer_iterations=OUTER_ITERATIONS,
                 eps=EPS,
                 iter_v=ITER_V,
                 # -------- Environment Dependents --------
                 cfg = None,
                 verbose = False
                 ): 
        # Mesh and fluid properties
        self.length = cfg.length if "length" in cfg else 20.0
        self.breadth = cfg.breadth if "breadth" in cfg else 1.0
        self.mesh_x = cfg.mesh_x
        self.mesh_y = cfg.mesh_y
        self.dx = self.length / self.mesh_x
        self.dy = self.breadth / self.mesh_y
        self.mu = cfg.mu if "mu" in cfg else 0.01
        self.rho = cfg.rho if "rho" in cfg else 1.0
        # Iteration parameters
        self.omega_u = cfg.omega_u
        self.omega_v = cfg.omega_v
        self.omega_p = cfg.omega_p
        self.outer_iterations = outer_iterations
        self.iter_v = iter_v
        self.res_iter_v_threshold = lambda k: max(1e-4, 1e-2 * (0.1 ** (k / 500))) if cfg.res_iter_v_threshold == "exp_decay" else cfg.res_iter_v_threshold
        self.res_iter_v_threshold_name = "exp_decay" if cfg.res_iter_v_threshold == "exp_decay" else cfg.res_iter_v_threshold
        self.eps = eps
        self.diff_u_threshold = cfg.diff_u_threshold
        self.diff_v_threshold = cfg.diff_v_threshold
        self.mass_conservation_threshold = cfg.mass_conservation_threshold if "mass_conservation_threshold" in cfg else 1e-8
        self.boundary_condition = cfg.boundary_condition
        self.other_params = cfg.other_params if "other_params" in cfg else {}
        # Coordinates
        self.X = np.linspace(0, self.length, self.mesh_x + 1)
        self.Y = np.linspace(0, self.breadth, self.mesh_y + 1)
        # Residual tracking
        self.u_residual = []
        self.v_residual = []
        self.tot = []
        # Fields
        self.reset_fields()
        self.dump_dir = (
            cfg.dump_dir + f"_{self.boundary_condition}_mesh_{self.mesh_x}_{self.mesh_y}_relax_{self.omega_u}_{self.omega_v}_{self.omega_p}_error_{self.diff_u_threshold}_{self.diff_v_threshold}_itererror_{self.res_iter_v_threshold_name}"
        )
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)
        self.verbose = verbose
        self.num_steps = None
        self.converged = None

    def reset_fields(self):
        my, mx = self.mesh_y, self.mesh_x
        self.u = np.zeros((my + 2, mx + 1))
        self.v = np.zeros((my + 1, mx + 2))
        self.u_old = np.zeros_like(self.u)
        self.v_old = np.zeros_like(self.v)
        self.pressure = np.zeros((my + 2, mx + 2))
        self.p_prime = np.zeros_like(self.pressure)
        self.u_diag_coeff = np.ones_like(self.pressure)
        self.v_diag_coeff = np.ones_like(self.pressure)
        self.pressure_diag_coeff = np.ones_like(self.pressure)
        self.east_coeff = np.zeros_like(self.pressure)
        self.west_coeff = np.zeros_like(self.pressure)
        self.north_coeff = np.zeros_like(self.pressure)
        self.south_coeff = np.zeros_like(self.pressure)
        self.source = np.zeros_like(self.pressure)
        self.U = np.zeros((my + 1, mx + 1))
        self.V = np.zeros((my + 1, mx + 1))
        self.P = np.zeros((my + 1, mx + 1))

    def _build_pressure_matrix(self, mesh_y, mesh_x, east_coeff, west_coeff, north_coeff, south_coeff, pressure_diag_coeff):
        N = mesh_y * mesh_x
        A = lil_matrix((N, N))
        def idx(i, j):
            return (i - 1) * mesh_x + (j - 1)
        for i in range(1, mesh_y + 1):
            for j in range(1, mesh_x + 1):
                row = idx(i, j)
                A[row, row] = pressure_diag_coeff[i, j]
                if j < mesh_x:
                    A[row, idx(i, j + 1)] = -east_coeff[i, j]
                if j > 1:
                    A[row, idx(i, j - 1)] = -west_coeff[i, j]
                if i < mesh_y:
                    A[row, idx(i + 1, j)] = -north_coeff[i, j]
                if i > 1:
                    A[row, idx(i - 1, j)] = -south_coeff[i, j]
        return A.tocsr()

    def _solve_pressure_correction(self, u, v, east_coeff, west_coeff, north_coeff, south_coeff, pressure_diag_coeff):
        mesh_y, mesh_x = u.shape[0] - 2, u.shape[1] - 1
        source = np.zeros((mesh_y + 2, mesh_x + 2))
        p_prime = np.zeros_like(source)
        source[1:mesh_y+1, 1:mesh_x+1] = self.rho * self.dy * (u[1:mesh_y+1, 1:mesh_x+1] - u[1:mesh_y+1, 0:mesh_x]) + self.rho * self.dx * (v[1:mesh_y+1, 1:mesh_x+1] - v[0:mesh_y, 1:mesh_x+1])
        b = -source[1:mesh_y+1, 1:mesh_x+1].reshape(-1)
        A = self._build_pressure_matrix(mesh_y, mesh_x, east_coeff, west_coeff, north_coeff, south_coeff, pressure_diag_coeff)
        p_solution = spsolve(A, b)
        p_prime[1:mesh_y+1, 1:mesh_x+1] = p_solution.reshape(mesh_y, mesh_x)
        return p_prime

    @staticmethod
    def compute_diff(new, old):
        return np.linalg.norm(new - old) / (np.linalg.norm(old) + 1e-12)

    def apply_boundary_conditions(self):
        self.node_type = np.full((self.mesh_y + 2, self.mesh_x + 2), "internal", dtype=object)
        my, mx = self.mesh_y, self.mesh_x
        if self.boundary_condition == "channel_flow":
            # Inlet (left boundary)
            self.node_type[:, 0] = "inlet"
            # Outlet (right boundary)
            self.node_type[:, -1] = "outlet"
            # Top wall
            self.node_type[-1, :] = "top-wall"
            # Bottom wall
            self.node_type[0, :] = "bottom-wall"
            
            # Apply boundary conditions for u
            for i in range(my + 2):
                for j in range(mx + 1):
                    # u[i, j] is located between pressure[i, j] and pressure[i, j+1]
                    left_type = self.node_type[i, j]
                    right_type = self.node_type[i, j + 1]
                    if left_type == "inlet":
                        if i >= 1 and i < self.u.shape[0]-1:
                            y = (i - 0.5) * self.dy
                            self.u[i, 0] = 4 * 1.0 * y * (self.breadth - y) / (self.breadth ** 2)
                    elif right_type == "outlet":
                        self.u[i, j] = 0.0
                    elif left_type == "bottom-wall" or right_type == "top-wall":
                        self.u[i, j] = 0.0
            
            # Apply boundary conditions for v
            for i in range(my + 1):
                for j in range(mx + 2):
                    # v[i, j] is located between pressure[i, j] and pressure[i+1, j]
                    bottom_type = self.node_type[i, j]
                    top_type = self.node_type[i + 1, j]
                    if bottom_type == "bottom-wall" or top_type == "top-wall":
                        self.v[i, j] = 0.0
                    elif bottom_type == "inlet" or top_type == "inlet":
                        self.v[i, j] = 0.0
                    elif bottom_type == "outlet" or top_type == "outlet":
                        self.v[i, j] = self.v[i - 1, j] if i > 0 else 0.0
            
            # Apply boundary conditions for pressure
            for i in range(my + 2):
                for j in range(mx + 2):
                    if self.node_type[i, j] == "outlet":
                        self.pressure[i, j] = 0.0
        
        elif self.boundary_condition == "back_stair_flow":
            wall_height = self.other_params.get("wall_height", 20)
            wall_width = self.other_params.get("wall_width", 50)
            # Inlet (left boundary)
            self.node_type[wall_height:, 0] = "inlet"
            # Outlet (right boundary)
            self.node_type[:, -1] = "outlet"
            # Top wall
            self.node_type[-1, :] = "top-wall"
            # Bottom wall
            self.node_type[0, :] = "bottom-wall"
            self.node_type[:wall_height, :wall_width] = "bottom-wall"
            
            # Apply boundary conditions for u
            for i in range(my + 2):
                for j in range(mx + 1):
                    # u[i, j] is located between pressure[i, j] and pressure[i, j+1]
                    left_type = self.node_type[i, j]
                    right_type = self.node_type[i, j + 1]
                    if left_type == "inlet":
                        if i >= 1 and i < self.u.shape[0]-1:
                            # Parabolic profile centered on the height of range 20: only
                            y = (i - wall_height + 0.5) * self.dy
                            height = (my + 2 - wall_height) * self.dy
                            self.u[i, 0] = 4 * 1.0 * y * (height - y) / (height ** 2)
                    elif right_type == "outlet":
                        self.u[i, j] = 0.0
                    elif left_type == "bottom-wall" or right_type == "top-wall":
                        self.u[i, j] = 0.0
            
             # Apply boundary conditions for pressure
            for i in range(my + 2):
                for j in range(mx + 2):
                    if self.node_type[i, j] == "outlet":
                        self.pressure[i, j] = 0.0
            
        elif self.boundary_condition == "expansion_channel":
            wall_height = self.other_params.get("wall_height", 15)
            wall_width = self.other_params.get("wall_width", 50)
            print(f"Applying expansion channel boundary conditions with wall_height={wall_height}, wall_width={wall_width}")
            # Inlet (left boundary)
            self.node_type[wall_height:-wall_height, 0] = "inlet"
            # Outlet (right boundary)
            self.node_type[:, -1] = "outlet"
            # Top wall
            self.node_type[-1, :] = "top-wall"
            self.node_type[-wall_height:, :wall_width] = "top-wall"
            # Bottom wall
            self.node_type[0, :] = "bottom-wall"
            self.node_type[:wall_height, :wall_width] = "bottom-wall"
            
            # Apply boundary conditions for u
            for i in range(my + 2):
                for j in range(mx + 1):
                    # u[i, j] is located between pressure[i, j] and pressure[i, j+1]
                    left_type = self.node_type[i, j]
                    right_type = self.node_type[i, j + 1]
                    if left_type == "inlet":
                        if i >= 1 and i < self.u.shape[0]-1:
                            # Parabolic profile centered in the range 15:-15
                            y = (i - wall_height + 0.5) * self.dy
                            height = (my + 2 - wall_height*2) * self.dy
                            self.u[i, 0] = 4 * 1.0 * y * (height - y) / (height ** 2)
                    elif right_type == "outlet":
                        self.u[i, j] = 0.0
                    elif left_type == "bottom-wall" or left_type == "top-wall":
                        self.u[i, j] = 0.0
            
             # Apply boundary conditions for pressure
            for i in range(my + 2):
                for j in range(mx + 2):
                    if self.node_type[i, j] == "outlet":
                        self.pressure[i, j] = 0.0
        
        elif self.boundary_condition == "cube_driven_flow":
            wall_height = self.other_params.get("wall_height", 10)
            wall_width = self.other_params.get("wall_width", 10)
            wall_start_height = self.other_params.get("wall_start_height", 20)
            wall_start_width = self.other_params.get("wall_start_width", 80)
            # Inlet (left boundary)
            self.node_type[:, 0] = "inlet"
            # Outlet (right boundary)
            self.node_type[:, -1] = "outlet"
            # Top wall
            self.node_type[-1, :] = "top-wall"
            # Bottom wall
            self.node_type[0, :] = "bottom-wall"
            
            # cube block inside
            self.node_type[wall_start_height:wall_start_height + wall_height, wall_start_width:wall_start_width + wall_width] = "bottom-wall"
            
            # Apply boundary conditions for u
            for i in range(my + 2):
                for j in range(mx + 1):
                    # u[i, j] is located between pressure[i, j] and pressure[i, j+1]
                    left_type = self.node_type[i, j]
                    right_type = self.node_type[i, j + 1]
                    if left_type == "inlet":
                        if i >= 1 and i < self.u.shape[0]-1:
                            y = (i - 0.5) * self.dy
                            self.u[i, 0] = 4 * 1.0 * y * (self.breadth - y) / (self.breadth ** 2)
                    elif right_type == "outlet":
                        self.u[i, j] = 0.0
                    elif left_type == "bottom-wall" or right_type == "top-wall":
                        self.u[i, j] = 0.0
            
            # Apply boundary conditions for pressure
            for i in range(my + 2):
                for j in range(mx + 2):
                    if self.node_type[i, j] == "outlet":
                        self.pressure[i, j] = 0.0


    def _dump_step(self, k):
        my, mx = self.mesh_y, self.mesh_x
        u, v, pressure = self.u, self.v, self.pressure
        U, V, P = self.U, self.V, self.P
        U[1:my, 1:mx] = 0.5 * (u[2:my+1, 1:mx] + u[1:my, 1:mx])
        V[1:my, 1:mx] = 0.5 * (v[1:my, 2:mx+1] + v[1:my, 1:mx])
        P[1:my, 1:mx] = 0.25 * (pressure[1:my, 1:mx] + pressure[2:my+1, 1:mx] + pressure[1:my, 2:mx+1] + pressure[2:my+1, 2:mx+1])
        U[self.node_type[:-1, :-1] == "top-wall"] = 0
        U[self.node_type[:-1, :-1] == "bottom-wall"] = 0
        V[self.node_type[:-1, :-1] == "top-wall"] = 0
        V[self.node_type[:-1, :-1] == "bottom-wall"] = 0
        P[self.node_type[:-1, :-1] == "top-wall"] = 0
        P[self.node_type[:-1, :-1] == "bottom-wall"] = 0
        
        file_base = os.dump_dir.join(self.dump_dir, f"res_{k}")

        # Save HDF5 data file
        with h5py.File(f"{file_base}.h5", "w") as f:
            f.create_dataset("u", data=U)
            f.create_dataset("v", data=V)
            f.create_dataset("p", data=P)
            
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        c1 = axs[0].imshow(U[1:-1, 1:-1], cmap='jet', origin='lower', aspect='auto')
        axs[0].set_title("X-Velocity (U)")
        fig.colorbar(c1, ax=axs[0])
        c2 = axs[1].imshow(V[1:-1, 1:-1], cmap='jet', origin='lower', aspect='auto')
        axs[1].set_title("Y-Velocity (V)")
        fig.colorbar(c2, ax=axs[1])
        c3 = axs[2].imshow(P[1:-1, 1:-1], cmap='jet', origin='lower', aspect='auto')
        axs[2].set_title("Pressure (P)")
        fig.colorbar(c3, ax=axs[2])
        plt.tight_layout()
        plt.savefig(os.path.join(self.dump_dir, f"visualizations_{k}.png"))
        plt.close()

    def _dump_final(self):
        my, mx = self.mesh_y, self.mesh_x
        u, v, pressure = self.u, self.v, self.pressure
        U, V, P = self.U, self.V, self.P
        # Center averages
        U[1:my, 1:mx] = 0.5 * (u[2:my+1, 1:mx] + u[1:my, 1:mx])
        V[1:my, 1:mx] = 0.5 * (v[1:my, 2:mx+1] + v[1:my, 1:mx])
        P[1:my, 1:mx] = 0.25 * (pressure[1:my, 1:mx] + pressure[2:my+1, 1:mx] + pressure[1:my, 2:mx+1] + pressure[2:my+1, 2:mx+1])
        # boundary excluding corners (as in original code)
        U[my, 1:mx] = u[my+1, 1:mx]
        U[0, 1:mx] = u[1, 1:mx]
        U[1:my, 0] = 0.5 * (u[1:my, 0] + u[2:my+1, 0])
        U[1:my, mx] = 0.5 * (u[1:my, mx] + u[2:my+1, mx])
        U[0, 0] = 0
        U[0, mx] = 0
        U[my, 0] = 0
        U[my, mx] = 0
        U[self.node_type[:-1, :-1] == "top-wall"] = 0
        U[self.node_type[:-1, :-1] == "bottom-wall"] = 0
        V[my, 1:mx] = 0.5 * (v[my, 1:mx] + v[my, 2:mx+1])
        V[0, 1:mx] = 0.5 * (v[0, 1:mx] + v[0, 2:mx+1])
        V[1:my, 0] = 0.5 * (v[1:my, 0] + v[2:my+1, 0])
        V[1:my, mx] = 0.5 * (v[1:my, mx] + v[2:my+1, mx])
        V[0, 0] = 0
        V[0, mx] = 0
        V[my, 0] = 0
        V[my, mx] = 0
        V[self.node_type[:-1, :-1] == "top-wall"] = 0
        V[self.node_type[:-1, :-1] == "bottom-wall"] = 0
        P[1:my, 0] = 0.5 * (pressure[1:my, 0] + pressure[2:my+1, 0])
        P[1:my, mx] = 0.5 * (pressure[1:my, mx] + pressure[2:my+1, mx])
        P[0, 1:mx] = 0.5 * (pressure[1, 1:mx] + pressure[1, 2:mx+1])
        P[my, 1:mx] = 0.5 * (pressure[my, 1:mx] + pressure[my, 2:mx+1])
        P[0, 0] = pressure[1, 1]
        P[0, mx] = pressure[1, mx]
        P[my, 0] = pressure[my, 1]
        P[my, mx] = pressure[my, mx]
        P[self.node_type[:-1, :-1] == "top-wall"] = 0
        P[self.node_type[:-1, :-1] == "bottom-wall"] = 0
        
        file_base = os.path.join(self.dump_dir, f"res_{self.num_steps}")

        # Save HDF5 data file
        with h5py.File(f"{file_base}.h5", "w") as f:
            f.create_dataset("u", data=U)
            f.create_dataset("v", data=V)
            f.create_dataset("p", data=P)
            
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        c1 = axs[0].imshow(U[1:-1, 1:-1], cmap='jet', origin='lower', aspect='auto')
        axs[0].set_title("X-Velocity (U)")
        fig.colorbar(c1, ax=axs[0])
        c2 = axs[1].imshow(V[1:-1, 1:-1], cmap='jet', origin='lower', aspect='auto')
        axs[1].set_title("Y-Velocity (V)")
        fig.colorbar(c2, ax=axs[1])
        c3 = axs[2].imshow(P[1:-1, 1:-1], cmap='jet', origin='lower', aspect='auto')
        axs[2].set_title("Pressure (P)")
        fig.colorbar(c3, ax=axs[2])
        plt.tight_layout()
        plt.savefig(os.path.join(self.dump_dir, f"visualizations_{self.num_steps}.png"))
        plt.close()


    def _plot_residuals(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.u_residual, label='U Residual', color='blue')
        plt.plot(self.v_residual, label='V Residual', color='orange')
        plt.plot(self.tot, label='Mass Conservation', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('Residuals Over Iterations')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.dump_dir, "residuals.png"))
        plt.close()
    
    def post_process(self):
        cost = (self.mesh_x * self.mesh_y) * self.num_steps
        meta = {"cost": cost, "num_steps": self.num_steps, "converged": int(self.converged)}  # ➜ Added
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(f"Run cost: {cost}, num_steps: {self.num_steps}")
        

    def run(self):
        self.apply_boundary_conditions()
        start_time = time.time()
        my, mx = self.mesh_y, self.mesh_x
        self.right_node_boundary = (self.node_type[:, 1:] == "outlet")
        self.left_node_boundary = (self.node_type[:, :-1] == "inlet")
        self.top_node_boundary = (self.node_type[1:, :] == "top-wall")
        self.bottom_node_boundary = (self.node_type[:-1, :] == "bottom-wall")
        for k in range(self.outer_iterations):
            self.u_old[:, :] = self.u
            self.v_old[:, :] = self.v
            # U-momentum coefficients
            i_range = slice(1, my + 1)
            j_range = slice(1, mx)
            self.east_coeff[i_range, j_range] = np.maximum(-0.5 * self.rho * self.dy * (self.u_old[i_range, j_range] + self.u_old[i_range, j_range.start+1:j_range.stop+1]), 0) + self.mu * self.dy / self.dx
            self.west_coeff[i_range, j_range] = np.maximum(0.5 * self.rho * self.dy * (self.u_old[i_range, j_range] + self.u_old[i_range, j_range.start-1:j_range.stop-1]), 0) + self.mu * self.dy / self.dx
            self.north_coeff[i_range, j_range] = np.maximum(-0.5 * self.rho * self.dx * (self.v_old[i_range, j_range] + self.v_old[i_range, j_range.start+1:j_range.stop+1]), 0) + self.mu * self.dx / self.dy
            self.south_coeff[i_range, j_range] = np.maximum(0.5 * self.rho * self.dx * (self.v_old[i_range.start-1:i_range.stop-1, j_range] + self.v_old[i_range.start-1:i_range.stop-1, j_range.start+1:j_range.stop+1]), 0) + self.mu * self.dx / self.dy
            self.north_coeff[my, 1:mx] = np.maximum(-0.5 * self.rho * self.dx * (self.v_old[my, 1:mx] + self.v_old[my, 2:mx+1]), 0) + self.mu * self.dx / (self.dy / 2)
            self.south_coeff[1, 1:mx] = np.maximum(0.5 * self.rho * self.dx * (self.v_old[0, 1:mx] + self.v_old[0, 2:mx+1]), 0) + self.mu * self.dx / (self.dy / 2)
            self.u_diag_coeff = self.east_coeff + self.west_coeff + self.north_coeff + self.south_coeff
            # self.u_diag_coeff[0:12, 19:21] = 1e30
            self.u_diag_coeff[self.node_type == "top-wall"] = 1e30
            self.u_diag_coeff[self.node_type == "bottom-wall"] = 1e30
            self.u_diag_coeff = np.maximum(self.u_diag_coeff, self.eps) / self.omega_u
            for _ in range(self.iter_v):
                try:
                    # Turn NumPy floating warnings into exceptions for this block
                    with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
                        self.u[i_range, j_range] = (1 - self.omega_u) * self.u_old[i_range, j_range] + (1 / self.u_diag_coeff[i_range, j_range]) * (
                            self.east_coeff[i_range, j_range] * self.u[i_range, j_range.start+1:j_range.stop+1] +
                            self.west_coeff[i_range, j_range] * self.u[i_range, j_range.start-1:j_range.stop-1] +
                            self.north_coeff[i_range, j_range] * self.u[i_range.start+1:i_range.stop+1, j_range] +
                            self.south_coeff[i_range, j_range] * self.u[i_range.start-1:i_range.stop-1, j_range] +
                            self.dy * (self.pressure[i_range, j_range] - self.pressure[i_range, j_range.start+1:j_range.stop+1])
                        )
                        self.u[:, -1] = self.u[:, -2]
                except FloatingPointError as e:
                    print(f"[Overflow caught] U-iteration failed at outer iter {k}: {e}")
                    # Minimal diagnostics to help debugging:
                    try:
                        ec = self.east_coeff[i_range, j_range]
                        block_u = self.u[i_range, j_range]
                        print(f"max|east_coeff|={np.nanmax(np.abs(ec)):.3e}, max|u|={np.nanmax(np.abs(block_u)):.3e}")
                    except Exception:
                        pass
                    # Stop early; caller can lower omega_u or check BCs.
                    return False
            # V-momentum coefficients
            i_range_v = slice(1, my)
            j_range_v = slice(1, mx + 1)
            self.east_coeff[i_range_v, j_range_v] = np.maximum(-0.5 * self.rho * self.dy * (self.u_old[i_range_v.start+1:i_range_v.stop+1, j_range_v] + self.u_old[i_range_v, j_range_v]), 0) + self.mu * self.dy / self.dx
            self.west_coeff[i_range_v, j_range_v] = np.maximum(0.5 * self.rho * self.dy * (self.u_old[i_range_v.start+1:i_range_v.stop+1, j_range_v.start-1:j_range_v.stop-1] + self.u_old[i_range_v, j_range_v.start-1:j_range_v.stop-1]), 0) + self.mu * self.dy / self.dx
            self.north_coeff[i_range_v, j_range_v] = np.maximum(-0.5 * self.rho * self.dx * (self.v_old[i_range_v, j_range_v] + self.v_old[i_range_v.start+1:i_range_v.stop+1, j_range_v]), 0) + self.mu * self.dx / self.dy
            self.south_coeff[i_range_v, j_range_v] = np.maximum(0.5 * self.rho * self.dx * (self.v_old[i_range_v, j_range_v] + self.v_old[i_range_v.start-1:i_range_v.stop-1, j_range_v]), 0) + self.mu * self.dx / self.dy
            self.east_coeff[i_range_v, -1] = np.maximum(-0.5 * self.rho * self.dy * (self.u_old[i_range_v.start+1:i_range_v.stop+1, -1] + self.u_old[i_range_v, -1]), 0) + self.mu * self.dy / (self.dx / 2)
            self.west_coeff[i_range_v, 1] = np.maximum(0.5 * self.rho * self.dy * (self.u_old[i_range_v.start+1:i_range_v.stop+1, 0] + self.u_old[i_range_v, 0]), 0) + self.mu * self.dy / (self.dx / 2)
            self.v_diag_coeff = self.east_coeff + self.west_coeff + self.north_coeff + self.south_coeff
            # self.v_diag_coeff[0:13, 20:21] = 1e30
            self.v_diag_coeff[self.node_type == "top-wall"] = 1e30
            self.v_diag_coeff[self.node_type == "bottom-wall"] = 1e30
            self.v_diag_coeff = np.maximum(self.v_diag_coeff, self.eps) / self.omega_v
            for _ in range(self.iter_v):
                try:
                    with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
                        v_new = self.v.copy()
                        self.v[i_range_v, j_range_v] = (1 - self.omega_v) * self.v_old[i_range_v, j_range_v] + (1 / self.v_diag_coeff[i_range_v, j_range_v]) * (
                            self.east_coeff[i_range_v, j_range_v] * self.v[i_range_v, j_range_v.start+1:j_range_v.stop+1] +
                            self.west_coeff[i_range_v, j_range_v] * self.v[i_range_v, j_range_v.start-1:j_range_v.stop-1] +
                            self.north_coeff[i_range_v, j_range_v] * self.v[i_range_v.start+1:i_range_v.stop+1, j_range_v] +
                            self.south_coeff[i_range_v, j_range_v] * self.v[i_range_v.start-1:i_range_v.stop-1, j_range_v] +
                            self.dx * (self.pressure[i_range_v, j_range_v] - self.pressure[i_range_v.start+1:i_range_v.stop+1, j_range_v])
                        )
                    res_v_inner = np.linalg.norm(self.v - v_new) / (np.linalg.norm(v_new) + 1e-12)
                    if res_v_inner < self.res_iter_v_threshold(k):
                        break
                except FloatingPointError as e:
                    print(f"[Overflow caught] V-iteration failed at outer iter {k}: {e}")
                    # Diagnostics for V-sweep
                    try:
                        ec = self.east_coeff[i_range_v, j_range_v]
                        block_v = self.v[i_range_v, j_range_v]
                        print(f"max|east_coeff|={np.nanmax(np.abs(ec)):.3e}, max|v|={np.nanmax(np.abs(block_v)):.3e}")
                    except Exception:
                        pass
                    return False
            # Pressure correction coefficients
            i_range_p = slice(1, my + 1)
            j_range_p = slice(1, mx + 1)
            self.east_coeff[i_range_p, j_range_p] = (self.rho * self.dy**2) / np.maximum(self.u_diag_coeff[i_range_p, j_range_p], self.eps)
            self.west_coeff[i_range_p, j_range_p] = (self.rho * self.dy**2) / np.maximum(self.u_diag_coeff[i_range_p, j_range_p.start-1:j_range_p.stop-1], self.eps)
            self.north_coeff[i_range_p, j_range_p] = (self.rho * self.dx**2) / np.maximum(self.v_diag_coeff[i_range_p, j_range_p], self.eps)
            self.south_coeff[i_range_p, j_range_p] = (self.rho * self.dx**2) / np.maximum(self.v_diag_coeff[i_range_p.start-1:i_range_p.stop-1, j_range_p], self.eps)
                        
            self.east_coeff[:, :-1][self.right_node_boundary] = 0
            self.west_coeff[:, 1:][self.left_node_boundary] = 0
            self.north_coeff[:-1, :][self.top_node_boundary] = 0
            self.south_coeff[1:, :][self.bottom_node_boundary] = 0

            self.pressure_diag_coeff = self.east_coeff + self.west_coeff + self.north_coeff + self.south_coeff
            # self.pressure_diag_coeff[1, 1] = 1e30
            self.pressure_diag_coeff[:, :-1][self.right_node_boundary] = 1e30
            # self.pressure_diag_coeff[self.node_type == "top-wall"] = 1e30
            # self.pressure_diag_coeff[self.node_type == "bottom-wall"] = 1e30
            self.p_prime = self._solve_pressure_correction(self.u, self.v, self.east_coeff, self.west_coeff, self.north_coeff, self.south_coeff, self.pressure_diag_coeff)
            self.pressure[1:-1, 1:-1] += self.omega_p * self.p_prime[1:-1, 1:-1]
            self.u[1:my+1, 1:mx] += (self.dy / self.u_diag_coeff[1:my+1, 1:mx]) * (self.p_prime[1:my+1, 1:mx] - self.p_prime[1:my+1, 2:mx+1])
            self.v[1:my, 1:mx+1] += (self.dx / self.v_diag_coeff[1:my, 1:mx+1]) * (self.p_prime[1:my, 1:mx+1] - self.p_prime[2:my+1, 1:mx+1])
            self.u[1:my+1, mx] = self.u[1:my+1, mx-1] + self.dx / self.dy * (self.v[0:my, mx] - self.v[1:my+1, mx-1])
            res_u = self.compute_diff(self.u, self.u_old)
            res_v = self.compute_diff(self.v, self.v_old)
            self.source[1:my+1, 1:mx+1] = self.rho * self.dy * (self.u[1:my+1, 1:mx+1] - self.u[1:my+1, 0:mx]) + self.rho * self.dx * (self.v[1:my+1, 1:mx+1] - self.v[0:my, 1:mx+1])
            total = np.sum(self.source**2)
            # in_mass_flow = np.sum(self.rho * self.dy * self.u[1:-1, 0])
            # out_mass_flow = np.sum(self.rho * self.dy * self.u[1:-1, -1])
            # total = self.compute_diff(in_mass_flow, out_mass_flow)
            if self.verbose:
                if k % 50 == 0 or k == self.outer_iterations - 1:
                    self.u_residual.append(res_u)
                    self.v_residual.append(res_v)
                    self.tot.append(total)
                    print(f"Iteration {k+1}: Residual U = {res_u:.2e}, V = {res_v:.2e}, Mass Conservation = {total:.2e}")
                    # self._dump_step(k)
            if res_u < self.diff_u_threshold and res_v < self.diff_v_threshold:
                if self.verbose:
                    print(f"Convergence achieved at iteration {k+1}: Residual U = {res_u:.2e}, V = {res_v:.2e}, Mass Conservation = {total:.2e}")
                self.num_steps = k + 1
                self.converged = True
                break
        if k == self.outer_iterations - 1:
            if self.verbose:
                print(f"Reached maximum iterations ({self.outer_iterations}) without convergence. Final residuals: U = {res_u:.2e}, V = {res_v:.2e}, Mass Conservation = {total:.2e}")
            self.num_steps = self.outer_iterations
            self.converged = False
        if self.verbose:
            if total > self.mass_conservation_threshold:
                print(f"Warning: Mass conservation threshold exceeded: {total:.2e} > {self.mass_conservation_threshold:.2e}")
            else:
                print(f"Mass conservation within threshold: {total:.2e} <= {self.mass_conservation_threshold:.2e}")
        self._plot_residuals()
        self._dump_final()
        end_time = time.time()
        if self.verbose:
            print(f"Total runtime: {end_time - start_time:.2f} seconds")
        self.post_process()
        
        return True
