import taichi as ti
import numpy as np
import os
from scipy.spatial import Delaunay
from ..helpers import ADV_TYPE_STR
import json
from .base import CTRL_DATA


def draw_cantilever(dx, envs_params, n_part):
    Lx, Ly = envs_params["Lx"], envs_params["Ly"]
    # _e means effective, only in these cells there are particles
    n_gx = int(Lx / dx)
    n_gy = int(Ly / dx)
    n_gx_e, p_perx = int(10 * n_gx / Lx), n_part
    n_gy_e, p_pery = int(2 * n_gy / Ly), n_part
    n_particles = int(n_gx_e * n_gy_e * p_perx * p_pery)

    dy = Ly / n_gy

    return Lx, Ly, n_gx, n_gx_e, p_perx, n_gy, n_gy_e, p_pery, n_particles, dx, dy


# override the ctrl data
@ti.data_oriented
class CTRL_DATA_CANTILEVER(CTRL_DATA):
    def __init__(self, dx, n_part, cfl, radii, flip_ratio, advect_scheme, verbose, output_dir, envs_params):
        self.dx = dx
        self.n_part = n_part
        self.cfl = cfl
        self.radii = radii
        self.flip_ratio = flip_ratio
        self.advect_scheme = advect_scheme
        self.verbose = verbose
        # 0. fixed fields
        DIM = 2
        end_time = envs_params["end_time"]
        case_name = "cantilever"
        cache_dir = output_dir
        
        self.envs_params = envs_params

        # 1.set the corresponding data
        # Calculate quality from dx for backward compatibility with dt calculation
        quality = 0.5 / dx  # This maintains the same dt as before when dx=0.5
        dt = cfl / quality
        frame_dt = 0.02
        end_frame = int(end_time / frame_dt)

        (
            Lx,
            Ly,
            n_gx,
            n_gx_e,
            p_perx,
            n_gy,
            n_gy_e,
            p_pery,
            n_particles,
            dx,
            dy,
        ) = draw_cantilever(dx, self.envs_params, self.n_part)
        
        self.n_particles = n_particles

        gravity = envs_params["gravity"]
        p_vol, p_rho = dx * dy / p_perx / p_pery, envs_params["p_rho"]
        E, nu = envs_params["E"], envs_params["nu"]  # Young's modulus and Poisson's ratio

        # 2.create the cache folder
        os.makedirs(cache_dir, exist_ok=True)
        # 3.then dump the json file
        ctrl_data = {
            "DIM": DIM,
            "dt": dt,
            "frame_dt": frame_dt,
            "end_frame": end_frame,
            "dx": dx,
            "dy": dy,
            "dz": 0.0,
            "verbose": verbose,
            "advect_scheme": advect_scheme.value,
            "gravity": gravity,
            "rho": p_rho,
            "p_vol": p_vol,
            "E": E,
            "nu": nu,
            "n_particles": n_particles,
            "flip_ratio": flip_ratio,
        }
        json_path = os.path.join(cache_dir, "ctrl_data.json")
        json.dump(ctrl_data, open(json_path, "w"), indent=4)

        self.bc_dist_dx, self.bc_dist_dy = 0.05 * dx, 0.05 * dy
        # finally call super
        super(CTRL_DATA_CANTILEVER, self).__init__(cache_dir, self.envs_params)

    def init_particles(self):
        (
            Lx,
            Ly,
            n_gx,
            n_gx_e,
            p_perx,
            n_gy,
            n_gy_e,
            p_pery,
            n_particles,
            dx,
            dy,
        ) = draw_cantilever(self.dx, self.envs_params, self.n_part)
        DIM = 2
        
        self.n_particles = n_particles

        x = np.zeros((n_particles, DIM), dtype=float)
        v = np.zeros((n_particles, DIM), dtype=float)
        material = np.zeros(n_particles, dtype=int)
        F = np.zeros((n_particles, DIM, DIM), dtype=float)
        Jp = np.zeros(n_particles, dtype=float)
        dxp = dx / p_perx
        dyp = dx / p_pery
        x_start = 0.0
        y_start = 5.0
        for i in range(n_gx_e * p_perx):
            for j in range(n_gy_e * p_pery):
                idx = i * n_gy_e * p_pery + j
                x[idx] = np.array([x_start + dxp * 0.5 + dxp * i, y_start + dyp * 0.5 + dyp * j])
                material[idx] = 1  # 0: fluid 1: jelly 2: snow
                v[idx] = np.array([0, 0])
                F[idx] = np.array([[1, 0], [0, 1]])
                Jp[idx] = 1

        return x, material, v, F, Jp

    def init_grid_geometry(self):
        (
            Lx,
            Ly,
            n_gx,
            n_gx_e,
            p_perx,
            n_gy,
            n_gy_e,
            p_pery,
            n_particles,
            dx,
            dy,
        ) = draw_cantilever(self.dx, self.envs_params, self.n_part)
        DIM = 2
        # grid is a 2d array of vertices
        # shape is Lx * Ly
        # grid number is n_gy, p_pery
        # grid resolution is dx, dy
        self.n_particles = n_particles
        v_pos = np.zeros(((n_gx + 1) * (n_gy + 1), DIM), dtype=float)
        for i in range(n_gx + 1):
            for j in range(n_gy + 1):
                v_pos[i * (n_gy + 1) + j] = np.array([dx * i, dy * j])

        # cell
        D = Delaunay(v_pos)
        cell = D.simplices

        return v_pos, cell
