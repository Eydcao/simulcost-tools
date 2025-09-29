import taichi as ti
import numpy as np
import os
from scipy.spatial import Delaunay
from ..helpers import ADV_TYPE_STR
import json
from .base import CTRL_DATA


def draw_vibration_bar(dx, n_part, envs_params):
    Lx, Ly = envs_params["Lx"], envs_params["Ly"]
    Lxe, Lye = envs_params["Lxe"], envs_params["Lye"]
    dy = min(dx, 1.0)

    bc_dist_dx, bc_dist_dy = 0.05 * dx, 0.05 * dy

    n_gx, p_perx = int(Lx / dx), n_part
    n_gy, p_pery = int(Ly / dy), n_part

    n_gx_e = int(Lxe / dx)
    n_gy_e = int(Lye / dy)

    p_in_disk = []
    dxp = dx / p_perx
    dyp = dy / p_pery
    x_start = envs_params["x_start"]
    y_start = envs_params["y_start"]
    for i in range(n_gx_e * p_perx):
        for j in range(n_gy_e * p_pery):
            idx = i * n_gy_e * p_pery + j
            temp_x = np.array([x_start + dxp * 0.5 + dxp * i, y_start + dyp * 0.5 + dyp * j])
            p_in_disk.append(temp_x)
    x = np.array(p_in_disk)
    # vel is 0.75 * sin(0.5 pi * x[0] / Lx)
    v0 = 0.75
    # v0 = 0.1
    vel = v0 * np.sin(0.5 * np.pi * (x[:, 0] - x_start) / Lxe)
    v = np.zeros_like(x)
    v[:, 0] = vel
    # create material F and Jp
    material = np.ones(x.shape[0], dtype=int)
    F = np.tile(np.array([[1, 0], [0, 1]]), (x.shape[0], 1, 1))
    Jp = np.ones(x.shape[0], dtype=float)

    n_particles = x.shape[0]

    return (
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
        x,
        v,
        material,
        F,
        Jp,
    )


# override the ctrl data
@ti.data_oriented
class CTRL_DATA_VIBRATION_BAR(CTRL_DATA):
    def __init__(self, dx, n_part, cfl, radii, flip_ratio, advect_scheme, verbose, output_dir, envs_params):
        self.n_part = n_part
        self.cfl = cfl
        self.radii = radii
        self.flip_ratio = flip_ratio
        self.dx = dx
        self.advect_scheme = advect_scheme
        self.verbose = verbose
        self.envs_params = envs_params

        # 0. fixed fields
        DIM = 2
        end_time = envs_params["end_time"]
        case_name = "vibration_bar"
        cache_dir = output_dir

        # 1.set the corresponding data
        # Calculate quality from dx for backward compatibility with dt calculation
        quality = 1.0 / dx  # This maintains the same dt as before when dx=1.0
        dt = cfl / quality
        frame_dt = 0.2
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
            x,
            v,
            material,
            F,
            Jp,
        ) = draw_vibration_bar(dx, n_part, envs_params)
        
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
            "dz": 0,
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
        self.Lx, self.Ly = Lx, Ly
        # finally call super
        super(CTRL_DATA_VIBRATION_BAR, self).__init__(cache_dir, self.envs_params)

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
            x,
            v,
            material,
            F,
            Jp,
        ) = draw_vibration_bar(self.dx, self.n_part, self.envs_params)
        self.n_particles = n_particles
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
            x,
            v,
            material,
            F,
            Jp,
        ) = draw_vibration_bar(self.dx, self.n_part, self.envs_params)
        DIM = 2
        self.n_particles = n_particles

        # grid is a 2d array of vertices
        # shape is Lx * Ly
        # grid number is n_gy, p_pery
        # grid resolution is dx, dy
        v_pos = np.zeros(((n_gx + 1) * (n_gy + 1), DIM), dtype=float)
        for i in range(n_gx + 1):
            for j in range(n_gy + 1):
                v_pos[i * (n_gy + 1) + j] = np.array([dx * i, dy * j])

        # cell
        D = Delaunay(v_pos)
        cell = D.simplices

        return v_pos, cell
