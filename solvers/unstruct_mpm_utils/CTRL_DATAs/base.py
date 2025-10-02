import taichi as ti
import numpy as np
import os
from scipy.spatial import Delaunay
from ..helpers import build_connectivity, ADV_TYPE, ADV_TYPE_STR
import json
import pickle


@ti.data_oriented
class CTRL_DATA:
    def __init__(self, cache_dir: str, envs_params):
        self.cache_dir = cache_dir
        self.envs_params = envs_params
        self.source_json = []
        JSON_file = os.path.join(cache_dir, "ctrl_data.json")
        with open(JSON_file) as f:
            self.source_json = json.load(f)
        self.DIM = self.source_json["DIM"]
        self.dt = self.source_json["dt"]
        self.frame_dt = self.source_json["frame_dt"]
        self.end_frame = self.source_json["end_frame"]

        self.dx = self.source_json["dx"]
        self.dy = self.source_json["dy"]
        self.dz = self.source_json["dz"]
        self.hash_dx = 2*np.array([self.dx, self.dy, self.dz]) if self.DIM == 3 else 2*np.array([self.dx, self.dy])
        self.eps = 1e-5

        self.verbose = self.source_json["verbose"]

        self.advect_scheme = ADV_TYPE(self.source_json["advect_scheme"])

        self.gravity = self.source_json["gravity"]
        self.rho = self.source_json["rho"]
        self.p_vol = self.source_json["p_vol"]
        self.p_mass = self.rho * self.p_vol
        self.E = self.source_json["E"]
        self.nu = self.source_json["nu"]
        self.mu_0, self.lambda_0 = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        # test if the geo_meta file is already there
        geo_meta_cached = os.path.exists(os.path.join(cache_dir, "geo_meta.pkl"))
        if not geo_meta_cached:
            print("(Py scope) recalc geo_meta")
            (
                self.p_pos,
                self.p_mat,
                self.p_vel,
                self.p_F,
                self.p_Jp,
            ) = self.init_particles()
            print("(Py scope) init particle done")
            self.v_pos, self.cell = self.init_grid_geometry()
            print("(Py scope) init grid done")
            (
                self.is_bc,
                self.outgoing_normal,
                self.N1_v_idx,
                self.N1_v_var,
                self.max_nonzero_N1_v,
                self.adj_N1_v_N0_v,
                self.hash2N0_idx,
                self.hash2N0_var,
                self.hash_min,
                self.hash_max,
                self.v_support_radii,
            ) = build_connectivity(self.v_pos, self.cell, self.hash_dx)
            print("(Py scope) build geo_meta done")
            print("(Py scope) custimized geo_meta done")
            # dump the geo_meta
            with open(os.path.join(cache_dir, "geo_meta.pkl"), "wb") as f:
                pickle.dump(
                    [
                        self.p_pos,
                        self.p_mat,
                        self.p_vel,
                        self.p_F,
                        self.p_Jp,
                        self.v_pos,
                        self.cell,
                        self.hash_dx,
                        self.is_bc,
                        self.outgoing_normal,
                        self.N1_v_idx,
                        self.N1_v_var,
                        self.max_nonzero_N1_v,
                        self.adj_N1_v_N0_v,
                        self.hash2N0_idx,
                        self.hash2N0_var,
                        self.hash_min,
                        self.hash_max,
                        self.v_support_radii,
                    ],
                    f,
                )
                print("(Py scope) dump geo_meta")

        else:
            with open(os.path.join(cache_dir, "geo_meta.pkl"), "rb") as f:
                print("(Py scope) load geo_meta")
                (
                    self.p_pos,
                    self.p_mat,
                    self.p_vel,
                    self.p_F,
                    self.p_Jp,
                    self.v_pos,
                    self.cell,
                    self.hash_dx,
                    self.is_bc,
                    self.outgoing_normal,
                    self.N1_v_idx,
                    self.N1_v_var,
                    self.max_nonzero_N1_v,
                    self.adj_N1_v_N0_v,
                    self.hash2N0_idx,
                    self.hash2N0_var,
                    self.hash_min,
                    self.hash_max,
                    self.v_support_radii,
                ) = pickle.load(f)
                print("(Py scope) load geo_meta done")

        self.custimized_geo_meta()

        self.n_p = self.p_pos.shape[0]
        self.n_v = self.v_pos.shape[0]
        self.n_cell = self.cell.shape[0]
        self.hash_stride = np.cumprod((self.hash_max - self.hash_min)[::-1])[::-1]
        self.hash_stride[:-1] = self.hash_stride[1:]
        self.hash_stride[-1] = 1
        print("(Py scope) init geo data done")

    def init_particles(self):
        # TODO implement in override
        pass

    def init_grid_geometry(self):
        # TODO implement in override
        pass

    def custimized_geo_meta(self):
        # TODO implement in override
        pass
