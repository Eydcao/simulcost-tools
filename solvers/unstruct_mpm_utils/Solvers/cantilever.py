import taichi as ti
import numpy as np
from .base import MPM_FIELD_BASE
from ..CTRL_DATAs.cantilever import draw_cantilever


@ti.data_oriented
class MPM_FIELD_CANTILEVER(MPM_FIELD_BASE):
    # def declare_additional_fields(self):
    #     # TODO implement in override
    #     pass

    # @ti.kernel
    # def zero_clear_additional_fields(self):
    #     # TODO implement in override
    #     pass

    # @ti.kernel
    # def call_back(self, t: float):
    #     # implement in override
    #     pass

    @ti.kernel
    def project_vertex_dirichelet(self):
        # hardcode dirichelet
        for v in self.v_pos:
            tmp_pos = self.v_pos[v]
            # check left set vel to 0
            if tmp_pos[0] < self.ctrl_data.bc_dist_dx:
                self.v_vel[v] *= 0

    # @ti.kernel
    # def particle_body_force(self):
    #     # TODO implement in overide
    #     pass

    @ti.kernel
    def push_back_into_mesh(self):
        # iterate all particles' position
        # clamp it into the box size
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
        ) = draw_cantilever(self.ctrl_data.dx, self.ctrl_data.envs_params, self.ctrl_data.n_part)
        for p in self.x:
            # push in x direction
            if self.x[p][0] < 0 + self.ctrl_data.bc_dist_dx:
                self.x[p][0] = 0 + self.ctrl_data.bc_dist_dx
            elif self.x[p][0] > Lx - self.ctrl_data.bc_dist_dx:
                self.x[p][0] = Lx - self.ctrl_data.bc_dist_dx
            # push in y direction
            if self.x[p][1] < 0 + self.ctrl_data.bc_dist_dy:
                self.x[p][1] = 0 + self.ctrl_data.bc_dist_dy
            elif self.x[p][1] > Ly - self.ctrl_data.bc_dist_dy:
                self.x[p][1] = Ly - self.ctrl_data.bc_dist_dy


if __name__ == "__main__":
    pass
