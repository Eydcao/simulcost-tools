import taichi as ti
import numpy as np
from .base import MPM_FIELD_BASE


@ti.data_oriented
class MPM_FIELD_DISK_COLLISION(MPM_FIELD_BASE):
    # def declare_additional_fields(self):
    #     # TODO implement in override
    #     pass

    #     x_momentum = ti.field(dtype=float, shape=1)
    #     sampled_theta_xx = ti.field(dtype=float, shape=1)

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
        # TODO implement in overide
        pass

    # @ti.kernel
    # def particle_body_force(self):
    #     # TODO implement in overide
    #     pass

    @ti.kernel
    def push_back_into_mesh(self):
        # iterate all particles' position
        # clamp it into the box size
        Lx, Ly, bc_dist_dx, bc_dist_dy = (
            self.ctrl_data.Lx,
            self.ctrl_data.Ly,
            self.ctrl_data.bc_dist_dx,
            self.ctrl_data.bc_dist_dy,
        )
        for p in self.x:
            # push in x direction
            if self.x[p][0] < 0 + bc_dist_dx:
                self.x[p][0] = 0 + bc_dist_dx
            elif self.x[p][0] > Lx - bc_dist_dx:
                self.x[p][0] = Lx - bc_dist_dx
            # push in y direction
            if self.x[p][1] < 0 + bc_dist_dy:
                self.x[p][1] = 0 + bc_dist_dy
            elif self.x[p][1] > Ly - bc_dist_dy:
                self.x[p][1] = Ly - bc_dist_dy


if __name__ == "__main__":
    pass
