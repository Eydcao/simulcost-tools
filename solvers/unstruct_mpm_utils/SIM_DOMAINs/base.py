import taichi as ti
import numpy as np
import json
import os


@ti.data_oriented
class SIM_DOMAIN:
    def __init__(self, nx, n_part, cfl, radii, flip_ratio, advect_scheme, verbose, output_dir, envs_params, CTRL_DATA_CLASS, MPM_FIELD_CLASS):
        dx = envs_params["Lx"] / nx
        self.ctrl_data = CTRL_DATA_CLASS(dx, n_part, cfl, radii, flip_ratio, advect_scheme, verbose, output_dir, envs_params)
        self.mpm_field = MPM_FIELD_CLASS(self.ctrl_data)
        self.create_additional_fields()
        self.mpm_field.init_fields()
        self.cost = 0

    def create_additional_fields(self):
        # TODO implement in override
        pass

    def pre_process(self):
        # TODO implement in override
        pass

    def call_back(self, frame):
        # TODO implement in override
        pass

    def post_process(self):
        # TODO implement in override
        pass

    def run(self):
        self.pre_process()

        # dump background mesh
        self.mpm_field.dump_mesh()

        frame = 0
        self.mpm_field.dump(frame)

        for frame in range(self.ctrl_data.end_frame + 1):
            # print(f"frame {frame}")

            step_per_frame = int(self.ctrl_data.frame_dt // self.ctrl_data.dt)
            for s in range(step_per_frame):
                self.mpm_field.substep(frame * self.ctrl_data.frame_dt + s * self.ctrl_data.dt)

            self.call_back(frame)

            # Only dump 10 times in total
            if frame % (self.ctrl_data.end_frame // 10) == 0 or frame == self.ctrl_data.end_frame:
                self.mpm_field.dump(frame)

        self.cost = self.ctrl_data.n_particles + self.mpm_field.sum_each_part_neighbor_communication[0]
        self.post_process()


if __name__ == "__main__":
    pass
