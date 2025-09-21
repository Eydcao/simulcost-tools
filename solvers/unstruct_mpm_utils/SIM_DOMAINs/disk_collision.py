import taichi as ti
import numpy as np
import os
import meshio
import matplotlib.pyplot as plt
from ..Solvers.disk_collision import MPM_FIELD_DISK_COLLISION
from ..CTRL_DATAs.disk_collision import CTRL_DATA_DISK_COLLISION
from .base import SIM_DOMAIN


@ti.data_oriented
class SIM_DOMAIN_DISK_COLLISION(SIM_DOMAIN):
    def __init__(self, dx, n_part, cfl, flip_ratio, advect_scheme, verbose, output_dir, envs_params):
        CTRL_DATA_CLASS, MPM_FIELD_CLASS = (
            CTRL_DATA_DISK_COLLISION,
            MPM_FIELD_DISK_COLLISION,
        )
        super(SIM_DOMAIN_DISK_COLLISION, self).__init__(
            dx, n_part, cfl, flip_ratio, advect_scheme, verbose, output_dir, envs_params, CTRL_DATA_CLASS, MPM_FIELD_CLASS
        )

    def create_additional_fields(self):
        bc_dist_dx, bc_dist_dy = self.ctrl_data.bc_dist_dx, self.ctrl_data.bc_dist_dy

        sample_pos = np.array([0.0 + bc_dist_dx * 2 + 0.2, 0.0 + bc_dist_dy * 2 + 0.2])
        self.sample_idx = np.argmin(np.linalg.norm(self.ctrl_data.p_pos - sample_pos, axis=1))
        self.sampled_pos = self.ctrl_data.p_pos[self.sample_idx]

        self.sample_thetaxx_log = np.zeros((self.ctrl_data.end_frame + 1, 1))
        self.x_momentum_log = np.zeros((self.ctrl_data.end_frame + 1, 1))
        self.kin_eng_log = np.zeros((self.ctrl_data.end_frame + 1, 1))
        self.pot_eng_log = np.zeros((self.ctrl_data.end_frame + 1, 1))

    # def pre_process(self):
    #     # TODO implement in override
    #     pass

    def call_back(self, frame):
        self.sample_thetaxx_log[frame] = self.mpm_field.sigma[self.sample_idx][0, 0]
        self.x_momentum_log[frame] = (
            self.mpm_field.v.to_numpy()[: self.ctrl_data.n_p // 2, 0].sum() * self.ctrl_data.p_vol * self.ctrl_data.rho
        )
        self.kin_eng_log[frame] = self.mpm_field.kin_eng[0]
        self.pot_eng_log[frame] = self.mpm_field.pot_eng[0]

    def post_process(self):
        # plot
        paint_res = 300
        dump_path = self.ctrl_data.cache_dir
        end_frame, frame_dt, end_time = (
            self.ctrl_data.end_frame,
            self.ctrl_data.frame_dt,
            self.ctrl_data.end_frame * self.ctrl_data.frame_dt,
        )

        # sampled theta xx
        # clean plt
        plt.clf()
        # set resolution
        plt.figure(figsize=(16, 4))
        plt.rcParams["figure.dpi"] = paint_res
        # set font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.xlabel(r"$t[s]$")
        plt.ylabel(r"$\theta_{xx}[Pa]$")
        plt.xlim(1.5, 3)
        plt.ylim(-110, 110)
        # set legend
        legend = [r"$ours$"]
        # time series: [0:end_frame] * frame_dt
        plt.plot(
            np.arange(0, end_frame + 1) * frame_dt,
            self.sample_thetaxx_log,
            label=legend[0],
        )
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(dump_path, f"sample_thetaxx_log.png"), dpi=paint_res)
        np.save(os.path.join(dump_path, f"sample_thetaxx_log.npy"), self.sample_thetaxx_log)

        # x momentum
        # clean plt
        plt.clf()
        # set resolution
        plt.figure(figsize=(8, 4))
        plt.rcParams["figure.dpi"] = paint_res
        # set font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.xlabel(r"$t[s]$")
        plt.ylabel(r"$L_{x}[kg m/s]$")
        plt.xlim(0, end_time)
        # plt.ylim(-12, 12)
        # set legend
        legend = [r"$ours$"]
        # time series: [0:end_frame] * frame_dt
        plt.plot(np.arange(0, end_frame + 1) * frame_dt, self.x_momentum_log, label=legend[0])
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(dump_path, f"x_momentum_log.png"), dpi=paint_res)
        np.save(os.path.join(dump_path, f"x_momentum_log.npy"), self.x_momentum_log)

        # total energy
        # clean plt
        plt.clf()
        # plot energies
        plt.figure(figsize=(24, 4))
        plt.rcParams["figure.dpi"] = paint_res
        # set font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.xlabel(r"$t[s]$")
        plt.ylabel(r"$E[J]$")
        # set the unit of y axis to be 10^(-3)
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(-7, -7))
        plt.xlim(0, end_time)
        plt.ylim(-0.25, 3)
        # set legend
        legend = [r"$E_{kin}$", r"$E_{pot}$", r"$E_{grav}$", r"$E_{tot}$"]
        # plot kin, pot, and tot vs time series: [0:end_frame] * frame_dt
        # tot_eng_log = kin_eng_log + pot_eng_log + p_gra_eng_log
        tot_eng_log = self.kin_eng_log + self.pot_eng_log
        plt.plot(np.arange(0, end_frame + 1) * frame_dt, self.kin_eng_log, label=legend[0])
        plt.plot(np.arange(0, end_frame + 1) * frame_dt, self.pot_eng_log, label=legend[1])
        # plt.plot(np.arange(0, end_frame + 1) * frame_dt, p_gra_eng_log, label=legend[2])
        plt.plot(np.arange(0, end_frame + 1) * frame_dt, tot_eng_log, label=legend[3])
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(dump_path, f"energies.png"), dpi=paint_res)
        np.savez(
            os.path.join(dump_path, f"energies.npz"),
            pot=self.pot_eng_log,
            kin=self.kin_eng_log,
            tot=tot_eng_log,
        )


if __name__ == "__main__":
    pass
