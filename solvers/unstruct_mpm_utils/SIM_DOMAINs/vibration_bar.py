import taichi as ti
import numpy as np
import os
import meshio
import matplotlib.pyplot as plt
from ..Solvers.vibration_bar import MPM_FIELD_VIBRATION_BAR
from ..CTRL_DATAs.vibration_bar import CTRL_DATA_VIBRATION_BAR
from .base import SIM_DOMAIN


@ti.data_oriented
class SIM_DOMAIN_VIBRATION_BAR(SIM_DOMAIN):
    def __init__(self, dx, n_part, cfl, flip_ratio, advect_scheme, verbose, output_dir, envs_params):
        CTRL_DATA_CLASS, MPM_FIELD_CLASS = (
            CTRL_DATA_VIBRATION_BAR,
            MPM_FIELD_VIBRATION_BAR,
        )
        super(SIM_DOMAIN_VIBRATION_BAR, self).__init__(
            dx, n_part, cfl, flip_ratio, advect_scheme, verbose, output_dir, envs_params, CTRL_DATA_CLASS, MPM_FIELD_CLASS
        )

    def create_additional_fields(self):
        sample_pos = np.array([17.5, 0.5])
        self.sample_idx = np.argmin(np.linalg.norm(self.ctrl_data.p_pos - sample_pos, axis=1))
        self.sampled_pos = self.ctrl_data.p_pos[self.sample_idx]
        self.sample_stress_log = np.zeros((self.ctrl_data.end_frame + 1, 1))

        self.mass_center_x_log = np.zeros((self.ctrl_data.end_frame + 1, 1))
        self.kin_eng_log = np.zeros((self.ctrl_data.end_frame + 1, 1))
        self.pot_eng_log = np.zeros((self.ctrl_data.end_frame + 1, 1))

    # def pre_process(self):
    #     # TODO implement in override
    #     pass

    def call_back(self, frame):
        self.sample_stress_log[frame] = self.mpm_field.sigma[self.sample_idx][0, 0]
        self.mass_center_x_log[frame] = self.mpm_field.x.to_numpy()[:, 0].mean() - 17.5
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
        plt.figure(figsize=(8, 4))
        plt.rcParams["figure.dpi"] = paint_res
        # set font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.xlabel(r"$t[s]$")
        plt.ylabel(r"$u_x[m]$")
        plt.xlim(0, end_time)
        # set legend
        legend = [r"$ours$"]
        # time series: [0:end_frame] * frame_dt
        plt.plot(
            np.arange(0, end_frame + 1) * frame_dt,
            self.mass_center_x_log,
            label=legend[0],
        )
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(dump_path, f"mass_center_x_log.png"), dpi=paint_res)
        np.save(os.path.join(dump_path, f"mass_center_x_log.npy"), self.mass_center_x_log)

        # total energy
        # clean plt
        plt.clf()
        # plot energies
        plt.figure(figsize=(8, 4))
        plt.rcParams["figure.dpi"] = paint_res
        # set font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.xlabel(r"$t[s]$")
        plt.ylabel(r"$E[J]$")
        plt.xlim(0, end_time)
        # set legend
        legend = [r"$E_{kin}$", r"$E_{pot}$", r"$E_{grav}$", r"$E_{tot}$"]
        # plot kin, pot, and tot vs time series: [0:end_frame] * frame_dt
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

        # sampled stress
        # clean plt
        plt.clf()
        # plot energies
        plt.figure(figsize=(24, 4))
        plt.rcParams["figure.dpi"] = paint_res
        # set font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.xlabel(r"$t[s]$")
        plt.ylabel(r"$Stress[Pa]$")
        plt.xlim(0, end_time)
        # set legend
        legend = [r"Ours$"]
        plt.plot(
            np.arange(0, end_frame + 1) * frame_dt,
            self.sample_stress_log,
            label=legend[0],
        )
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(dump_path, f"stress.png"), dpi=paint_res)
        np.save(os.path.join(dump_path, f"stress.npy"), self.sample_stress_log)


if __name__ == "__main__":
    pass
