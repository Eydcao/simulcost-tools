import taichi as ti
import numpy as np
import os
import meshio
import matplotlib.pyplot as plt
from ..Solvers.cantilever import MPM_FIELD_CANTILEVER
from ..CTRL_DATAs.cantilever import CTRL_DATA_CANTILEVER
from .base import SIM_DOMAIN


@ti.data_oriented
class SIM_DOMAIN_CANTILEVER(SIM_DOMAIN):
    def __init__(
        self,
        dx,
        n_part,
        cfl,
        radii,
        flip_ratio,
        advect_scheme,
        verbose,
        output_dir,
        envs_params,
        CTRL_DATA_CLASS=CTRL_DATA_CANTILEVER,
        MPM_FIELD_CLASS=MPM_FIELD_CANTILEVER,
    ):
        super(SIM_DOMAIN_CANTILEVER, self).__init__(dx, n_part, cfl, radii, flip_ratio, advect_scheme, verbose, output_dir, envs_params, CTRL_DATA_CLASS, MPM_FIELD_CLASS)

    def create_additional_fields(self):
        sample_pos = np.array([10.0, 5.0 + 1.0])
        self.sample_idx = np.argmin(np.linalg.norm(self.ctrl_data.p_pos - sample_pos, axis=1))
        self.sampled_pos = self.ctrl_data.p_pos[self.sample_idx]
        self.sample_p_disy_log = np.zeros((self.ctrl_data.end_frame + 1, 1))
        self.kin_eng_log = np.zeros((self.ctrl_data.end_frame + 1, 1))
        self.pot_eng_log = np.zeros((self.ctrl_data.end_frame + 1, 1))
        self.gra_eng_log = np.zeros((self.ctrl_data.end_frame + 1, 1))

    # def pre_process(self):
    #     # TODO implement in override
    #     pass

    def call_back(self, frame):
        self.sample_p_disy_log[frame] = self.mpm_field.x[self.sample_idx][1] - self.sampled_pos[1]
        self.kin_eng_log[frame] = self.mpm_field.kin_eng[0]
        self.pot_eng_log[frame] = self.mpm_field.pot_eng[0]
        self.gra_eng_log[frame] = self.mpm_field.gra_eng[0]

    def post_process(self):
        # plot
        paint_res = 300

        # middle point dis y
        # clean plt
        plt.clf()
        # set resolution
        plt.rcParams["figure.dpi"] = paint_res
        # set font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.xlabel(r"$t[s]$")
        plt.ylabel(r"$u_y[m]$")
        # plt.xlim(0, 3)
        # plt.ylim(-3.5, 0)
        # set legend
        legend = [r"$ours$"]
        # time series: [0:end_frame] * frame_dt
        plt.plot(
            np.arange(0, self.ctrl_data.end_frame + 1) * self.ctrl_data.frame_dt,
            self.sample_p_disy_log,
            label=legend[0],
        )
        plt.legend(loc="upper right")
        plt.savefig(
            os.path.join(self.ctrl_data.cache_dir, f"sample_p_disy_log.png"),
            dpi=paint_res,
        )
        np.save(
            os.path.join(self.ctrl_data.cache_dir, f"sample_p_disy_log.npy"),
            self.sample_p_disy_log,
        )

        # total energy
        # clean plt
        plt.clf()
        # plot energies
        plt.rcParams["figure.dpi"] = paint_res
        # set font
        plt.rcParams["font.family"] = "Times New Roman"
        plt.xlabel(r"$t[s]$")
        plt.ylabel(r"$E[J]$")
        # set the unit of y axis to be 10^(-3)
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(-7, -7))
        # plt.xlim(0, 2.5)
        # plt.ylim(-75, 75)
        # set legend
        legend = [r"$E_{kin}$", r"$E_{pot}$", r"$E_{grav}$", r"$E_{tot}$"]
        # plot kin, pot, and tot vs time series: [0:end_frame] * frame_dt
        self.gra_eng_log -= self.gra_eng_log[0]  # rid of the initial height
        tot_eng_log = self.kin_eng_log + self.pot_eng_log + self.gra_eng_log
        plt.plot(
            np.arange(0, self.ctrl_data.end_frame + 1) * self.ctrl_data.frame_dt,
            self.kin_eng_log,
            label=legend[0],
        )
        plt.plot(
            np.arange(0, self.ctrl_data.end_frame + 1) * self.ctrl_data.frame_dt,
            self.pot_eng_log,
            label=legend[1],
        )
        plt.plot(
            np.arange(0, self.ctrl_data.end_frame + 1) * self.ctrl_data.frame_dt,
            self.gra_eng_log,
            label=legend[2],
        )
        plt.plot(
            np.arange(0, self.ctrl_data.end_frame + 1) * self.ctrl_data.frame_dt,
            tot_eng_log,
            label=legend[3],
        )
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.ctrl_data.cache_dir, f"energies.png"), dpi=paint_res)
        np.savez(
            os.path.join(self.ctrl_data.cache_dir, f"energies.npz"),
            pot=self.pot_eng_log,
            kin=self.kin_eng_log,
            gra=self.gra_eng_log,
            tot=tot_eng_log,
        )


if __name__ == "__main__":
    pass
