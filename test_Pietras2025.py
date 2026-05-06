# =======================================================================
# Convenience simplification layer for NeuroNumba:
#     https://github.com/neich/neuronumba
#
# By Albert Juncà
# adapted by Gustavo Patow
# =======================================================================
import argparse
import math

import numpy as np
import scipy.io as sio
from scipy.signal import detrend
from matplotlib import pyplot as plt
from HCP_dbs80 import HCP
from WorkBrainFolder import *

# If need to debug numba code, uncomment this
#from numba import config
#config.DISABLE_JIT = True

from neuronumba.tools.filters import BandPassFilter

import Pietras2025
from compact_generic_bold_model import Compact_Simulator
from compact_bold_simulator import CompactMontbrioSimulator


def filer_fMRI(fMRI):  # fMRI in (time, RoIs) format
    # ========================================================================
    # We create the bandpass filter we will use for the signals
    # 3 Filters(Bandpass 0.008 - 0.08 Hz)
    flp = 0.008
    fhi = 0.08
    k = 2
    TR = 2.0

    bpf = BandPassFilter(
        k=k,
        flp=flp,
        fhi=fhi,
        tr=TR * 1000.,
        apply_detrend=True,
        apply_demean=True,
        remove_artifacts=True
    )
    return bpf.filter(fMRI)


def run_max_firing_rate(cfg):
    compact_simulator = Compact_Simulator(
        model = Pietras2025.Pietras2025(),
        obs_var = 'R_e_Hz',
        weights = cfg['sc_norm'],
        use_temporal_avg_monitor = False,
        # g = cfg['g'],
        sigma = cfg['sigma'],
        tr = cfg['tr']*1000,  # milliseconds
        dt = cfg['dt'],   # milliseconds
        use_bold = False # False for maxRate
    )
    g_values = np.linspace(cfg['g_min'], cfg['g_max'], cfg['g_steps'])  # 100 values between 0 and 10
    max_rates = []
    for g in g_values:
        compact_simulator.g = g
        print(f"Running ({g})", end=" / ")
        simulated_bold = compact_simulator.generate_bold(
            warmup_time=cfg['T_warm_seconds'] * 1000,  # This samples will be discarded
            simulated_time=cfg['T_sim_seconds'] * 1000
            # Number of useful samples to generate, this will be the size of the generated bold
        )
        maxRate = np.max(np.mean(simulated_bold, axis=0))
        max_rates.append(maxRate)
        print(f"MaxRate = {maxRate}")

    fig, axs = plt.subplots(1)
    fig.suptitle(f'Maximum mean rate vs coupling g')
    axs.set_xlabel('Coupling g')
    axs.set_ylabel('Maximum mean firing rate [Hz]')
    axs.plot(g_values, max_rates)
    plt.show()


def run_BOLD(cfg, plot=True):
    compact_simulator = Compact_Simulator(
        model = Pietras2025.Pietras2025(),
        obs_var = 'R_e_Hz',
        weights = cfg['sc_norm'],
        use_temporal_avg_monitor = False,
        g = cfg['g'],
        sigma = cfg['sigma'],
        tr = cfg['tr']*1000,  # milliseconds
        dt = cfg['dt'],   # milliseconds
        use_bold = True # False for maxRate
    )

    simulated_bold = compact_simulator.generate_bold(
        warmup_time = cfg['T_warm_seconds'] * 1000, # This samples will be discarded
        simulated_time = cfg['T_sim_seconds'] * 1000   # Number of useful samples to generate, this will be the size of the generated bold
    )

    if plot:
        fig, axs = plt.subplots(1)
        fig.suptitle(f'Result for model Pietras2025 (g={cfg['g']})')
        axs.plot(np.arange(simulated_bold.shape[0]), simulated_bold)
        plt.show()

    rates = np.mean(simulated_bold, axis=0)
    return rates



def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tmax", help="Simulation time (milliseconds)", type=float, default=1000.0)
    parser.add_argument("--tr", help="Temporal resolution (TR) for the BOLD signal (milliseconds)", type=float, default=2000.0)
    parser.add_argument("--dt", help=("Simulation delta-time (milliseconds)."), type=float, default=0.1)
    parser.add_argument("--g", help="Global scaling for SC matrix normalization", type=float, default=1.0)

    args = parser.parse_args()
    return args  # returns something like: Namespace(model='Hopf', tmax=10000.0, tr=2000.0, dt=100, g=1.0)


def run():
    # args = parse_arguments()
    cfg = {}

    hcp = HCP()
    sc_norm = hcp.get_AvgSC_ctrl()
    #sc_norm = sio.loadmat('./_Data_Raw/CNT_S01_structure.mat')['CNT_S01_structure']
    cfg['sc_norm'] = sc_norm / np.max(sc_norm) * 0.2  # Normalize
    #sc_norm = np.array([[0.0]])
    # plt.matshow(sc_norm)
    # plt.show()

    cfg['tr'] = 2.0
    cfg['dt'] = 0.01 # milliseconds (1e-5 seconds)
    Tmax_vol = 295
    cfg['T_sim_seconds'] = (Tmax_vol * cfg['tr'])
    cfg['T_warm_seconds'] = 20
    cfg['sigma'] = 1e-03

    # =============== compute the rates for two different
    cfg['g'] = 3.0
    rates = run_BOLD(cfg, plot=False)
    np.save(f'_Data_Produced/rates_{cfg['g']}.npy', rates)

    cfg['g'] = 4.0
    rates = run_BOLD(cfg, plot=False)
    np.save(f'_Data_Produced/rates_{cfg['g']}.npy', rates)

    # =============== compute the max firing rate vs g plot
    cfg['g_min'] = 3.25
    cfg['g_max'] = 3.4
    cfg['g_steps'] = 20
    run_max_firing_rate(cfg)


if __name__ == '__main__':
    run()
