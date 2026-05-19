import numpy as np
import scipy.io as sio
from scipy.signal import detrend
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessPool

# from WorkBrainFolder import *

# If need to debug numba code, uncomment this
#from numba import config
#config.DISABLE_JIT = True

from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables.fc import FC
import neuronumba.observables.measures as measures

from Utils.decorators import loadOrCompute

import pietras2025
from MiniNeuroNumba.compact_generic_bold_model import Compact_Simulator
from MiniNeuroNumba.compact_bold_simulator import CompactMontbrioSimulator


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


# def parse_arguments():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--tmax", help="Simulation time (milliseconds)", type=float, default=1000.0)
#     parser.add_argument("--tr", help="Temporal resolution (TR) for the BOLD signal (milliseconds)", type=float, default=2000.0)
#     parser.add_argument("--dt", help=("Simulation delta-time (milliseconds)."), type=float, default=0.1)
#     parser.add_argument("--g", help="Global scaling for SC matrix normalization", type=float, default=1.0)

#     args = parser.parse_args()
#     return args  # returns something like: Namespace(model='Hopf', tmax=10000.0, tr=2000.0, dt=100, g=1.0)


def dataLoader(database='HCP'):
    """ 
    This function is used to load the data from different databases and return the correspondig object.
    In this case HCP or ADNI.

    Parameters
    ----------
    database : str
        Name of the database to load. Options: 'HCP', 'ADNI'
    
    Returns
    -------
    DataLoader object
    """
    
    if database == 'HCP':
        from DataLoaders.HCP_dbs80 import HCP
        return HCP()
    elif database == 'ADNI':
        from DataLoaders.ADNI_G import ADNI_G
        return ADNI_G()


def FC_mean(hcp):
    """
    For each subject observable FC is calculated, aplying first a band-pass filter to the BOLD signal. 
    Then the results are averaged into a single FC matrix. Then we changed the value of G and calculate the BOLD
    and choose the one that best fits the averaged FC. 
    """
    gCtrl = hcp.get_groupSubjects('REST1')[:50]
    FC_all = []
    for subj_id in gCtrl:
        subj_data = hcp.get_subjectData(subj_id)[subj_id]
        ts = subj_data['timeseries']
        measure = FC()
        bold_fit = filer_fMRI(ts.T) # input bold in (time, RoIs) format
        fc = measure._compute_from_fmri(bold_fit)
        FC_all.append(fc['FC'])
    FC_all = np.array(FC_all)  # shape: (N_subjectes, 80, 80)
    return np.mean(FC_all, axis=0)


@loadOrCompute
def run_subject_sim(cfg):
    DL = cfg['DL']
    subj_id = cfg['subj_id']
    g_values = cfg['g_values']

    compact_simulator = Compact_Simulator(
        model = pietras2025.Pietras2025(),
        obs_var = 'R_e_Hz',
        weights = cfg['sc_norm'],
        use_temporal_avg_monitor = False,
        g = 5.30,
        sigma = 1e-03,
        tr = cfg['tr']*1000,  # milliseconds
        dt = cfg['dt'],   # milliseconds
        use_bold = True # False for maxRate
    )

    subj_data = DL.get_subjectData(subj_id)[subj_id]
    ts = subj_data['timeseries']
    observable = FC()
    bold_fit = filer_fMRI(ts.T)  # input bold in (time, RoIs) format
    fc_subject = observable._compute_from_fmri(bold_fit)

    fc_corrs = np.full(len(g_values), np.nan)

    for g_indx, g in enumerate(g_values):  # For each value of G we calculate the FC and compare it with the subject FC, to find the optimal G for each subject
        print(f'  Fitting subject {subj_id} @ {g}')
        compact_simulator.g = g
        fcs_sim = []
        for _ in range(cfg['sim_repes']):  # Run multiple trials for each G and average results
            simulated_bold = compact_simulator.generate_bold(
                warmup_time=cfg['T_warm_seconds'] * 1000,  # This samples will be discarded
                simulated_time=cfg['T_sim_seconds'] * 1000
                # Number of useful samples to generate, this will be the size of the generated bold
            )
            # FC simulated
            bold_fit = filer_fMRI(simulated_bold)  # input bold in (time, RoIs) format
            fc_sim = observable._compute_from_fmri(bold_fit)
            fcs_sim.append(fc_sim['FC'])
        fc_average = np.mean(fcs_sim, axis=0)  # Average FC across trials for this G

        # Compare FC_sim with FC_subject
        pearsonDiss = measures.PearsonDissimilarity()
        PD_value = pearsonDiss.distance(fc_subject['FC'], fc_average)
        fc_corrs[g_indx] = PD_value
        print(f'  -> {PD_value}')
    optimum_g = g_values[np.argmin(fc_corrs)]
    return {'optim_g': optimum_g}


def run_subject(cfg):
    file_name = f'./_Data_Produced/{cfg['subj_id']}_{cfg["group"]}.mat'
    fc_subj_corr = run_subject_sim(cfg, file_name)['optim_g'].flatten()
    return fc_subj_corr


def run():
    cfg = {}
    #args = parse_arguments()

    # We generate a Mock-up structural connectivity (SC) matrix for the purpose of the example. In a real-world scenario
    # you should use the real one.
    # sc_norm = np.random.uniform(0.05, 0.2, size=(n_rois, n_rois))
    # np.fill_diagonal(sc_norm, 0.0)
    DL = dataLoader(database='ADNI')
    cfg['DL'] = DL
    sc_norm = DL.get_AvgSC_ctrl()
    cfg['sc_norm'] = sc_norm / np.max(sc_norm) * 0.2  # Normalize
    # plt.matshow(sc_norm)
    # plt.show()

    cfg['tr'] = 2.0
    cfg['dt'] = 0.01 # milliseconds (1e-4 seconds)
    cfg['Tmax_vol'] = 100
    cfg['T_sim_seconds'] = (cfg['Tmax_vol'] * cfg['tr'])
    cfg['T_warm_seconds'] = 10
    cfg['sim_repes'] = 2

    num_AD_subjects = len(DL.get_groupSubjects('AD'))
    subjs={group:DL.get_groupSubjects(group)[:num_AD_subjects]
           for group in DL.get_groupLabels()}

    # g_values = np.array([1., 2.])  # 10 values between 0 and 10
    g_values = np.arange(0., 2.7, 0.1)
    cfg['g_values'] = g_values

    # optimal_g = {group: np.full(len(subjs[group]), np.nan)
    #             for group in subjs}
    # # For each subject of the dataset we calculate the FC
    # for group in subjs:
    #     cfg['group'] = group
    #     for subj_indx, subj_id in enumerate(subjs[group]):
    #         cfg['subj_id'] = subj_id
    #         # Find optimal G for this subject
    #         fc_subj_corr = run_subject(subj_id, cfg)
    #         optimal_g[group][subj_indx] = fc_subj_corr[0]

    for group in subjs:
        cfg['group'] = group
        pool = ProcessPool(nodes=20)
        cfgs = []
        for subj_id in subjs[group]:
            conf = cfg.copy()
            conf['subj_id'] = subj_id
            cfgs.append(conf)
        results = pool.map(run_subject, cfgs)

    print('all done!')


if __name__ == '__main__':
    run()
