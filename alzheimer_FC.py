import numpy as np
import scipy.io as sio
from scipy.signal import detrend
from matplotlib import pyplot as plt
from WorkBrainFolder import *

# If need to debug numba code, uncomment this
#from numba import config
#config.DISABLE_JIT = True

from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables.fc import FC
import neuronumba.observables.measures as measures

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
        from HCP_dbs80 import HCP
        return HCP()
    elif database == 'ADNI':
        from ADNI_A import ADNI_A
        return ADNI_A()

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

def run():
    #args = parse_arguments()

    # We generate a Mock-up structural connectivity (SC) matrix for the purpose of the example. In a real-world scenario
    # you should use the real one.
    # sc_norm = np.random.uniform(0.05, 0.2, size=(n_rois, n_rois))
    # np.fill_diagonal(sc_norm, 0.0)
    DL = dataLoader(database='HCP')
    sc_norm = DL.get_AvgSC_ctrl()
    #sc_norm = sio.loadmat('./_Data_Raw/CNT_S01_structure.mat')['CNT_S01_structure']
    sc_norm = sc_norm / np.max(sc_norm) * 0.2  # Normalize
    #sc_norm = np.array([[0.0]])
    # plt.matshow(sc_norm)
    # plt.show()

    # ts = sio.loadmat('./_Data_Raw/CNT.mat')['ts_emp_raw']
    # # Reorder AAL to Deco
    # left_idx = list(range(0, 90, 2))
    # right_idx = list(range(89,0,-2))
    # order_deco = left_idx + right_idx
    # ts_emp = ts[order_deco,:]
    # ts_emp = detrend(ts_emp)
    # ts_emp_filt = filer_fMRI(ts_emp.T).T
    # FC_emp = np.corrcoef(ts_emp_filt)
    #fc_mean = FC_mean(hcp)

    tr = 2.0
    dt = 0.1 # milliseconds (1e-4 seconds)
    Tmax_vol = 100
    T_sim_seconds = (Tmax_vol * tr)
    T_warm_seconds = 10
    sim_repes = 5


    compact_simulator = Compact_Simulator(
        model = Pietras2025.Pietras2025(),
        obs_var = 'R_e_Hz',
        weights = sc_norm,
        use_temporal_avg_monitor = False,
        g = 5.30,
        sigma = 1e-03,
        tr = tr*1000,  # milliseconds
        dt = dt,   # milliseconds
        use_bold = True # False for maxRate
    )

    subjs={group:DL.get_groupSubjects(group)[:10]
           for group in DL.get_groupLabels()}

    g_values = np.arange(0, 11, 1)  # 10 values between 0 and 10
    fc_corrs = {group: np.full((len(subjs[group]), len(g_values)), np.nan) 
                for group in subjs}
    optimal_g = {group: np.full(len(subjs[group]), np.nan)
                for group in subjs}

    # For each subject of the dataset we calculate the FC
    for group in subjs:
        for subj_indx, subj_id in enumerate(subjs[group]):
            subj_data = DL.get_subjectData(subj_id)[subj_id]
            ts = subj_data['timeseries']
            observable = FC()
            bold_fit = filer_fMRI(ts.T) # input bold in (time, RoIs) format
            fc_subject = observable._compute_from_fmri(bold_fit)

            for g_indx, g in enumerate(g_values): # For each value of G we calculate the FC and compare it with the subject FC, to find the optimal G for each subject
                compact_simulator.g = g
                print(g)
                fcs_sim = []
                for _ in range(sim_repes):  # Run multiple trials for each G and average results
                    simulated_bold = compact_simulator.generate_bold(
                        warmup_time = T_warm_seconds*1000, # This samples will be discarded
                        simulated_time = T_sim_seconds*1000   # Number of useful samples to generate, this will be the size of the generated bold
                    )
                    # FC simulated
                    bold_fit = filer_fMRI(simulated_bold) # input bold in (time, RoIs) format
                    fc_sim = observable._compute_from_fmri(bold_fit)
                    fcs_sim.append(fc_sim['FC'])
                fc_average = np.mean(fcs_sim, axis=0)  # Average FC across trials for this G
                
                
                # Compare FC_sim with FC_subject
                pearsonDiss = measures.PearsonDissimilarity()
                PD_value= pearsonDiss.distance(fc_subject['FC'], fc_average)
                fc_corrs[group][subj_indx, g_indx] = PD_value
            
            # Find optimal G for this subject
            optimal_g[group][subj_indx] = g_values[np.argmin(fc_corrs[group][subj_indx])]
    

    # Plot of PersonDissimilarity result for each G value of one subject
    subj_idx = 0  
    group = list(subjs.keys())[0]
    fig, axs = plt.subplots(1)
    fig.suptitle(f'Optimal G of subject {subj_idx} ({group})')
    axs.set_xlabel('Coupling g')
    axs.set_ylabel('Pearson Dissimilarity')
    axs.plot(g_values, fc_corrs[group][subj_idx])
    plt.show()

    # Plot mean for each group?

if __name__ == '__main__':
    run()
