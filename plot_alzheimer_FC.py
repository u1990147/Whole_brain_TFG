import numpy as np
import os
import scipy.io as sio

from ADNI_G import ADNI_G
import p_values_raincloud as plot


def load_data(DL):
    subjects = DL.get_classification()
    res = {group: [] for group in DL.get_groupLabels()}
    for subject in subjects:
        group = subjects[subject]
        file_name = f'./_Data_Produced/{subject}_{group}.mat'
        if os.path.exists(file_name):
            data = sio.loadmat(file_name)['optim_g']
            res[group].append(data)
    res = {group: np.array(res[group]).flatten() for group in res}
    return res


def run():
    DL = ADNI_G()
    data = load_data(DL)
    plot.plotComparisonAcrossLabels2(data)


if __name__ == '__main__':
    run()