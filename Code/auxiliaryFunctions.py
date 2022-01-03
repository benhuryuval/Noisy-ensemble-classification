import numpy as np
import scipy
import csv


def QFunc(x):
    """ This function calculates the Q function: integral from x to inf of gaussian """
    return 0.5 - 0.5 * scipy.special.erf(x / np.sqrt(2))


def white_noise(var_m, n_estimators, n_data_samps, seed=None):
    """ Returns n_estimators*n_data_samps matrix, in which every row is an independent identically distributed random
    Gaussian vector with variance var_m[m] """
    noise_matrix = np.zeros([n_estimators, n_data_samps])
    # if seed is not None:
    #     np.random.seed(seed)
    for m in range(n_estimators):
        noise_matrix[m] = np.random.normal(0, np.sqrt(var_m[m]), n_data_samps)
    return noise_matrix


def write_csv_line(line, path_file):
    """ Writes line in csv file in pathFile"""
    f = open(path_file, 'a', newline='')
    with f:
        writer_fun = csv.writer(f)
        writer_fun.writerow(line)


def calc_accuracy(confusion_matrix):
    return confusion_matrix.trace() / confusion_matrix.sum()
