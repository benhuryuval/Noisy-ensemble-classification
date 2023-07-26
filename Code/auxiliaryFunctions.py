import numpy as np
import scipy
import csv
import sklearn as sk
import pandas as pd


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

def calc_constraint(beta, pNorm):
    if pNorm == 1:
        return np.linalg.norm(beta, pNorm)
    elif pNorm == 2:
        return np.sum(beta**2)
    elif pNorm==3:  # infinity norm
        return np.max(beta)

def calc_uniform(T, G, pNorm):
    if pNorm == 1:
        return G/T * np.ones([T,1])
    elif pNorm == 2:
        return np.sqrt(G/T) * np.ones([T,1])
    elif pNorm == 3:  # infinity norm
        return np.concatenate(([G], np.zeros([T,])))

def scale_to_constraint(beta, G, pNorm):
    if pNorm == 1:
        K = np.sum(np.absolute(beta))/G
    elif pNorm == 2:
        K = np.sqrt(np.sum(beta**2)/G)
    elif pNorm == 3:  # infinity norm
        K = np.max(beta)/G
    return beta / K

def load_dataset(database_name, database_path):
    if database_name == 'Wisconsin Breast Cancer Database':
        from sklearn.datasets import load_breast_cancer
        dataset = sk.datasets.load_breast_cancer()
        features = pd.DataFrame(dataset.data, columns=dataset.feature_names).to_numpy()
        labels = pd.Categorical.from_codes(dataset.target, dataset.target_names)
    elif database_name == 'Heart Diseases Database':
        dataset = np.genfromtxt(database_path, delimiter=',')
        features = dataset[:, 0:13]
        labels = dataset[:, 13] > 0
    elif database_name == 'Parkinson Database':
        dataset = np.genfromtxt(database_path, delimiter=',')
        features = dataset[1:, 1:23]
        labels = dataset[1:, 23]
    else:
        raise NameError('NoSuchDatabase')
    return features, labels