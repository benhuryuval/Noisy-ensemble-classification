# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 08:14:17 2020
Error Resilient Real AdaBoost - additive white Gaussian noise simulation
@authors: Yuval Ben-Hur
"""

import sklearn as sk
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import time
import datetime

from NoisyPredictor_ import NoisyPredictor_
from auxiliaryFunctions import white_noise, write_csv_line, calc_accuracy
from AdaBoostModel import AdaBoostModel
from CoefficientsOptimizer import CoefficientsOptimizer


np.random.seed(42)  # set rng seed for reproducibility
relativeRootPath = "..\\"
resultsFolderPath = relativeRootPath + "Results\\"
dataFolderPath = relativeRootPath + "Data\\"

# simulation parameters
# databaseList = ["Heart Diseases Database", 'Parkinson Database', 'Wisconsin Breast Cancer Database']
# databasePathList = [dataFolderPath + "processed_cleveland_data_fixed.csv", dataFolderPath + "parkinsons_data_fixed.csv", dataFolderPath + ""]
databaseList = ["Heart Diseases Database"]
databasePathList = [dataFolderPath + "processed_cleveland_data_fixed.csv"]

# simulation and dataset parameters
powerLimitList = [0.1, 1, 2]  # power limitation, sum_t \beta_t^2 < PowerLimit
numBaseClassifiersList = [30]  # [15, 30, 45, 60]  # loop over number of base classifiers
pNorm = 2  # parameter p for p-norm in constraint for \beta
train_size = 0.8  # fraction of database used for training
simIterNum = 10  # number of iterations for each SNR

# constants and optimization parameters
confidence_eps = 1e-8  # tolerance value to preventing division by 0 in confidence-level calculation
tie_breaker = 1e-8
max_iter = 30000  # maximum number of gradient-descent iterations
learn_rate = 0.2  # gradient-descent step size
decay_rate = 0.2  # 0.9
eps = 0.00001

# init snr array
snr_array_db = np.arange(-25, 15, 2)
snr_array = 10 ** (snr_array_db / 10)

# set specific parameters
numBaseClassifiers = numBaseClassifiersList[0]
powerLimit = powerLimitList[1]
database_idx = 0
database = databaseList[0]
database_path = databasePathList[database_idx]
snr = snr_array[10]

# - * - * - * - * - * - * Training * - * - * - * - * - * - * -
# train real adaboost model and calculate confidence levels for train and test sets
model = AdaBoostModel(database, database_path, train_size, numBaseClassifiers)
confidence_levels_train, confidence_levels_test = model.calc_confidence_levels(confidence_eps)

# normalize noise variances so that SNR is constant for all estimators
sigma = np.ones((numBaseClassifiers, 1))
X = np.sum(np.sum(np.multiply(confidence_levels_train, confidence_levels_train))) / (
            np.sum(sigma) * len(model.train_y))  #
sigma = sigma * (X / snr)

# optimize coefficients (\alpha and \beta)
optimizer = CoefficientsOptimizer(confidence_levels_train, sigma, powerLimit, pNorm)

# --- gradient function testing ---
# a, b = np.ones([numBaseClassifiers, 1]) / numBaseClassifiers, np.ones([numBaseClassifiers, 1]) / numBaseClassifiers
# h = confidence_levels_train
# sqrt_one_h_ht_one = abs(h.sum(axis=0))
# alpha_sigma_alpha = (np.power(a, 2) * sigma).sum()
# g = ((a * b).T @ h) * h.sum(axis=0) / (sqrt_one_h_ht_one * np.sqrt(alpha_sigma_alpha))
# # calculate gradient w.r.t alpha
# parenthesis_term = (b * h) * h.sum(axis=0) / sqrt_one_h_ht_one + g * (a * sigma) / alpha_sigma_alpha
# grad_alpha = -1 / np.sqrt(2 * np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1)
# # calculate gradient w.r.t beta
# parenthesis_term = (a * h) * h.sum(axis=0) / sqrt_one_h_ht_one / np.sqrt(alpha_sigma_alpha)
# grad_beta = -1 / np.sqrt(2 * np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1)
# # project gradient for beta on feasible domain
# w = ((a * h) * h.sum(axis=0)).sum(axis=1)
# if pNorm == 1:
#     s = np.zeros([len(h), 1])
#     s[np.argmax(w)] = powerLimit
# elif pNorm == 2:
#     s = powerLimit / np.sqrt(np.power(w, 2).sum()) * w
# # grad_a, grad_b = optimizer.gradient_constrained_alpha_beta(a, b)
# print("grad_a=")
# print(grad_alpha)
# print("grad_b=")
# print(grad_beta)
# --- gradient function testing (end) ---

mismatch_prob, alpha, beta, stop_iter = optimizer.optimize_constrained_coefficients(method='Frank-Wolfe')
print(alpha[stop_iter])
print(beta[stop_iter])
print(mismatch_prob[stop_iter])
fig = plt.figure()
plt.plot(range(0, stop_iter, 1), mismatch_prob[0:stop_iter], label='Mismatch Prob', linestyle='--', marker='v', color='blue')
plt.xlabel("Frank-Wolfe iteration", fontsize=14)
plt.ylabel("Mismatch probability [%]", fontsize=14)
# - * - * - * - * - * - * Training end * - * - * - * - * - * - * -

# - * - * - * - * - * - * Noisy channel * - * - * - * - * - * - * -
# generate additive noise matrix
noise_matrix = white_noise(sigma, numBaseClassifiers, len(model.test_y))
noisy_predictor = NoisyPredictor_(model, noise_matrix, tie_breaker)
# - * - * - * - * - * - * Noisy channel end * - * - * - * - * - * - * -

# # - * - * - * - * - * - * Inference * - * - * - * - * - * - * -
# --- inference test ---
# soft_decision = beta[stop_iter].T @ np.diag(alpha[stop_iter].reshape((len(alpha[stop_iter]),))) @ confidence_levels_train
# hard_decision = 0.5*(1+np.sign(soft_decision))
# --- inference test (end)
predictTest_constrained = noisy_predictor.optimalPredict(alpha[stop_iter], beta[stop_iter],
                                                         confidence_levels_test)  # optimal alpha, optimal beta
predictTest_unconstrained = noisy_predictor.optimalPredict(alpha[stop_iter], np.ones([numBaseClassifiers, 1]),
                                                           confidence_levels_test)  # optimal alpha, uniform beta

# # predict according to the noisy data with:
# # trivial allocation, vanilla gradient-descent, early stop and minimum selective gradient-descent
# predictTest_trivial = noisy_predictor.trivialPredict(confidence_levels_test)
# predictTest_gd_es_min = predictTest_trivial  # noisy_predictor.optimalPredict(alpha_gd_es_min.T, confidence_levels_test)
# # - * - * - * - * - * - * Inference end * - * - * - * - * - * - * -

CM_constrained = sk.metrics.confusion_matrix(model.test_y, predictTest_constrained[0]) #.reshape((len(predictTest_constrained),)))