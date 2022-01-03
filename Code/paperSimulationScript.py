# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 08:14:17 2020
Power-Constrained Ensemble Classification with Noisy Soft Confidence Levels
Datasets for simulations: Heart Diseases Database, Parkinson Database, Wisconsin Breast Cancer Database
Noise model: AWGN (additive white Gaussian noise)
Base classifiers: decision trees
Ensemble method: AdaBoost + weighted combining
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

# def main():
np.random.seed(42)  # set rng seed for reproducibility
relativeRootPath = "..\\"
resultsFolderPath = relativeRootPath + "Results\\"
dataFolderPath = relativeRootPath + "Data\\"
csvHeader = ["Database", "Time", "SNR", "Number of classifiers", "Power limit",
            "Lower bound (mismatch probability)", "Upper bound (mismatch probability)",
            "Mismatch probability (trivial)", "Error probability (trivial)",
            "Mismatch probability (alpha, gd_es_min)", "Error probability (alpha, gd_es_min)",
            "Mismatch probability (alpha, gd_vanilla)", "Error probability (alpha, gd_vanilla)",
            "Mismatch probability (alpha_beta)", "Error probability (alpha_beta)"]

# simulation parameters
databaseList = ["Heart Diseases Database", 'Parkinson Database', 'Wisconsin Breast Cancer Database']
databasePathList = [dataFolderPath + "processed_cleveland_data_fixed.csv", dataFolderPath + "parkinsons_data_fixed.csv", dataFolderPath + ""]
# databaseList = ["Heart Diseases Database"]
# databasePathList = [dataFolderPath + "processed_cleveland_data_fixed.csv"]

# simulation and dataset parameters
numBaseClassifiersList = [45] #[15, 30] # [15, 30, 45, 60]  # loop over number of base classifiers
powerLimitList = [45] #[1.5, 15, 150] #[np.sqrt(45)] #[0.1, 1, 2] # power limitation, sum_t \beta_t^2 < PowerLimit
pNorm = 2  # parameter p for p-norm in constraint for \beta
train_size = 0.8  # fraction of database used for training
simIterNum = 25  # number of iterations for each SNR

# model parameters
confidence_eps = 1e-8  # tolerance value to preventing division by 0 in confidence-level calculation
tie_breaker = 1e-8  # addition to confidence level to avoid zero value

# optimization parameters
max_iter = 2500  # maximum number of gradient-descent iterations
learn_rate = 0.2  # gradient-descent step size
decay_rate = 0.2  # 0.9
tol_val_baseline = 1e-5  # tolerance value to stop optimization upon convergence

# init snr array
snr_array_db = np.arange(-5, 21, 5)
snr_array = 10 ** (snr_array_db / 10)

# loop over different numbers of base classifiers
for numBaseClassifiers in numBaseClassifiersList:
    print('Simulating ' + numBaseClassifiers.__str__() + ' base classifiers...')

    # set noise variance profile vector
    sigma_profile = np.ones([numBaseClassifiers, 1])
    # sigma_profile[1::2] *= 0.01

    # loop over power limits
    for powerLimit in powerLimitList:

        # calculate uniform beta vector
        beta_uniform = powerLimit * np.ones([numBaseClassifiers, 1]) / np.linalg.norm(np.ones([numBaseClassifiers, 1]), pNorm)

        # loop over databases
        for database_idx, database in enumerate(databaseList):
            print('Database: ' + database)
            database_path = databasePathList[database_idx]

            # create csv file with header
            pathToCsvFile = resultsFolderPath + database + "_" + datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".csv"
            write_csv_line(csvHeader, pathToCsvFile)

            # initialize result arrays for simulation results
            simulationResultsList = {"Database": [], "Time": [], "SNR": [], "Number of classifiers": [], "Power limit": [],
                "Lower bound (mismatch probability)": [], "Upper bound (mismatch probability)": [],
                "Mismatch probability (trivial)": [], "Error probability (trivial)": [],
                "Mismatch probability (alpha, gd_es_min)": [], "Error probability (alpha, gd_es_min)": [],
                "Mismatch probability (alpha, gd_vanilla)": [], "Error probability (alpha, gd_vanilla)": [],
                "Mismatch probability (alpha_beta)": [], "Error probability (alpha_beta)": []}

            # loop over SNR values
            for snr in snr_array:
                tol_val = tol_val_baseline / snr  # update tolerance per snr
                print('SNR: ' + str(10 * np.log10(snr)) + '[dB]')

                # initialize confusion matrices (binary class)
                CM_trivial, CM_unconstrained, CM_unconstrained2, CM_constrained = (np.zeros([2, 2]) for i in range(4))
                CM_trivial_miss, CM_unconstrained_miss, CM_unconstrained2_miss, CM_constrained_miss = (np.zeros([2, 2]) for i in range(4))
                mismatchUpperBound, mismatchLowerBound = 0, 0  # initialize lower and upper bound

                # loop over iterations
                t0 = time.perf_counter()
                for iteration in range(simIterNum):
                    # - * - * - * - * - * - * Training * - * - * - * - * - * - * -
                    # train real adaboost model and calculate confidence levels for train and test sets
                    model = AdaBoostModel(database, database_path, train_size, numBaseClassifiers)
                    confidence_levels_train, confidence_levels_test = model.calc_confidence_levels(confidence_eps)

                    # scale noise variances to obtain desired average SNR
                    # curr_snr = np.sum(np.power(confidence_levels_train, 2)) / \
                    #     (np.sum(sigma_profile) * len(model.train_y))
                    # curr_snr = np.mean((confidence_levels_train.T.dot(beta_uniform))**2) / np.sum(sigma_profile)
                    curr_snr = powerLimit**2 * np.mean(np.linalg.norm(confidence_levels_train, ord=2, axis=0)**2) / (np.linalg.norm(sigma_profile, ord=2))
                    sigma = sigma_profile * (curr_snr / snr)

                    # optimize coefficients (\alpha and \beta)
                    optimizer = CoefficientsOptimizer(confidence_levels_train, sigma, powerLimit, pNorm)

                    alpha_gd_es_min, mismatch_gd_es_min = optimizer.optimize_unconstrained_coefficients(method='GD_ES_min',
                      tol_val=tol_val, max_iter=max_iter, learn_rate=learn_rate, decay_rate=decay_rate)

                    mismatch_prob_a, alpha, tmp, stop_iter_a = optimizer.optimize_constrained_coefficients(
                        method='Grad-Alpha', tol=tol_val, max_iter=max_iter)
                    alpha_gd_uni_beta = alpha[np.argmin(mismatch_prob_a[0:stop_iter_a+1])]

                    mismatch_prob_ab, alpha, beta, stop_iter_ab = optimizer.optimize_constrained_coefficients(
                        method='Frank-Wolfe', tol=tol_val, max_iter=max_iter)
                    alpha_ab = alpha[np.argmin(mismatch_prob_ab[0:stop_iter_ab+1])-1]
                    beta_ab = beta[np.maximum(0, np.argmin(mismatch_prob_ab[0:stop_iter_ab+1])-1)]

                    if iteration % 5 == 0:
                        fig_debug = plt.figure()
                        plt.plot(range(0, len(mismatch_gd_es_min), 1), mismatch_gd_es_min, label='Mismatch Prob (a, gd_es_min)', linestyle=':',
                                 marker='*', color='green')
                        plt.plot(range(0, stop_iter_a+1, 1), mismatch_prob_a[0:stop_iter_a+1], label='Mismatch Prob (a, gd_vanilla)', linestyle='--',
                                 marker='o', color='blue')
                        plt.plot(range(0, stop_iter_ab+1, 1), mismatch_prob_ab[0:stop_iter_ab+1], label='Mismatch Prob (ab)', linestyle='--',
                                 marker='x', color='red')
                        plt.legend(loc="upper right", fontsize=12)
                        plt.xlabel("Frank-Wolfe iteration", fontsize=14)
                        plt.ylabel("Mismatch probability", fontsize=14)
                        plt.pause(0.05)
                        plt.close(fig_debug)

                    # - * - * - * - * - * - * Training end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Noisy channel * - * - * - * - * - * - * -
                    # generate additive noise matrix
                    noise_matrix = white_noise(sigma, numBaseClassifiers, len(model.test_y))
                    noisy_predictor = NoisyPredictor_(model, noise_matrix, tie_breaker)
                    # - * - * - * - * - * - * Noisy channel end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Inference * - * - * - * - * - * - * -
                    # predict according to the noisy data with:
                    predictTest_trivial = noisy_predictor.optimalPredict(np.ones([numBaseClassifiers, 1]), beta_uniform, confidence_levels_test) #noisy_predictor.trivialPredict(confidence_levels_test)  # uniform alpha=1, uniform beta=1
                    predictTest_unconstrained = noisy_predictor.optimalPredict(alpha_gd_es_min, beta_uniform, confidence_levels_test)  # optimal alpha, uniform beta=1
                    predictTest_unconstrained2 = noisy_predictor.optimalPredict(alpha_gd_uni_beta, beta_uniform, confidence_levels_test)  # optimal alpha, uniform beta=1
                    predictTest_constrained = noisy_predictor.optimalPredict(alpha_ab, beta_ab, confidence_levels_test)  # optimal alpha, optimal beta
                    predictTest_noiseless = model.classifier.predict(model.test_X)  # predicting the non-noisy data as reference
                    # - * - * - * - * - * - * Inference end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Evaluation * - * - * - * - * - * - * -
                    # get confusion matrix between noisy inference and true class (for error probability)
                    CM_trivial += sk.metrics.confusion_matrix(model.test_y, predictTest_trivial[0])
                    CM_unconstrained += sk.metrics.confusion_matrix(model.test_y, predictTest_unconstrained[0])
                    CM_unconstrained2 += sk.metrics.confusion_matrix(model.test_y, predictTest_unconstrained2[0])
                    CM_constrained += sk.metrics.confusion_matrix(model.test_y, predictTest_constrained[0])

                    # get confusion matrix between noisy inference and noise-less inference (for mismatch probability)
                    CM_trivial_miss += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_trivial[0])
                    CM_unconstrained_miss += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_unconstrained[0])
                    CM_unconstrained2_miss += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_unconstrained2[0])
                    CM_constrained_miss += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_constrained[0])

                    # sum of upper and lower bounds over iterations
                    mismatchUpperBound += noisy_predictor.upperBound(confidence_levels_test, sigma, powerLimit, pNorm)
                    mismatchLowerBound += noisy_predictor.lowerBound(confidence_levels_test, sigma, powerLimit, pNorm)
                    # - * - * - * - * - * - * Evaluation end * - * - * - * - * - * - * -

                    if iteration % 10 == 0:
                        print('Iteration ' + str(iteration + 1) + ', Time elapsed: ' + str(time.perf_counter() - t0) + ' [s]')

                # - * - * - * - * - * - * Analyze results * - * - * - * - * - * - * -
                simulationResultsList["Database"].append(database)
                simulationResultsList["Time"].append(datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
                simulationResultsList["SNR"].append(10 * np.log10(snr))
                simulationResultsList["Number of classifiers"].append(numBaseClassifiers)
                simulationResultsList["Power limit"].append(powerLimit)
                simulationResultsList["Lower bound (mismatch probability)"].append(mismatchLowerBound / simIterNum * 100)
                simulationResultsList["Upper bound (mismatch probability)"].append(mismatchUpperBound / simIterNum * 100)
                simulationResultsList["Mismatch probability (trivial)"].append((1-calc_accuracy(CM_trivial_miss))*100)
                simulationResultsList["Error probability (trivial)"].append((1-calc_accuracy(CM_trivial)) * 100)
                simulationResultsList["Mismatch probability (alpha, gd_es_min)"].append((1 - calc_accuracy(CM_unconstrained_miss)) * 100)
                simulationResultsList["Error probability (alpha, gd_es_min)"].append((1 - calc_accuracy(CM_unconstrained)) * 100)
                simulationResultsList["Mismatch probability (alpha, gd_vanilla)"].append((1 - calc_accuracy(CM_unconstrained2_miss)) * 100)
                simulationResultsList["Error probability (alpha, gd_vanilla)"].append((1 - calc_accuracy(CM_unconstrained2)) * 100)
                simulationResultsList["Mismatch probability (alpha_beta)"].append((1 - calc_accuracy(CM_constrained_miss)) * 100)
                simulationResultsList["Error probability (alpha_beta)"].append((1 - calc_accuracy(CM_constrained)) * 100)
                # add line to simulation csv file
                line_for_csv = [simulationResultsList.get(key)[-1] for key in simulationResultsList]
                write_csv_line(line_for_csv, pathToCsvFile)

                print("Trivial: " + str(simulationResultsList["Mismatch probability (trivial)"][-1]))
                print("alpha, gd_es_min: " + str(simulationResultsList["Mismatch probability (alpha, gd_es_min)"][-1]))
                print("alpha, gd_vanilla: " + str(simulationResultsList["Mismatch probability (alpha, gd_vanilla)"][-1]))
                print("alpha_beta: " + str(simulationResultsList["Mismatch probability (alpha_beta)"][-1]))
                print("Simulation results saved to " + pathToCsvFile)
                # - * - * - * - * - * - * Analyze results end * - * - * - * - * - * - * -

            print("Completed simulating " + database)

            # plot error probability and mismatch probability
            fig = plt.figure()
            plt.plot(snr_array_db, simulationResultsList["Error probability (trivial)"], label='a,b: uniform')
            plt.plot(snr_array_db, simulationResultsList["Error probability (alpha, gd_es_min)"], label='a: gd_es_min, b: uniform')
            plt.plot(snr_array_db, simulationResultsList["Error probability (alpha, gd_vanilla)"], label='a: gd_vanilla, b: uniform')
            plt.plot(snr_array_db, simulationResultsList["Error probability (alpha_beta)"], label='a, b: frank-wolfe')
            plt.legend(loc="upper right", fontsize=12)
            matplotlib.pyplot.title(str(database) + "\n T=" + str(numBaseClassifiers) + ", P=" + str(powerLimit) + ", l=" + str(pNorm))
            plt.xlabel("SNR [dB]", fontsize=14)
            plt.ylabel("Classification error probability [%]", fontsize=14)
            plt.grid()
            plt.pause(0.05)

            fig = plt.figure()
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (trivial)"], label='a,b: uniform')
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha, gd_es_min)"], label='a: gd_es_min, b: uniform')
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha, gd_vanilla)"], label='a: gd_vanilla, b: uniform')
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha_beta)"], label='a, b: frank-wolfe')
            plt.plot(snr_array_db, simulationResultsList["Upper bound (mismatch probability)"], label='Upper bound')
            plt.plot(snr_array_db, simulationResultsList["Lower bound (mismatch probability)"], label='Lower bound')
            plt.legend(loc="upper right", fontsize=12)
            matplotlib.pyplot.title(str(database) + "\n T=" + str(numBaseClassifiers) + ", P=" + str(powerLimit) + ", l=" + str(pNorm))
            plt.xlabel("SNR [dB]", fontsize=14)
            plt.ylabel("Mismatch probability [%]", fontsize=14)
            plt.grid()
            plt.pause(0.05)
#
# if __name__ == "__main__":
#     main()
