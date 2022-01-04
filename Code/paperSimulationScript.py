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
            "Mismatch probability (alpha, unit beta)", "Error probability (alpha, unit beta)",
            "Mismatch probability (alpha, uniform beta)", "Error probability (alpha, uniform beta)",
            "Mismatch probability (alpha, beta)", "Error probability (alpha, beta)"]

# simulation parameters
databaseList = ["Heart Diseases Database", 'Parkinson Database', 'Wisconsin Breast Cancer Database']
databasePathList = [dataFolderPath + "processed_cleveland_data_fixed.csv", dataFolderPath + "parkinsons_data_fixed.csv", dataFolderPath + ""]

# simulation and dataset parameters
numBaseClassifiersList = [30, 60] #[15, 30] # [15, 30, 45, 60]  # loop over number of base classifiers
powerLimitList = [np.sqrt(30), np.sqrt(60)] #[1.5, 15, 150] #[np.sqrt(45)] #[0.1, 1, 2] # power limitation, sum_t \beta_t^2 < PowerLimit
pNorm = 2  # parameter p for p-norm in constraint for \beta
train_size = 0.8  # fraction of database used for training
simIterNum = 50  # number of iterations for each SNR

# model parameters
confidence_eps = 1e-8  # tolerance value to preventing division by 0 in confidence-level calculation
tie_breaker = 1e-8  # addition to confidence level to avoid zero value

# optimization parameters
max_iter = 250  # maximal number of gradient-descent iterations
min_iter = 15  # minimal number of gradient-descent iterations
learn_rate = 0.2  # gradient-descent step size
decay_rate = 0.2  # gradient-descent momentum
tol_val_baseline = 1e-5  # tolerance value to stop optimization upon convergence

# init snr array
snr_array_db = np.arange(-25, 16, 5)
# snr_array_db = np.array([20])
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
                "Mismatch probability (alpha, unit beta)": [], "Error probability (alpha, unit beta)": [],
                "Mismatch probability (alpha, uniform beta)": [], "Error probability (alpha, uniform beta)": [],
                "Mismatch probability (alpha, beta)": [], "Error probability (alpha, beta)": []}

            # loop over SNR values
            for snr in snr_array:
                tol_val = tol_val_baseline / snr  # update tolerance per snr
                print('SNR: ' + str(10 * np.log10(snr)) + '[dB]')

                # initialize confusion matrices (binary class)
                CM_trivial, CM_unitb, CM_unifb, CM_ab = (np.zeros([2, 2]) for i in range(4))
                CM_miss_trivial, CM_miss_unitb, CM_miss_unifb, CM_miss_ab = (np.zeros([2, 2]) for i in range(4))
                mismatchUpperBound, mismatchLowerBound = 0, 0  # initialize lower and upper bound

                # loop over iterations
                t0 = time.perf_counter()
                for iteration in range(simIterNum):
                    # - * - * - * - * - * - * Training * - * - * - * - * - * - * -
                    # train real adaboost model and calculate confidence levels for train and test sets
                    model = AdaBoostModel(database, database_path, train_size, numBaseClassifiers)
                    confidence_levels_train, confidence_levels_test = model.calc_confidence_levels(confidence_eps)

                    # scale noise variances to obtain desired average SNR
                    # curr_snr = np.sum(np.power(confidence_levels_train, 2)) \ (np.sum(sigma_profile) * len(model.train_y))
                    # curr_snr = np.mean((confidence_levels_train.T.dot(beta_uniform))**2) / np.sum(sigma_profile)
                    # curr_snr = powerLimit * np.mean(np.linalg.norm(confidence_levels_train, ord=2, axis=0)**2) / (np.linalg.norm(sigma_profile, ord=2))
                    curr_snr = powerLimit * np.mean(np.linalg.norm(confidence_levels_train, ord=2, axis=0) ** 2) / sigma_profile.sum()
                    sigma = sigma_profile * (curr_snr / snr)

                    # optimize coefficients (\alpha and \beta)
                    optimizer = CoefficientsOptimizer(confidence_levels_train, sigma, powerLimit, pNorm)

                    mismatch_unitb, alpha_unitb, opt_beta_unitb, stop_iter_unitb = optimizer.optimize_coefficients_power(
                        method='Alpha-UnitBeta', tol=tol_val, max_iter=max_iter, min_iter=min_iter)
                    opt_alpha_unitb = alpha_unitb[np.argmin(mismatch_unitb[0:stop_iter_unitb])]

                    mismatch_unifb, alpha_unifb, opt_beta_unifb, stop_iter_unifb = optimizer.optimize_coefficients_power(
                        method='Alpha-UniformBeta', tol=tol_val, max_iter=max_iter, min_iter=min_iter)
                    opt_alpha_unifb = alpha_unifb[np.argmin(mismatch_unifb[0:stop_iter_unifb])]

                    mismatch_ab, alpha_ab, beta_ab, stop_iter_ab = optimizer.optimize_coefficients_power(
                        method='Alpha-Beta', tol=tol_val, max_iter=max_iter, min_iter=min_iter)
                    opt_alpha_ab = alpha_ab[np.argmin(mismatch_ab[0:stop_iter_ab])]
                    opt_beta_ab = beta_ab[np.argmin(mismatch_ab[0:stop_iter_ab])]

                    # if iteration % 5 == 0:
                    #     fig_debug = plt.figure()
                    #     plt.plot(range(0, stop_iter_unitb, 1), mismatch_unitb[0:stop_iter_unitb], label='Mismatch, unit b', linestyle='-', color='blue')
                    #     plt.plot(np.argmin(mismatch_unitb[0:stop_iter_unitb]), np.min(mismatch_unitb[0:stop_iter_unitb]), marker='x', color='blue')
                    #
                    #     plt.plot(range(0, stop_iter_unifb, 1), mismatch_unifb[0:stop_iter_unifb], label='Mismatch, uniform b', linestyle='-', color='black')
                    #     plt.plot(np.argmin(mismatch_unifb[0:stop_iter_unifb]), np.min(mismatch_unifb[0:stop_iter_unifb]), marker='o', color='black')
                    #
                    #     plt.plot(range(0, stop_iter_ab, 1), mismatch_ab[0:stop_iter_ab], label='Mismatch, ab', linestyle='-', color='red')
                    #     plt.plot(np.argmin(mismatch_ab[0:stop_iter_ab]), np.min(mismatch_ab[0:stop_iter_ab]), marker='*', color='red')
                    #
                    #     plt.legend(loc="upper right", fontsize=12)
                    #     plt.xlabel("Optimization iteration", fontsize=14)
                    #     plt.ylabel("Mismatch probability", fontsize=14)
                    #     plt.grid()
                    #     plt.pause(0.05)
                    #     plt.close(fig_debug)

                    # - * - * - * - * - * - * Training end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Noisy channel * - * - * - * - * - * - * -
                    # generate additive noise matrix
                    noise_matrix = white_noise(sigma, numBaseClassifiers, len(model.test_y))
                    noisy_predictor = NoisyPredictor_(model, noise_matrix, tie_breaker)
                    # - * - * - * - * - * - * Noisy channel end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Inference * - * - * - * - * - * - * -
                    # predict according to the noisy data with:
                    predictTest_trivial = noisy_predictor.optimalPredict(np.ones([numBaseClassifiers, 1]), np.ones([numBaseClassifiers, 1]), confidence_levels_test)
                    predictTest_unitb = noisy_predictor.optimalPredict(opt_alpha_unitb, opt_beta_unitb, confidence_levels_test)
                    predictTest_unifb = noisy_predictor.optimalPredict(opt_alpha_unifb, opt_beta_unifb, confidence_levels_test)
                    predictTest_ab = noisy_predictor.optimalPredict(opt_alpha_ab, opt_beta_ab, confidence_levels_test)
                    predictTest_noiseless = model.classifier.predict(model.test_X)  # predicting on non-noisy data as reference
                    # - * - * - * - * - * - * Inference end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Evaluation * - * - * - * - * - * - * -
                    # get confusion matrix between noisy inference and true class (for error probability)
                    CM_trivial += sk.metrics.confusion_matrix(model.test_y, predictTest_trivial[0])
                    CM_unitb += sk.metrics.confusion_matrix(model.test_y, predictTest_unitb[0])
                    CM_unifb += sk.metrics.confusion_matrix(model.test_y, predictTest_unifb[0])
                    CM_ab += sk.metrics.confusion_matrix(model.test_y, predictTest_ab[0])

                    # get confusion matrix between noisy inference and noise-less inference (for mismatch probability)
                    CM_miss_trivial += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_trivial[0])
                    CM_miss_unitb += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_unitb[0])
                    CM_miss_unifb += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_unifb[0])
                    CM_miss_ab += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_ab[0])

                    # sum of upper and lower bounds over iterations
                    mismatchUpperBound += noisy_predictor.upperBound(confidence_levels_test, sigma, powerLimit, pNorm)
                    mismatchLowerBound += noisy_predictor.lowerBound(confidence_levels_test, sigma, powerLimit, pNorm)
                    # - * - * - * - * - * - * Evaluation end * - * - * - * - * - * - * -

                    if iteration % 10 == 0:
                        print('Iteration ' + str(iteration + 1) + ', Time elapsed: ' + str(time.perf_counter() - t0) + ' [s]')

                    # if calc_accuracy(CM_unifb) > calc_accuracy(CM_ab):
                    #     tttmp=1

                # - * - * - * - * - * - * Analyze results * - * - * - * - * - * - * -
                simulationResultsList["Database"].append(database)
                simulationResultsList["Time"].append(datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))
                simulationResultsList["SNR"].append(10 * np.log10(snr))
                simulationResultsList["Number of classifiers"].append(numBaseClassifiers)
                simulationResultsList["Power limit"].append(powerLimit)
                simulationResultsList["Lower bound (mismatch probability)"].append(mismatchLowerBound / simIterNum * 100)
                simulationResultsList["Upper bound (mismatch probability)"].append(mismatchUpperBound / simIterNum * 100)
                simulationResultsList["Mismatch probability (trivial)"].append((1-calc_accuracy(CM_miss_trivial))*100)
                simulationResultsList["Error probability (trivial)"].append((1-calc_accuracy(CM_trivial)) * 100)
                simulationResultsList["Mismatch probability (alpha, unit beta)"].append((1 - calc_accuracy(CM_miss_unitb)) * 100)
                simulationResultsList["Error probability (alpha, unit beta)"].append((1 - calc_accuracy(CM_unitb)) * 100)
                simulationResultsList["Mismatch probability (alpha, uniform beta)"].append((1 - calc_accuracy(CM_miss_unifb)) * 100)
                simulationResultsList["Error probability (alpha, uniform beta)"].append((1 - calc_accuracy(CM_unifb)) * 100)
                simulationResultsList["Mismatch probability (alpha, beta)"].append((1 - calc_accuracy(CM_miss_ab)) * 100)
                simulationResultsList["Error probability (alpha, beta)"].append((1 - calc_accuracy(CM_ab)) * 100)
                # add line to simulation csv file
                line_for_csv = [simulationResultsList.get(key)[-1] for key in simulationResultsList]
                write_csv_line(line_for_csv, pathToCsvFile)

                print('             \t\t' + "Mismatch probability" + "      \tError probability")
                print("Trivial:     \t\t\t" + "{:3.6f}".format(simulationResultsList["Mismatch probability (trivial)"][-1]) +             "       \t\t\t" + "{:3.6f}".format(simulationResultsList["Error probability (trivial)"][-1]))
                print("Unit-Beta:   \t\t\t" + "{:3.6f}".format(simulationResultsList["Mismatch probability (alpha, unit beta)"][-1]) +    "       \t\t\t" + "{:3.6f}".format(simulationResultsList["Error probability (alpha, unit beta)"][-1]))
                print("Uniform-Beta:\t\t\t" + "{:3.6f}".format(simulationResultsList["Mismatch probability (alpha, uniform beta)"][-1]) + "       \t\t\t" + "{:3.6f}".format(simulationResultsList["Error probability (alpha, uniform beta)"][-1]))
                print("Alpha-Beta:  \t\t\t" + "{:3.6f}".format(simulationResultsList["Mismatch probability (alpha, beta)"][-1]) +         "       \t\t\t" + "{:3.6f}".format(simulationResultsList["Error probability (alpha, beta)"][-1]))


                print("Simulation results saved to " + pathToCsvFile)
                # - * - * - * - * - * - * Analyze results end * - * - * - * - * - * - * -

            print("Completed simulating " + database)

            # plot error probability and mismatch probability
            fig = plt.figure()
            plt.plot(snr_array_db, simulationResultsList["Error probability (trivial)"], label='a, b: unit')
            plt.plot(snr_array_db, simulationResultsList["Error probability (alpha, unit beta)"], label='a: gd, b: unit')
            plt.plot(snr_array_db, simulationResultsList["Error probability (alpha, uniform beta)"], label='a: gd, b: uniform')
            plt.plot(snr_array_db, simulationResultsList["Error probability (alpha, beta)"], label='a, b: frank-wolfe')
            plt.legend(loc="upper right", fontsize=12)
            plt.title(str(database) + "\n T=" + str(numBaseClassifiers) + ", P=" + str(powerLimit) + ", l=" + str(pNorm))
            plt.xlabel("SNR [dB]", fontsize=14)
            plt.ylabel("Classification error probability [%]", fontsize=14)
            plt.grid()
            plt.pause(0.05)

            fig = plt.figure()
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (trivial)"], label='a,b: unit')
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha, unit beta)"], label='a: gd, b: unit')
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha, uniform beta)"], label='a: gd, b: uniform')
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha, beta)"], label='a, b: frank-wolfe')
            plt.plot(snr_array_db, simulationResultsList["Upper bound (mismatch probability)"], label='Upper bound')
            plt.plot(snr_array_db, simulationResultsList["Lower bound (mismatch probability)"], label='Lower bound')
            plt.legend(loc="upper right", fontsize=12)
            plt.title(str(database) + "\n T=" + str(numBaseClassifiers) + ", P=" + str(powerLimit) + ", l=" + str(pNorm))
            plt.xlabel("SNR [dB]", fontsize=14)
            plt.ylabel("Mismatch probability [%]", fontsize=14)
            plt.grid()
            plt.pause(0.05)
#
# if __name__ == "__main__":
#     main()
