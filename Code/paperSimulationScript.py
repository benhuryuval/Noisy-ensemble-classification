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
from auxiliaryFunctions import white_noise, write_csv_line, calc_accuracy, calc_constraint
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
            "Mismatch probability (alpha, beta)", "Error probability (alpha, beta)",
            "Mismatch probability (joint: alpha, beta)", "Error probability (joint: alpha, beta)"]

# simulation parameters
databaseList = ["Wisconsin Breast Cancer Database", "Heart Diseases Database", "Parkinson Database"]
databasePathList = [dataFolderPath + "", dataFolderPath + "processed_cleveland_data_fixed.csv", dataFolderPath + "parkinsons_data_fixed.csv"]
optimization_flag = True

# simulation and dataset parameters
numBaseClassifiersList = [30, 45, 60] #[15, 30] # [15, 30, 45, 60]  # loop over number of base classifiers
# powerLimitList = [np.sqrt(60), 60, 60**2]  # power limitation, sum_t \beta_t^2 < PowerLimit
pNorm = 2  # parameter p for p-norm in constraint for \beta
train_size = 0.8  # fraction of database used for training
simIterNum = 500  # number of iterations for each SNR

# model parameters
confidence_eps = 1e-8  # tolerance value to preventing division by 0 in confidence-level calculation
tie_breaker = 1e-8  # addition to confidence level to avoid zero value

# optimization parameters
max_iter = 250  # maximal number of gradient-descent iterations
min_iter = 15  # minimal number of gradient-descent iterations
learn_rate = 0.2  # gradient-descent step size
decay_rate = 0.2  # gradient-descent momentum
tol_val_baseline = 5e-4  # tolerance value to stop optimization upon convergence

# init snr array
snr_array_db = np.arange(-30, 16, 1)
# snr_array_db = np.array([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15])
# snr_array_db = np.array([0])
snr_array = 10 ** (snr_array_db / 10)

# loop over different numbers of base classifiers
for numBaseClassifiers in numBaseClassifiersList:
    print('Simulating ' + numBaseClassifiers.__str__() + ' base classifiers...')

    # set noise variance profile vector
    sigma_profile = np.ones([numBaseClassifiers, 1])
    # sigma_profile[1::2] *= 0.01

    # set power limits vector (G)
    # powerLimitList = [np.sqrt(numBaseClassifiers), numBaseClassifiers/2, numBaseClassifiers]  # power limitation, sum_t \beta_t^2 < PowerLimit
    powerLimitList = [numBaseClassifiers/100, np.sqrt(numBaseClassifiers), numBaseClassifiers]  # power limitation, sum_t \beta_t^2 < PowerLimit

    # loop over power limits
    for powerLimit in powerLimitList:

        # calculate uniform beta vector
        beta_uniform = powerLimit * np.ones([numBaseClassifiers, 1]) / calc_constraint(np.ones([numBaseClassifiers, 1]), pNorm)

        # loop over databases
        for database_idx, database in enumerate(databaseList):
            print('Database: ' + database)
            database_path = databasePathList[database_idx]

            if database == "Wisconsin Breast Cancer Database" and numBaseClassifiers != 60:
                continue
            if database == "Heart Diseases Database" and numBaseClassifiers != 45:
                continue
            if database == "Parkinson Database" and numBaseClassifiers != 30:
                continue

            # create csv file with header
            pathToCsvFile = resultsFolderPath + database + "_" + datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".csv"
            write_csv_line(csvHeader, pathToCsvFile)

            # initialize result arrays for simulation results
            simulationResultsList = {"Database": [], "Time": [], "SNR": [], "Number of classifiers": [], "Power limit": [],
                "Lower bound (mismatch probability)": [],           "Upper bound (mismatch probability)": [],
                "Mismatch probability (trivial)": [],               "Error probability (trivial)": [],
                "Mismatch probability (alpha, unit beta)": [],      "Error probability (alpha, unit beta)": [],
                "Mismatch probability (alpha, uniform beta)": [],   "Error probability (alpha, uniform beta)": [],
                "Mismatch probability (alpha, beta)": [],           "Error probability (alpha, beta)": [],
                "Mismatch probability (joint: alpha, beta)": [],    "Error probability (joint: alpha, beta)": []}

            # loop over SNR values
            for snr in snr_array:
                tol_val = tol_val_baseline / snr  # update tolerance per snr
                print('SNR: ' + str(10 * np.log10(snr)) + '[dB]')

                # initialize confusion matrices (binary class)
                CM_trivial, CM_unitb, CM_unifb, CM_ab, CM_ab_j = (np.zeros([2, 2]) for i in range(5))
                CM_miss_trivial, CM_miss_unitb, CM_miss_unifb, CM_miss_ab, CM_miss_ab_j = (np.zeros([2, 2]) for i in range(5))
                mismatchUpperBound, mismatchLowerBound = 0, 0  # initialize lower and upper bound

                # loop over iterations
                t0 = time.perf_counter()
                for iteration in range(simIterNum):
                    # - * - * - * - * - * - * Training * - * - * - * - * - * - * -
                    # train real adaboost model and calculate confidence levels for train and test sets
                    model = AdaBoostModel(database, database_path, train_size, numBaseClassifiers)
                    confidence_levels_train, confidence_levels_test = model.calc_confidence_levels(confidence_eps)

                    # scale noise variances to obtain desired average SNR
                    # curr_snr = powerLimit * np.mean(np.linalg.norm(confidence_levels_train, ord=2, axis=0) ** 2) / sigma_profile.sum()  # G \times SNR
                    curr_snr = np.mean(np.linalg.norm(confidence_levels_train, ord=2, axis=0) ** 2) / sigma_profile.sum()
                    sigma = sigma_profile * (curr_snr / snr)

                    # optimize coefficients (\alpha and \beta)
                    optimizer = CoefficientsOptimizer(confidence_levels_train, sigma, powerLimit, pNorm)

                    if optimization_flag:
                        mismatch_unitb, alpha_unitb, opt_beta_unitb, stop_iter_unitb = optimizer.optimize_coefficients_power(
                            method='Alpha-UnitBeta', tol=tol_val, max_iter=max_iter, min_iter=min_iter)
                        opt_alpha_unitb = alpha_unitb[np.argmin(mismatch_unitb[0:stop_iter_unitb])]
                    else:
                        opt_alpha_unitb, opt_beta_unitb = np.ones([numBaseClassifiers, 1]), np.ones([numBaseClassifiers, 1])

                    if optimization_flag:
                        mismatch_unifb, alpha_unifb, opt_beta_unifb, stop_iter_unifb = optimizer.optimize_coefficients_power(
                            method='Alpha-UniformBeta', tol=tol_val, max_iter=max_iter, min_iter=min_iter)
                        opt_alpha_unifb = alpha_unifb[np.argmin(mismatch_unifb[0:stop_iter_unifb])]
                    else:
                        opt_alpha_unifb, opt_beta_unifb = np.ones([numBaseClassifiers, 1]), np.ones([numBaseClassifiers, 1])

                    # if optimization_flag:
                    #     mismatch_ab, alpha_ab, beta_ab, stop_iter_ab = optimizer.optimize_coefficients_power(
                    #         method='Alpha-Beta-Alternate', tol=tol_val, max_iter=max_iter, min_iter=min_iter)
                    #     opt_alpha_ab = alpha_ab[np.argmin(mismatch_ab[0:stop_iter_ab])]
                    #     opt_beta_ab = beta_ab[np.argmin(mismatch_ab[0:stop_iter_ab])]
                    # else:
                    opt_alpha_ab, opt_beta_ab = np.ones([numBaseClassifiers, 1]), np.ones([numBaseClassifiers, 1])

                    if optimization_flag:
                        mismatch_ab_j, alpha_ab_j, beta_ab_j, stop_iter_ab_j = optimizer.optimize_coefficients_power(
                            method='Alpha-Beta-Joint', tol=tol_val, max_iter=max_iter, min_iter=min_iter)
                        opt_alpha_ab_j = alpha_ab_j[np.argmin(mismatch_ab_j[0:stop_iter_ab_j])]
                        opt_beta_ab_j = beta_ab_j[np.argmin(mismatch_ab_j[0:stop_iter_ab_j])]
                    else:
                        opt_alpha_ab_j, opt_beta_ab_j = np.ones([numBaseClassifiers, 1]), np.ones([numBaseClassifiers, 1])

                    # if iteration % 5 == 0 and optimization_flag:
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
                    #     plt.plot(range(0, stop_iter_ab_j, 1), mismatch_ab_j[0:stop_iter_ab_j], label='Mismatch, ab_j', linestyle='-', color='green')
                    #     plt.plot(np.argmin(mismatch_ab_j[0:stop_iter_ab_j]), np.min(mismatch_ab_j[0:stop_iter_ab_j]), marker='*', color='green')
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
                    predictTest_ab_j = noisy_predictor.optimalPredict(opt_alpha_ab_j, opt_beta_ab_j, confidence_levels_test)
                    predictTest_noiseless = model.classifier.predict(model.test_X)  # predicting on non-noisy data as reference
                    # - * - * - * - * - * - * Inference end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Evaluation * - * - * - * - * - * - * -
                    # get confusion matrix between noisy inference and true class (for error probability)
                    CM_trivial += sk.metrics.confusion_matrix(model.test_y, predictTest_trivial[0])
                    CM_unitb += sk.metrics.confusion_matrix(model.test_y, predictTest_unitb[0])
                    CM_unifb += sk.metrics.confusion_matrix(model.test_y, predictTest_unifb[0])
                    CM_ab += sk.metrics.confusion_matrix(model.test_y, predictTest_ab[0])
                    CM_ab_j += sk.metrics.confusion_matrix(model.test_y, predictTest_ab_j[0])

                    # get confusion matrix between noisy inference and noise-less inference (for mismatch probability)
                    CM_miss_trivial += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_trivial[0])
                    CM_miss_unitb += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_unitb[0])
                    CM_miss_unifb += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_unifb[0])
                    CM_miss_ab += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_ab[0])
                    CM_miss_ab_j += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_ab_j[0])

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
                simulationResultsList["Mismatch probability (trivial)"].append((1-calc_accuracy(CM_miss_trivial))*100)
                simulationResultsList["Error probability (trivial)"].append((1-calc_accuracy(CM_trivial)) * 100)
                simulationResultsList["Mismatch probability (alpha, unit beta)"].append((1 - calc_accuracy(CM_miss_unitb)) * 100)
                simulationResultsList["Error probability (alpha, unit beta)"].append((1 - calc_accuracy(CM_unitb)) * 100)
                simulationResultsList["Mismatch probability (alpha, uniform beta)"].append((1 - calc_accuracy(CM_miss_unifb)) * 100)
                simulationResultsList["Error probability (alpha, uniform beta)"].append((1 - calc_accuracy(CM_unifb)) * 100)
                simulationResultsList["Mismatch probability (alpha, beta)"].append((1 - calc_accuracy(CM_miss_ab)) * 100)
                simulationResultsList["Error probability (alpha, beta)"].append((1 - calc_accuracy(CM_ab)) * 100)
                simulationResultsList["Mismatch probability (joint: alpha, beta)"].append((1 - calc_accuracy(CM_miss_ab_j)) * 100)
                simulationResultsList["Error probability (joint: alpha, beta)"].append((1 - calc_accuracy(CM_ab_j)) * 100)
                # add line to simulation csv file
                line_for_csv = [simulationResultsList.get(key)[-1] for key in simulationResultsList]
                write_csv_line(line_for_csv, pathToCsvFile)

                print('             \t\t' + "Mismatch probability" + "      \tError probability")
                print("Trivial:     \t\t\t" + "{:3.6f}".format(simulationResultsList["Mismatch probability (trivial)"][-1]) +             "       \t\t\t" + "{:3.6f}".format(simulationResultsList["Error probability (trivial)"][-1]))
                print("Unit-Beta:   \t\t\t" + "{:3.6f}".format(simulationResultsList["Mismatch probability (alpha, unit beta)"][-1]) +    "       \t\t\t" + "{:3.6f}".format(simulationResultsList["Error probability (alpha, unit beta)"][-1]))
                print("Uniform-Beta:\t\t\t" + "{:3.6f}".format(simulationResultsList["Mismatch probability (alpha, uniform beta)"][-1]) + "       \t\t\t" + "{:3.6f}".format(simulationResultsList["Error probability (alpha, uniform beta)"][-1]))
                print("Alpha-Beta-Alternate:  \t\t\t" + "{:3.6f}".format(simulationResultsList["Mismatch probability (alpha, beta)"][-1]) +         "       \t\t\t" + "{:3.6f}".format(simulationResultsList["Error probability (alpha, beta)"][-1]))
                print("Alpha-Beta-Joint:  \t\t\t" + "{:3.6f}".format(simulationResultsList["Mismatch probability (joint: alpha, beta)"][-1]) +         "       \t\t\t" + "{:3.6f}".format(simulationResultsList["Error probability (joint: alpha, beta)"][-1]))


                print("Simulation results saved to " + pathToCsvFile)
                # - * - * - * - * - * - * Analyze results end * - * - * - * - * - * - * -

            print("Completed simulating " + database)

            # plot error probability and mismatch probability
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(str(database) + "\n T=" + str(numBaseClassifiers) + ", G=" + str(powerLimit) + ", l=" + str(pNorm))
            # fig = plt.figure()
            ax1.plot(snr_array_db, simulationResultsList["Error probability (trivial)"], label='a, b: unit')
            ax1.plot(snr_array_db, simulationResultsList["Error probability (alpha, unit beta)"], label='a: gd, b: unit')
            ax1.plot(snr_array_db, simulationResultsList["Error probability (alpha, uniform beta)"], label='a: gd, b: uniform')
            ax1.plot(snr_array_db, simulationResultsList["Error probability (alpha, beta)"], label='a, b: alternate')
            ax1.plot(snr_array_db, simulationResultsList["Error probability (joint: alpha, beta)"], label='a, b: joint')
            ax1.legend(loc="upper right", fontsize=12)
            ax1.set_xlabel("G SNR [dB]", fontsize=14)
            ax1.set_ylabel("Classification error probability [%]", fontsize=14)
            ax1.set_xlim([-25,15])
            ax1.set_ylim([0,50])
            ax1.grid()
            plt.pause(0.05)
            # plt.savefig(resultsFolderPath + 'perr_ab_' + database + '_T' + str(numBaseClassifiers) + '_P' + str(int(powerLimit)))

            # fig = plt.figure()
            ax2.plot(snr_array_db, simulationResultsList["Mismatch probability (trivial)"], label='a,b: unit')
            ax2.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha, unit beta)"], label='a: gd, b: unit')
            ax2.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha, uniform beta)"], label='a: gd, b: uniform')
            ax2.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha, beta)"], label='a, b: alternate')
            ax2.plot(snr_array_db, simulationResultsList["Mismatch probability (joint: alpha, beta)"], label='a, b: joint')
            ax2.plot(snr_array_db, simulationResultsList["Upper bound (mismatch probability)"], label='Upper bound')
            ax2.plot(snr_array_db, simulationResultsList["Lower bound (mismatch probability)"], label='Lower bound')
            ax2.legend(loc="upper right", fontsize=12)
            ax2.set_xlabel("G SNR [dB]", fontsize=14)
            ax2.set_ylabel("Mismatch probability [%]", fontsize=14)
            ax2.set_xlim([-25,15])
            ax2.set_ylim([0,50])
            ax2.grid()
            plt.pause(0.05)
            # plt.savefig(resultsFolderPath + 'pmis_ab_' + database + '_T' + str(numBaseClassifiers) + '_P' + str(int(powerLimit)))
            plt.savefig(resultsFolderPath + 'perrmis_ab_' + database + '_T' + str(numBaseClassifiers) + '_G' + str(int(powerLimit)))

    plt.close('all')
#
# if __name__ == "__main__":
#     main()
