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
            "Mismatch probability (alpha)", "Error probability (alpha)",
            "Mismatch probability (alpha_beta)", "Error probability (alpha_beta)"]

# simulation parameters
# databaseList = ["Heart Diseases Database", 'Parkinson Database', 'Wisconsin Breast Cancer Database']
# databasePathList = [dataFolderPath + "processed_cleveland_data_fixed.csv", dataFolderPath + "parkinsons_data_fixed.csv", dataFolderPath + ""]
databaseList = ["Heart Diseases Database"]
databasePathList = [dataFolderPath + "processed_cleveland_data_fixed.csv"]

# simulation and dataset parameters
powerLimitList = [1, 2] #[0.1, 1, 2] # power limitation, sum_t \beta_t^2 < PowerLimit
numBaseClassifiersList = [30] #[15, 30] # [15, 30, 45, 60]  # loop over number of base classifiers
pNorm = 2  # parameter p for p-norm in constraint for \beta
train_size = 0.8  # fraction of database used for training
simIterNum = 50  # number of iterations for each SNR

# constants and optimization parameters
confidence_eps = 1e-8  # tolerance value to preventing division by 0 in confidence-level calculation
tie_breaker = 1e-8  # addition to confidence level to avoid zero value
max_iter = 30000  # maximum number of gradient-descent iterations
learn_rate = 0.2  # gradient-descent step size
decay_rate = 0.2  # 0.9
eps = 0.00001

# init snr array
snr_array_db = np.arange(-10, 15, 4)
snr_array = 10 ** (snr_array_db / 10)

# loop over different numbers of base classifiers
for numBaseClassifiers in numBaseClassifiersList:
    print('Simulating ' + numBaseClassifiers.__str__() + ' base classifiers...')

    # loop over power limits
    for powerLimit in powerLimitList:

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
                "Mismatch probability (alpha)": [], "Error probability (alpha)": [],
                "Mismatch probability (alpha_beta)": [], "Error probability (alpha_beta)": []}

            # loop over SNR values
            for snr in snr_array:
                tol_val = 1e-5 / snr
                print('SNR: ' + str(10 * np.log10(snr)) + '[dB]')

                # initialize confusion matrices (binary class)
                CM_triv, CM_unconstrained, CM_constrained = (np.zeros([2, 2]) for i in range(3))
                CM_triv_miss, CM_unconstrained_miss, CM_constrained_miss = (np.zeros([2, 2]) for i in range(3))
                mismatchUpperBound, mismatchLowerBound = 0, 0  # initialize lower and upper bound

                # loop over iterations
                t0 = time.perf_counter()
                for iteration in range(simIterNum):
                    # - * - * - * - * - * - * Training * - * - * - * - * - * - * -
                    # train real adaboost model and calculate confidence levels for train and test sets
                    model = AdaBoostModel(database, database_path, train_size, numBaseClassifiers)
                    confidence_levels_train, confidence_levels_test = model.calc_confidence_levels(confidence_eps)

                    # normalize noise variances so that SNR is constant for all estimators
                    sigma = np.ones((numBaseClassifiers,))
                    X = np.sum(np.sum(np.multiply(confidence_levels_train, confidence_levels_train))) / (np.sum(sigma) * len(model.train_y)) #
                    sigma = sigma * (X / snr)

                    # optimize coefficients (\alpha and \beta)
                    optimizer = CoefficientsOptimizer(confidence_levels_train, sigma, powerLimit, pNorm)
                    alpha_gd_es_min = optimizer.optimize_unconstrained_coefficients(method='GD_ES_min',
                      tol_val=tol_val, max_iter=max_iter, learn_rate=learn_rate, decay_rate=decay_rate)
                    mismatch_prob, alpha, beta, stop_iter = optimizer.optimize_constrained_coefficients(
                        method='Frank-Wolfe')

                    fig = plt.figure()
                    plt.plot(range(0, stop_iter, 1), mismatch_prob[0:stop_iter], label='Mismatch Prob', linestyle='--',
                             marker='v', color='blue')
                    plt.xlabel("Frank-Wolfe iteration", fontsize=14)
                    plt.ylabel("Mismatch probability [%]", fontsize=14)
                    plt.pause(0.05)
                    plt.close(fig)
                    # - * - * - * - * - * - * Training end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Noisy channel * - * - * - * - * - * - * -
                    # generate additive noise matrix
                    noise_matrix = white_noise(sigma, numBaseClassifiers, len(model.test_y))
                    noisy_predictor = NoisyPredictor_(model, noise_matrix, tie_breaker)
                    # - * - * - * - * - * - * Noisy channel end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Inference * - * - * - * - * - * - * -
                    # predict according to the noisy data with:
                    predictTest_trivial = noisy_predictor.trivialPredict(confidence_levels_test)  # uniform alpha, uniform beta
                    predictTest_unconstrained = noisy_predictor.optimalPredict(alpha_gd_es_min.T, np.ones([numBaseClassifiers, 1]), confidence_levels_test)  # optimal alpha, uniform beta
                    predictTest_constrained = noisy_predictor.optimalPredict(alpha[stop_iter], beta[stop_iter], confidence_levels_test)  # optimal alpha, optimal beta
                    predictTest_noiseless = model.classifier.predict(model.test_X)  # predicting the non-noisy data as reference
                    # - * - * - * - * - * - * Inference end * - * - * - * - * - * - * -

                    # - * - * - * - * - * - * Evaluation * - * - * - * - * - * - * -
                    # get confusion matrix between noisy inference and true class (for error probability)
                    CM_triv += sk.metrics.confusion_matrix(model.test_y, predictTest_trivial)
                    CM_unconstrained += sk.metrics.confusion_matrix(model.test_y, predictTest_unconstrained)
                    CM_constrained += sk.metrics.confusion_matrix(model.test_y, predictTest_constrained[0])

                    # get confusion matrix between noisy inference and noise-less inference (for mismatch probability)
                    CM_triv_miss += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_trivial)
                    CM_unconstrained_miss += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_unconstrained)
                    CM_constrained_miss += sk.metrics.confusion_matrix(predictTest_noiseless, predictTest_constrained[0])

                    # sum of upper and lower bounds over iterations
                    mismatchUpperBound += noisy_predictor.upperBound(confidence_levels_test, sigma.T, powerLimit, pNorm)
                    mismatchLowerBound += noisy_predictor.lowerBound(confidence_levels_test, sigma.T, powerLimit, pNorm)
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
                simulationResultsList["Mismatch probability (trivial)"].append((1-calc_accuracy(CM_triv_miss))*100)
                simulationResultsList["Error probability (trivial)"].append((1-calc_accuracy(CM_triv)) * 100)
                simulationResultsList["Mismatch probability (alpha)"].append((1 - calc_accuracy(CM_unconstrained_miss)) * 100)
                simulationResultsList["Error probability (alpha)"].append((1 - calc_accuracy(CM_unconstrained)) * 100)
                simulationResultsList["Mismatch probability (alpha_beta)"].append((1 - calc_accuracy(CM_constrained_miss)) * 100)
                simulationResultsList["Error probability (alpha_beta)"].append((1 - calc_accuracy(CM_constrained)) * 100)
                # add line to simulation csv file
                line_for_csv = [simulationResultsList.get(key)[-1] for key in simulationResultsList]
                write_csv_line(line_for_csv, pathToCsvFile)

                print("Simulation results saved to " + pathToCsvFile)
                # - * - * - * - * - * - * Analyze results end * - * - * - * - * - * - * -

            print("Completed simulating " + database)

            # # plot the error probability and mismatch probability
            # fig = plt.figure()
            # plt.plot(snr_array_db, simulationResultsList["Error probability (trivial)"], label='Trivial')
            # plt.plot(snr_array_db, simulationResultsList["Error probability (Optimized GD)"], label='Optimal (GD)')
            # plt.legend(loc="upper right", fontsize=12)
            # # plt.title(str(database) + "\n Number of Estimators:" + str(numBaseClassifiers))
            # plt.xlabel("SNR [dB]", fontsize=14)
            # plt.ylabel("Classification error probability [%]", fontsize=14)
            # plt.grid()
            # # fig.tight_layout()

            fig = plt.figure()
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (trivial)"], label='Trivial')
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha)"], label='Unconstrained')
            plt.plot(snr_array_db, simulationResultsList["Mismatch probability (alpha_beta)"], label='Constrained')
            plt.plot(snr_array_db, simulationResultsList["Upper bound (mismatch probability)"], label='Upper bound')
            plt.plot(snr_array_db, simulationResultsList["Lower bound (mismatch probability)"], label='Lower bound')
            plt.legend(loc="upper right", fontsize=12)
            matplotlib.pyplot.title(str(database) + "\n T=" + str(numBaseClassifiers) + ", P=" + str(powerLimit) + ", l=" + str(pNorm))
            plt.xlabel("SNR [dB]", fontsize=14)
            plt.ylabel("Mismatch probability [%]", fontsize=14)
            plt.grid()
            # fig.tight_layout()
            # matplotlib.pyplot.savefig('histogram.pgf')
            # matplotlib.pyplot.show()
            plt.pause(0.05)
#
# if __name__ == "__main__":
#     main()
