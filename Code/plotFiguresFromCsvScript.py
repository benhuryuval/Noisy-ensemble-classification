import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.legend import Legend

# Figures for JSAC paper
cm = 1/2.54  # centimeters in inches

### - - - Figures from ISIT paper - - -
# Load simulation files
relativeRootPath = "C:\\Users\\Yuval\\Documents\\GitHub\\Noisy_Real_AdaBoost\\Paper\\"
resultsFolderPath = relativeRootPath + "results\\2022_01_03\\"

# filenameHeart30 = "Heart Diseases Database_03_01_2022-12_12_16"
# filenameHeart45 = "Heart Diseases Database_03_01_2022-13_52_56"
# filenameHeart60 = "Heart Diseases Database_03_01_2022-16_31_58"
filenameHeart30 = "Parkinson Database_03_01_2022-12_00_47"
filenameHeart45 = "Parkinson Database_03_01_2022-13_21_05"
filenameHeart60 = "Parkinson Database_03_01_2022-15_39_44"

filenameCancer30 = "Wisconsin Breast Cancer Database_03_01_2022-12_22_55"
filenameCancer45 = "Wisconsin Breast Cancer Database_03_01_2022-14_24_06"
filenameCancer60 = "Wisconsin Breast Cancer Database_03_01_2022-17_46_11"

# filenameParkinson30 = "Parkinson Database_03_01_2022-12_00_47"
# filenameParkinson45 = "Parkinson Database_03_01_2022-13_21_05"
# filenameParkinson60 = "Parkinson Database_03_01_2022-15_39_44"
filenameParkinson30 = "Heart Diseases Database_03_01_2022-12_12_16"
filenameParkinson45 = "Heart Diseases Database_03_01_2022-13_52_56"
filenameParkinson60 = "Heart Diseases Database_03_01_2022-16_31_58"

Heart30 = pd.read_csv(resultsFolderPath + filenameHeart30 + ".csv")
Heart45 = pd.read_csv(resultsFolderPath + filenameHeart45 + ".csv")
Heart60 = pd.read_csv(resultsFolderPath + filenameHeart60 + ".csv")

Cancer30 = pd.read_csv(resultsFolderPath + filenameCancer30 + ".csv")
Cancer45 = pd.read_csv(resultsFolderPath + filenameCancer45 + ".csv")
Cancer60 = pd.read_csv(resultsFolderPath + filenameCancer60 + ".csv")

Parkinson30 = pd.read_csv(resultsFolderPath + filenameParkinson30 + ".csv")
Parkinson45 = pd.read_csv(resultsFolderPath + filenameParkinson45 + ".csv")
Parkinson60 = pd.read_csv(resultsFolderPath + filenameParkinson60 + ".csv")

# Figure: Parkinson, T=30; Heart, T=45; Cancer, T=60
if False:
    with plt.style.context(['science', 'grid']):
        fig, axe = plt.subplots(figsize=(8.4,8.4))
        im1 ,= plt.plot(Parkinson30["SNR"], Parkinson30["Error probability (trivial)"], label="Parkinson, T=30, Unweighted", marker='', color='blue')
        im2 ,= plt.plot(Heart45["SNR"], Heart45["Error probability (trivial)"], label="Heart, T=45, Unweighted", linestyle='--', marker='', color='green')
        im3 ,= plt.plot(Cancer60["SNR"], Cancer60["Error probability (trivial)"], label="Cancer, T=60, Unweighted", linestyle=':', marker='', color='red')
        im4 ,= plt.plot(Parkinson30["SNR"], Parkinson30["Error probability (Vanilla GD)"], label="Parkinson, T=30, Alg. 3", marker='D', color='blue')
        im5 ,= plt.plot(Heart45["SNR"], Heart45["Error probability (Vanilla GD)"], label="Heart, T=45, Alg. 3", linestyle='--', marker='D', color='green')
        im6 ,= plt.plot(Cancer60["SNR"], Cancer60["Error probability (Vanilla GD)"], label="Cancer, T=60, Alg. 3", linestyle=':', marker='D', color='red')

        # specify the lines and labels of the first legend
        axe.legend([im1, im2, im3], [im1._label, im2._label, im3._label],
                  loc='lower left', framealpha=0.8, fontsize=16)
        # Create the second legend and add the artist manually.
        leg = Legend(axe, [im4, im5, im6], [im4._label, im5._label, im6._label],
                     loc='upper right', framealpha=0.8, fontsize=16)
        axe.add_artist(leg)

        # plt.legend(fontsize=16, ncol=1, loc='upper right', framealpha=0.8)
        plt.autoscale(tight=True)
        plt.xlabel(r'SNR [dB]', fontsize=20)
        plt.ylabel(r'Error probability [\%]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0,50])
        plt.show()

    # fig.savefig('error_prob.png', format='png', dpi=300)
    fig.savefig('error_prob_a.pdf')

# Figure: lower and upper bounds on mismatch
if True:
    with plt.style.context(['science', 'grid']):
        fig, axe = plt.subplots(figsize=(8.4, 8.4))
        im1 ,= plt.plot(Parkinson30["SNR"], Parkinson30["Upper bound (mismatch probability)"], label='Parkinson\'s disease (Train, Upper bound)', marker='v', color='blue')
        im2 ,= plt.plot(Parkinson30["SNR"], Parkinson30["Lower bound (mismatch probability)"], label='Parkinson\'s disease (Train, Lower bound)', marker='^', color='blue')
        im3 ,= plt.plot(Parkinson30["SNR"], Parkinson30["Mismatch probability (Vanilla GD)"], label='Parkinson\'s disease (Test, Optimized)', linestyle='-', marker='', color='blue')
        im4 ,= plt.plot(Heart45["SNR"], Heart45["Upper bound (mismatch probability)"], label='Heart disease (Train, Upper bound)', linestyle='--', marker='v', color='green')
        im5 ,= plt.plot(Heart45["SNR"], Heart45["Lower bound (mismatch probability)"], label='Heart disease (Train, Lower bound)', linestyle='--', marker='^', color='green')
        im6 ,= plt.plot(Heart45["SNR"], Heart45["Mismatch probability (Vanilla GD)"], label='Heart disease (Test, Optimized)', linestyle='--', marker='', color='green')
        im7 ,= plt.plot(Cancer60["SNR"], Cancer60["Upper bound (mismatch probability)"], label='Breast cancer (Train, Upper bound)', linestyle=':', marker='v', color='red')
        im8 ,= plt.plot(Cancer60["SNR"], Cancer60["Lower bound (mismatch probability)"], label='Breast cancer (Train, Lower bound)', linestyle=':', marker='^', color='red')
        im9 ,= plt.plot(Cancer60["SNR"], Cancer60["Mismatch probability (Vanilla GD)"], label='Breast cancer (Test, Optimized)', linestyle=':', marker='', color='red')

        # create blank rectangle
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        # Create organized list containing all handles for table. Extra represent empty space
        legend_handle = [extra, extra, extra, extra, extra, im1, im2, im3, extra, im4, im5, im6, extra, im7, im8, im9]
        # Define the labels
        label_row_1 = ["", r"Upper bnd (Train)", r"Lower bnd (Train)", r"Alg. 3 (Test)"]
        label_j_1 = [r"Parkinson"]
        label_j_2 = [r"Heart disease"]
        label_j_3 = [r"Breast cancer"]
        label_empty = [""]
        # organize labels for table construction
        legend_labels = np.concatenate(
            [label_row_1, label_j_1, label_empty * 3, label_j_2, label_empty * 3, label_j_3, label_empty * 3])
        # Create legend
        axe.legend(legend_handle, legend_labels, fontsize=14,
                  loc='upper right', ncol=4, shadow=False, handletextpad=-2, framealpha=0.8)

        plt.xlabel(r'SNR [dB]', fontsize=20)
        plt.ylabel(r'Mismatch probability [\%]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0, 50])
        plt.xlim([-25, 10])
        plt.show()

    # fig.savefig('mismatch_bounds.png', format='png', dpi=300)
    fig.savefig('mismatch_bounds_a.pdf')

# Figure: SNR gain
if True:
    def getSnrGain(Data, nPoints):
        x = Data["SNR"].values
        # y_interp = np.linspace(0, 50, num=51)

        y1 = Data["Error probability (Vanilla GD)"].values
        y_interp = np.linspace(np.min(y1), np.max(y1), num=nPoints)
        ind = np.argsort(y1)
        x1_interp = np.interp(y_interp, y1[ind], x[ind])
        # fig = plt.figure()
        # plt.plot(y1, x, linestyle='', marker='o', color='blue')
        # plt.plot(y_interp, x1_interp, linestyle='', marker='x', color='blue')

        y2 = Data["Error probability (trivial)"].values
        # y_interp = np.linspace(np.min(y1), np.max(y1), num=51)
        ind = np.argsort(y2)
        x2_interp = np.interp(y_interp, y2[ind], x[ind])
        # fig = plt.figure()
        # plt.plot(y2, x, linestyle='', marker='o', color='blue')
        # plt.plot(y_interp, x2_interp, linestyle='', marker='x', color='blue')

        return y_interp, x2_interp-x1_interp

    # from results.plotFiguresFromCsvScript import getSnrGain
    nPoints = 20
    x, y = np.zeros([nPoints,]), np.zeros([nPoints,])

    x11, y11 = getSnrGain(Parkinson30, nPoints)
    x12, y12 = getSnrGain(Parkinson45, nPoints)
    x13, y13 = getSnrGain(Parkinson60, nPoints)

    x21, y21 = getSnrGain(Heart30,nPoints)
    x22, y22 = getSnrGain(Heart45,nPoints)
    x23, y23 = getSnrGain(Heart60, nPoints)

    x31, y31 = getSnrGain(Cancer30, nPoints)
    x32, y32 = getSnrGain(Cancer45, nPoints)
    x33, y33 = getSnrGain(Cancer60, nPoints)

    with plt.style.context(['science', 'grid']):
        fig, axe = plt.subplots(figsize=(8.4, 8.4))

        plt.plot(x11, y11, linestyle='', marker='o', color='blue', markersize=3, label='Parkinson, T=30')
        plt.plot(x12, y12, linestyle='', marker='o', color='blue', markersize=6, label='Parkinson, T=45')
        plt.plot(x13, y13, linestyle='', marker='o', color='blue', markersize=9, label='Parkinson, T=60')

        plt.plot(x21, y21, linestyle='', marker='o', color='green', markersize=3, label='Heart, T=30')
        plt.plot(x22, y22, linestyle='', marker='o', color='green', markersize=6,  label='Heart, T=45')
        plt.plot(x23, y23, linestyle='', marker='o', color='green', markersize=9, label='Heart, T=60')

        plt.plot(x31, y31, linestyle='', marker='o', color='red', markersize=3, label='Cancer, T=30')
        plt.plot(x32, y32, linestyle='', marker='o', color='red', markersize=6, label='Cancer, T=45')
        plt.plot(x33, y33, linestyle='', marker='o', color='red', markersize=9, label='Cancer, T=60')

        plt.legend(fontsize=15, ncol=3, loc='upper center', framealpha=0.8)
        plt.xlabel(r'Error probability [\%]', fontsize=15)
        plt.ylabel(r'SNR gain [dB]', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0,12])
        plt.show()

    # fig.savefig('snr_gain.png', format='png', dpi=300)
    fig.savefig('snr_gain_a.pdf')

# - - - New figures: gain and coefficients - - -
# Load simulation files
relativeRootPath = "C:\\Users\\Yuval\\Google Drive\\PhD\\Boosting\\Power-constrained weighted combining\\Noisy-ensemble-classification\\"
resultsFolderPath = relativeRootPath + "Results\\19_03_2022\\"

# Parkinson
filenameParkinson_T1_G1 = "Parkinson Database_16_03_2022-23_42_54"
filenameParkinson_T1_G2 = "Parkinson Database_17_03_2022-02_04_26"
filenameParkinson_T1_G3 = "Parkinson Database_17_03_2022-04_26_24"

# filenameParkinson_T2_G1 = "Parkinson Database_18_02_2022-23_56_00"
# filenameParkinson_T2_G2 = "Parkinson Database_19_02_2022-00_54_37"
# filenameParkinson_T2_G3 = "Parkinson Database_19_02_2022-01_53_22"
#
# filenameParkinson_T3_G1 = "Parkinson Database_19_02_2022-02_55_02"
# filenameParkinson_T3_G2 = "Parkinson Database_19_02_2022-04_03_33"
# filenameParkinson_T3_G3 = "Parkinson Database_19_02_2022-05_09_28"

# Heart
# filenameHeart_T1_G1 = "Heart Diseases Database_18_02_2022-21_10_01"
# filenameHeart_T1_G2 = "Heart Diseases Database_18_02_2022-22_01_25"
# filenameHeart_T1_G3 = "Heart Diseases Database_18_02_2022-22_48_32"

filenameHeart_T2_G1 = "Heart Diseases Database_17_03_2022-06_48_38"
filenameHeart_T2_G2 = "Heart Diseases Database_17_03_2022-10_21_51"
filenameHeart_T2_G3 = "Heart Diseases Database_17_03_2022-14_08_22"

# filenameHeart_T3_G1 = "Heart Diseases Database_19_02_2022-02_37_09"
# filenameHeart_T3_G2 = "Heart Diseases Database_19_02_2022-03_44_42"
# filenameHeart_T3_G3 = "Heart Diseases Database_19_02_2022-04_52_11"

# Cancer
# filenameCancer_T1_G1 = "Wisconsin Breast Cancer Database_18_02_2022-20_44_35"
# filenameCancer_T1_G2 = "Wisconsin Breast Cancer Database_18_02_2022-21_35_07"
# filenameCancer_T1_G3 = "Wisconsin Breast Cancer Database_18_02_2022-22_25_36"
#
# filenameCancer_T2_G1 = "Wisconsin Breast Cancer Database_18_02_2022-23_12_34"
# filenameCancer_T2_G2 = "Wisconsin Breast Cancer Database_19_02_2022-00_08_59"
# filenameCancer_T2_G3 = "Wisconsin Breast Cancer Database_19_02_2022-01_08_28"

filenameCancer_T3_G1 = "Wisconsin Breast Cancer Database_17_03_2022-18_15_19"
filenameCancer_T3_G2 = "Wisconsin Breast Cancer Database_18_03_2022-02_35_53"
filenameCancer_T3_G3 = "Wisconsin Breast Cancer Database_18_03_2022-11_17_14"

# Load files
# Parkinson
Parkinson_T1_G1 = pd.read_csv(resultsFolderPath + filenameParkinson_T1_G1 + ".csv")
Parkinson_T1_G2 = pd.read_csv(resultsFolderPath + filenameParkinson_T1_G2 + ".csv")
Parkinson_T1_G3 = pd.read_csv(resultsFolderPath + filenameParkinson_T1_G3 + ".csv")
# Parkinson_T2_G1 = pd.read_csv(resultsFolderPath + filenameParkinson_T2_G1 + ".csv")
# Parkinson_T2_G2 = pd.read_csv(resultsFolderPath + filenameParkinson_T2_G2 + ".csv")
# Parkinson_T2_G3 = pd.read_csv(resultsFolderPath + filenameParkinson_T2_G3 + ".csv")
# Parkinson_T3_G1 = pd.read_csv(resultsFolderPath + filenameParkinson_T3_G1 + ".csv")
# Parkinson_T3_G2 = pd.read_csv(resultsFolderPath + filenameParkinson_T3_G2 + ".csv")
# Parkinson_T3_G3 = pd.read_csv(resultsFolderPath + filenameParkinson_T3_G3 + ".csv")

# Heart
# Heart_T1_G1 = pd.read_csv(resultsFolderPath + filenameHeart_T1_G1 + ".csv")
# Heart_T1_G2 = pd.read_csv(resultsFolderPath + filenameHeart_T1_G2 + ".csv")
# Heart_T1_G3 = pd.read_csv(resultsFolderPath + filenameHeart_T1_G3 + ".csv")
Heart_T2_G1 = pd.read_csv(resultsFolderPath + filenameHeart_T2_G1 + ".csv")
Heart_T2_G2 = pd.read_csv(resultsFolderPath + filenameHeart_T2_G2 + ".csv")
Heart_T2_G3 = pd.read_csv(resultsFolderPath + filenameHeart_T2_G3 + ".csv")
# Heart_T3_G1 = pd.read_csv(resultsFolderPath + filenameHeart_T3_G1 + ".csv")
# Heart_T3_G2 = pd.read_csv(resultsFolderPath + filenameHeart_T3_G2 + ".csv")
# Heart_T3_G3 = pd.read_csv(resultsFolderPath + filenameHeart_T1_G3 + ".csv")
# Cancer
# Cancer_T1_G1 = pd.read_csv(resultsFolderPath + filenameCancer_T1_G1 + ".csv")
# Cancer_T1_G2 = pd.read_csv(resultsFolderPath + filenameCancer_T1_G2 + ".csv")
# Cancer_T1_G3 = pd.read_csv(resultsFolderPath + filenameCancer_T1_G3 + ".csv")
# Cancer_T2_G1 = pd.read_csv(resultsFolderPath + filenameCancer_T2_G1 + ".csv")
# Cancer_T2_G2 = pd.read_csv(resultsFolderPath + filenameCancer_T2_G2 + ".csv")
# Cancer_T2_G3 = pd.read_csv(resultsFolderPath + filenameCancer_T2_G3 + ".csv")
Cancer_T3_G1 = pd.read_csv(resultsFolderPath + filenameCancer_T3_G1 + ".csv")
Cancer_T3_G2 = pd.read_csv(resultsFolderPath + filenameCancer_T3_G2 + ".csv")
Cancer_T3_G3 = pd.read_csv(resultsFolderPath + filenameCancer_T3_G3 + ".csv")

# - - - Recreate ISIT paper figures from new simulation - - -
# Figure: Error probability only alpha, unit gain (Parkinson, T=30; Heart, T=45; Cancer, T=60)
if True:
    with plt.style.context(['science', 'grid']):
        fig, axe = plt.subplots(figsize=(8.4,8.4))
        im1 ,= plt.plot(Parkinson_T1_G1["SNR"], Parkinson_T1_G1["Error probability (trivial)"], label="Parkinson, T=30, Unweighted", marker='', color='blue')
        im2 ,= plt.plot(Heart_T2_G2["SNR"], Heart_T2_G2["Error probability (trivial)"], label="Heart, T=45, Unweighted", linestyle='--', marker='', color='green')
        im3 ,= plt.plot(Cancer_T3_G3["SNR"], Cancer_T3_G3["Error probability (trivial)"], label="Cancer, T=60, Unweighted", linestyle=':', marker='', color='red')
        im4 ,= plt.plot(Parkinson_T1_G1["SNR"], Parkinson_T1_G1["Error probability (alpha, unit beta)"], label="Parkinson, T=30, Alg. 3", marker='D', color='blue')
        im5 ,= plt.plot(Heart_T2_G2["SNR"], Heart_T2_G2["Error probability (alpha, unit beta)"], label="Heart, T=45, Alg. 3", linestyle='--', marker='D', color='green')
        im6 ,= plt.plot(Cancer_T3_G3["SNR"], Cancer_T3_G3["Error probability (alpha, unit beta)"], label="Cancer, T=60, Alg. 3", linestyle=':', marker='D', color='red')

        # specify the lines and labels of the first legend
        axe.legend([im1, im2, im3], [im1._label, im2._label, im3._label],
                  loc='lower left', framealpha=0.8, fontsize=16)
        # Create the second legend and add the artist manually.
        leg = Legend(axe, [im4, im5, im6], [im4._label, im5._label, im6._label],
                     loc='upper right', framealpha=0.8, fontsize=16)
        axe.add_artist(leg)

        # plt.legend(fontsize=16, ncol=1, loc='upper right', framealpha=0.8)
        plt.autoscale(tight=True)
        plt.xlabel(r'SNR [dB]', fontsize=20)
        plt.ylabel(r'Error probability [\%]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0,50])
        plt.show()

    # fig.savefig('error_prob.png', format='png', dpi=300)
    fig.savefig('error_prob_a.pdf')

# Figure: lower and upper bounds on mismatch probability only alpha
#       cannot recreate from new simulation...

# Figure: SNR gain only alpha
#       cannot recreate from new simulation...
if False:
    def getSnrGain(Data, nPoints):
        x = Data["SNR"].values
        # y_interp = np.linspace(0, 50, num=51)

        y1 = Data["Error probability (alpha, unit beta)"].values
        y_interp = np.linspace(np.min(y1), np.max(y1), num=nPoints)
        ind = np.argsort(y1)
        x1_interp = np.interp(y_interp, y1[ind], x[ind])
        # fig = plt.figure()
        # plt.plot(y1, x, linestyle='', marker='o', color='blue')
        # plt.plot(y_interp, x1_interp, linestyle='', marker='x', color='blue')

        y2 = Data["Error probability (trivial)"].values
        # y_interp = np.linspace(np.min(y1), np.max(y1), num=51)
        ind = np.argsort(y2)
        x2_interp = np.interp(y_interp, y2[ind], x[ind])
        # fig = plt.figure()
        # plt.plot(y2, x, linestyle='', marker='o', color='blue')
        # plt.plot(y_interp, x2_interp, linestyle='', marker='x', color='blue')

        return y_interp, x2_interp-x1_interp

    # from results.plotFiguresFromCsvScript import getSnrGain
    nPoints = 20
    x, y = np.zeros([nPoints,]), np.zeros([nPoints,])

    x11, y11 = getSnrGain(Parkinson_T1_G1, nPoints)
    x12, y12 = getSnrGain(Parkinson_T2_G2, nPoints)
    x13, y13 = getSnrGain(Parkinson_T3_G3, nPoints)

    x21, y21 = getSnrGain(Heart_T1_G1,nPoints)
    x22, y22 = getSnrGain(Heart_T2_G2,nPoints)
    x23, y23 = getSnrGain(Heart_T3_G3, nPoints)

    x31, y31 = getSnrGain(Cancer_T1_G1, nPoints)
    x32, y32 = getSnrGain(Cancer_T2_G2, nPoints)
    x33, y33 = getSnrGain(Cancer_T3_G3, nPoints)

    with plt.style.context(['science', 'grid']):
        fig, axe = plt.subplots(figsize=(8.4, 8.4))

        plt.plot(x11, y11, linestyle='', marker='o', color='blue', markersize=3, label='Parkinson, T=30')
        plt.plot(x12, y12, linestyle='', marker='o', color='blue', markersize=6, label='Parkinson, T=45')
        plt.plot(x13, y13, linestyle='', marker='o', color='blue', markersize=9, label='Parkinson, T=60')

        plt.plot(x21, y21, linestyle='', marker='o', color='green', markersize=3, label='Heart, T=30')
        plt.plot(x22, y22, linestyle='', marker='o', color='green', markersize=6,  label='Heart, T=45')
        plt.plot(x23, y23, linestyle='', marker='o', color='green', markersize=9, label='Heart, T=60')

        plt.plot(x31, y31, linestyle='', marker='o', color='red', markersize=3, label='Cancer, T=30')
        plt.plot(x32, y32, linestyle='', marker='o', color='red', markersize=6, label='Cancer, T=45')
        plt.plot(x33, y33, linestyle='', marker='o', color='red', markersize=9, label='Cancer, T=60')

        plt.legend(fontsize=10, ncol=3, loc='upper center', framealpha=0.8)
        plt.xlabel(r'Error probability [\%]', fontsize=15)
        plt.ylabel(r'SNR gain [dB]', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0,12])
        plt.show()

    # fig.savefig('snr_gain.png', format='png', dpi=300)
    fig.savefig('snr_gain_a.pdf')

# - - - New figures for JSAC - - -
# Preparations for alpha, gain figures
G1_str, G2_str, G3_str = "T/100", "\sqrt{T}", "T"
T1, T2, T3 = 30, 45, 60

# Figure: Error probability of alpha, gain
if True:
    with plt.style.context(['science', 'grid']):
        fig, axe = plt.subplots(figsize=(8.4, 8.4))
        im1 ,= plt.plot(Parkinson_T1_G1["SNR"], Parkinson_T1_G1["Error probability (joint: alpha, beta)"], label="Parkinson, $G="+G1_str+"$, Alg. 2", linestyle=':', marker='o', color='blue', markersize=3)
        im2 ,= plt.plot(Parkinson_T1_G2["SNR"], Parkinson_T1_G2["Error probability (joint: alpha, beta)"], label="Parkinson, $G="+G2_str+"$, Alg. 2", linestyle='--', marker='o', color='blue', markersize=6)
        im3 ,= plt.plot(Parkinson_T1_G3["SNR"], Parkinson_T1_G3["Error probability (joint: alpha, beta)"], label="Parkinson, $G="+G3_str+"$, Alg. 2", linestyle='-', marker='o', color='blue', markersize=9)
        im4 ,= plt.plot(Parkinson_T1_G3["SNR"], Parkinson_T1_G3["Error probability (alpha, unit beta)"], label="Parkinson, Alg. 3", linestyle='-', marker='x', color='blue', markersize=9)
        # im4 ,= plt.plot(Parkinson30["SNR"], Parkinson30["Error probability (Vanilla GD)"], label="Parkinson, , Alg. 3", linestyle='-', marker='x', color='blue', markersize=9)
        axe.legend(fontsize=14, loc='lower left', ncol=1, shadow=False, framealpha=0.8)
        plt.xlabel(r'SNR [dB]', fontsize=20)
        plt.ylabel(r'Error probability [\%]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0, 50])
        plt.xlim([-25, 10])
        plt.show()
        # fig.savefig('error_prob_a_g_parkinson.png', format='png', dpi=300)
        fig.savefig('error_prob_a_g_parkinson.pdf', format='pdf')

        fig, axe = plt.subplots(figsize=(8.4, 8.4))
        im1 ,= plt.plot(Heart_T2_G1["SNR"], Heart_T2_G1["Error probability (joint: alpha, beta)"], label="Heart, $G="+G1_str+"$, Alg. 2", linestyle=':', marker='o', color='green', markersize=3)
        im2 ,= plt.plot(Heart_T2_G2["SNR"], Heart_T2_G2["Error probability (joint: alpha, beta)"], label="Heart, $G="+G2_str+"$, Alg. 2", linestyle='--', marker='o', color='green', markersize=6)
        im3 ,= plt.plot(Heart_T2_G3["SNR"], Heart_T2_G3["Error probability (joint: alpha, beta)"], label="Heart, $G="+G3_str+"$, Alg. 2", linestyle='-', marker='o', color='green', markersize=9)
        im4 ,= plt.plot(Heart_T2_G3["SNR"], Heart_T2_G3["Error probability (alpha, unit beta)"], label="Heart, Alg. 3", linestyle='-', marker='x', color='green', markersize=9)
        # im4 ,= plt.plot(Heart45["SNR"], Heart45["Error probability (Optimized GD)"], label="Heart, Alg. 3", linestyle='-', marker='x', color='green', markersize=9)
        axe.legend(fontsize=14, loc='lower left', ncol=1, shadow=False, framealpha=0.8)
        plt.xlabel(r'SNR [dB]', fontsize=20)
        plt.ylabel(r'Mismatch probability [\%]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0, 50])
        plt.xlim([-25, 10])
        plt.show()
        # fig.savefig('error_prob_a_g_heart.png', format='png', dpi=300)
        fig.savefig('error_prob_a_g_heart.pdf', format='pdf')

        fig, axe = plt.subplots(figsize=(8.4, 8.4))
        im1 ,= plt.plot(Cancer_T3_G1["SNR"], Cancer_T3_G1["Error probability (joint: alpha, beta)"], label="Cancer, $G="+G1_str+"$, Alg. 2", linestyle='--', marker='o', color='red', markersize=3)
        im2 ,= plt.plot(Cancer_T3_G2["SNR"], Cancer_T3_G2["Error probability (joint: alpha, beta)"], label="Cancer, $G="+G2_str+"$, Alg. 2", linestyle='--', marker='o', color='red', markersize=6)
        im3 ,= plt.plot(Cancer_T3_G3["SNR"], Cancer_T3_G3["Error probability (joint: alpha, beta)"], label="Cancer, $G="+G3_str+"$, Alg. 2", linestyle='--', marker='o', color='red', markersize=9)
        im4 ,= plt.plot(Cancer_T3_G3["SNR"], Cancer_T3_G3["Error probability (alpha, unit beta)"], label="Cancer, Alg. 3", linestyle='-', marker='x', color='red', markersize=9)
        # im4 ,= plt.plot(Cancer60["SNR"], Cancer60["Error probability (Vanilla GD)"], label="Cancer, Alg. 3", linestyle='-', marker='x', color='red', markersize=9)
        axe.legend(fontsize=14, loc='lower left', ncol=1, shadow=False, framealpha=0.8)
        plt.xlabel(r'SNR [dB]', fontsize=20)
        plt.ylabel(r'Mismatch probability [\%]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0, 50])
        plt.xlim([-25, 10])
        plt.show()
        # fig.savefig('error_prob_a_g_cancer.png', format='png', dpi=300)
        fig.savefig('error_prob_a_g_cancer.pdf', format='pdf')

# Figure: SNR gain of alpha, gain
if True:
    def getSnrGain(Data, nPoints):
        x = Data["SNR"].values
        # y_interp = np.linspace(0, 50, num=51)

        y1 = Data["Error probability (joint: alpha, beta)"].values
        y_interp = np.linspace(np.min(y1), np.max(y1), num=nPoints)
        ind = np.argsort(y1)
        x1_interp = np.interp(y_interp, y1[ind], x[ind])
        # fig = plt.figure()
        # plt.plot(y1, x, linestyle='', marker='o', color='blue')
        # plt.plot(y_interp, x1_interp, linestyle='', marker='x', color='blue')

        y2 = Data["Error probability (alpha, uniform beta)"].values
        # y_interp = np.linspace(np.min(y1), np.max(y1), num=51)
        ind = np.argsort(y2)
        x2_interp = np.interp(y_interp, y2[ind], x[ind])
        # fig = plt.figure()
        # plt.plot(y2, x, linestyle='', marker='o', color='blue')
        # plt.plot(y_interp, x2_interp, linestyle='', marker='x', color='blue')

        return y_interp, x2_interp-x1_interp

    nPoints = 20
    x, y = np.zeros([nPoints,]), np.zeros([nPoints,])

    x11, y11 = getSnrGain(Parkinson_T1_G1, nPoints)
    x12, y12 = getSnrGain(Parkinson_T1_G2, nPoints)
    x13, y13 = getSnrGain(Parkinson_T1_G3, nPoints)

    x21, y21 = getSnrGain(Heart_T2_G1, nPoints)
    x22, y22 = getSnrGain(Heart_T2_G2, nPoints)
    x23, y23 = getSnrGain(Heart_T2_G3, nPoints)

    x31, y31 = getSnrGain(Cancer_T3_G1, nPoints)
    x32, y32 = getSnrGain(Cancer_T3_G2, nPoints)
    x33, y33 = getSnrGain(Cancer_T3_G3, nPoints)

    with plt.style.context(['science', 'grid']):
        fig, axe = plt.subplots(figsize=(8.4, 8.4))

        plt.plot(x11, y11, linestyle='', marker='o', color='blue', markersize=3, label='Parkinson, T=30, $G='+G1_str+"$")
        plt.plot(x12, y12, linestyle='', marker='o', color='blue', markersize=6, label='Parkinson, T=30, $G='+G2_str+"$")
        plt.plot(x13, y13, linestyle='', marker='o', color='blue', markersize=9, label='Parkinson, T=30, $G='+G3_str+"$")

        plt.plot(x21, y21, linestyle='', marker='o', color='green', markersize=3, label='Heart, T=45, $G='+G1_str+"$")
        plt.plot(x22, y22, linestyle='', marker='o', color='green', markersize=6,  label='Heart, T=45, $G='+G2_str+"$")
        plt.plot(x23, y23, linestyle='', marker='o', color='green', markersize=9, label='Heart, T=45, $G='+G3_str+"$")

        plt.plot(x31, y31, linestyle='', marker='o', color='red', markersize=3, label='Cancer, T=60, $G='+G1_str+"$")
        plt.plot(x32, y32, linestyle='', marker='o', color='red', markersize=6, label='Cancer, T=60, $G='+G2_str+"$")
        plt.plot(x33, y33, linestyle='', marker='o', color='red', markersize=9, label='Cancer, T=60, $G='+G3_str+"$")

        plt.legend(fontsize=9.5, ncol=3, loc='lower center', framealpha=0.8)
        plt.xlabel(r'Error probability [\%]', fontsize=15)
        plt.ylabel(r'SNR gain [dB]', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0, 15])
        # plt.xlim([-25, 10])
        plt.show()

    # fig.savefig('snr_gain.png', format='png', dpi=300)
    fig.savefig('snr_gain_a_g.pdf', format='pdf')

