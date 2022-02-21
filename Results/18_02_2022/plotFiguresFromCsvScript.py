import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.legend import Legend

cm = 1/2.54  # centimeters in inches

# load simulation files
relativeRootPath = "C:\\Users\\Yuval\\Google Drive\\PhD\\Boosting\\Power-constrained weighted combining\\Noisy-ensemble-classification\\"
resultsFolderPath = relativeRootPath + "Results\\18_02_2022\\"

# Parkinson
filenameParkinson_T1_G1 = "Parkinson Database_18_02_2022-12_06_56"
filenameParkinson_T1_G2 = "Parkinson Database_18_02_2022-11_27_16"
filenameParkinson_T1_G3 = "Parkinson Database_18_02_2022-10_47_52"

filenameParkinson_T2_G1 = "Parkinson Database_18_02_2022-14_46_50"
filenameParkinson_T2_G2 = "Parkinson Database_18_02_2022-13_42_38"
filenameParkinson_T2_G3 = "Parkinson Database_18_02_2022-12_51_51"

filenameParkinson_T3_G1 = "Parkinson Database_18_02_2022-18_28_45"
filenameParkinson_T3_G2 = "Parkinson Database_18_02_2022-17_15_17"
filenameParkinson_T3_G3 = "Parkinson Database_18_02_2022-16_03_11"

# Heart
filenameHeart_T1_G1 = "Heart Diseases Database_18_02_2022-11_57_20"
filenameHeart_T1_G2 = "Heart Diseases Database_18_02_2022-11_16_34"
filenameHeart_T1_G3 = "Heart Diseases Database_18_02_2022-10_39_26"

filenameHeart_T2_G1 = "Heart Diseases Database_18_02_2022-14_24_45"
filenameHeart_T2_G2 = "Heart Diseases Database_18_02_2022-13_26_07"
filenameHeart_T2_G3 = "Heart Diseases Database_18_02_2022-12_38_25"

filenameHeart_T3_G1 = "Heart Diseases Database_18_02_2022-18_08_54"
filenameHeart_T3_G2 = "Heart Diseases Database_18_02_2022-16_55_12"
filenameHeart_T3_G3 = "Heart Diseases Database_18_02_2022-15_41_48"

# Cancer
filenameCancer_T1_G1 = "Wisconsin Breast Cancer Database_18_02_2022-11_37_56"
filenameCancer_T1_G2 = "Wisconsin Breast Cancer Database_18_02_2022-10_56_31"
filenameCancer_T1_G3 = "Wisconsin Breast Cancer Database_18_02_2022-10_21_46"

filenameCancer_T2_G1 = "Wisconsin Breast Cancer Database_18_02_2022-13_54_26"
filenameCancer_T2_G2 = "Wisconsin Breast Cancer Database_18_02_2022-13_01_32"
filenameCancer_T2_G3 = "Wisconsin Breast Cancer Database_18_02_2022-12_16_24"

filenameCancer_T3_G1 = "Wisconsin Breast Cancer Database_18_02_2022-17_34_41"
filenameCancer_T3_G2 = "Wisconsin Breast Cancer Database_18_02_2022-16_21_41"
filenameCancer_T3_G3 = "Wisconsin Breast Cancer Database_18_02_2022-15_03_33"


# Load files
# Parkinson
Parkinson_T1_G1 = pd.read_csv(resultsFolderPath + filenameParkinson_T1_G1 + ".csv")
Parkinson_T1_G2 = pd.read_csv(resultsFolderPath + filenameParkinson_T1_G2 + ".csv")
Parkinson_T1_G3 = pd.read_csv(resultsFolderPath + filenameParkinson_T1_G3 + ".csv")
Parkinson_T2_G1 = pd.read_csv(resultsFolderPath + filenameParkinson_T2_G1 + ".csv")
Parkinson_T2_G2 = pd.read_csv(resultsFolderPath + filenameParkinson_T2_G2 + ".csv")
Parkinson_T2_G3 = pd.read_csv(resultsFolderPath + filenameParkinson_T2_G3 + ".csv")
Parkinson_T3_G1 = pd.read_csv(resultsFolderPath + filenameParkinson_T3_G1 + ".csv")
Parkinson_T3_G2 = pd.read_csv(resultsFolderPath + filenameParkinson_T3_G2 + ".csv")
Parkinson_T3_G3 = pd.read_csv(resultsFolderPath + filenameParkinson_T3_G3 + ".csv")
# Heart
Heart_T1_G1 = pd.read_csv(resultsFolderPath + filenameHeart_T1_G1 + ".csv")
Heart_T1_G2 = pd.read_csv(resultsFolderPath + filenameHeart_T1_G2 + ".csv")
Heart_T1_G3 = pd.read_csv(resultsFolderPath + filenameHeart_T1_G3 + ".csv")
Heart_T2_G1 = pd.read_csv(resultsFolderPath + filenameHeart_T2_G1 + ".csv")
Heart_T2_G2 = pd.read_csv(resultsFolderPath + filenameHeart_T2_G2 + ".csv")
Heart_T2_G3 = pd.read_csv(resultsFolderPath + filenameHeart_T2_G3 + ".csv")
Heart_T3_G1 = pd.read_csv(resultsFolderPath + filenameHeart_T3_G1 + ".csv")
Heart_T3_G2 = pd.read_csv(resultsFolderPath + filenameHeart_T3_G2 + ".csv")
Heart_T3_G3 = pd.read_csv(resultsFolderPath + filenameHeart_T1_G3 + ".csv")
# Cancer
Cancer_T1_G1 = pd.read_csv(resultsFolderPath + filenameCancer_T1_G1 + ".csv")
Cancer_T1_G2 = pd.read_csv(resultsFolderPath + filenameCancer_T1_G2 + ".csv")
Cancer_T1_G3 = pd.read_csv(resultsFolderPath + filenameCancer_T1_G3 + ".csv")
Cancer_T2_G1 = pd.read_csv(resultsFolderPath + filenameCancer_T2_G1 + ".csv")
Cancer_T2_G2 = pd.read_csv(resultsFolderPath + filenameCancer_T2_G2 + ".csv")
Cancer_T2_G3 = pd.read_csv(resultsFolderPath + filenameCancer_T2_G3 + ".csv")
Cancer_T3_G1 = pd.read_csv(resultsFolderPath + filenameCancer_T3_G1 + ".csv")
Cancer_T3_G2 = pd.read_csv(resultsFolderPath + filenameCancer_T3_G2 + ".csv")
Cancer_T3_G3 = pd.read_csv(resultsFolderPath + filenameCancer_T3_G3 + ".csv")

T1, T2, T3 = 30, 45, 60
G1_1, G2_1, G3_1 = 0, 0, 0  #10*np.log10(np.sqrt(T1)), 10*np.log10(T1/2), 10*np.log10(T1)
G1_2, G2_2, G3_2 = 0, 0, 0  #10*np.log10(np.sqrt(T2)), 10*np.log10(T2/2), 10*np.log10(T2)
G1_3, G2_3, G3_3 = 0, 0, 0  #10*np.log10(np.sqrt(T3)), 10*np.log10(T3/2), 10*np.log10(T3)

# fig, axe = plt.subplots(figsize=(8.4, 8.4))
# plt.plot(Cancer60_G2["SNR"], Cancer60_G2["Error probability (joint: alpha, beta)"], label="Cancer, $G=T$",
#                 linestyle='--', marker='o', color='green', markersize=6)
# plt.plot(Cancer60_G3["SNR"], Cancer60_G3["Error probability (joint: alpha, beta)"], label="Cancer, $G=T^2$",
#                 linestyle='--', marker='o', color='green', markersize=9)

# Figures for JSAC paper
if False:
    # Figure: T=30, Parkinson, G=sqrt(T), G=T/2, G=T; T=45, Heart, G=sqrt(T), G=T/2, G=T; T=60, Cancer, G=sqrt(T), G=T/2, G=T
    with plt.style.context(['science', 'grid']):
        fig, axe = plt.subplots(figsize=(8.4, 8.4))
        im1 ,= plt.plot(Parkinson_T1_G1["SNR"]-G1_1, Parkinson_T1_G1["Error probability (joint: alpha, beta)"], label="Parkinson, $G=\sqrt{T}$",    linestyle=':', marker='o', color='red', markersize=3)
        im2 ,= plt.plot(Parkinson_T1_G2["SNR"]-G2_1, Parkinson_T1_G2["Error probability (joint: alpha, beta)"], label="Parkinson, $G=T/2$",           linestyle=':', marker='o', color='red', markersize=6)
        im3 ,= plt.plot(Parkinson_T1_G3["SNR"]-G3_1, Parkinson_T1_G3["Error probability (joint: alpha, beta)"], label="Parkinson, $G=T$",         linestyle=':', marker='o', color='red', markersize=9)
        im4 ,= plt.plot(Heart_T2_G1["SNR"]-G1_2, Heart_T2_G1["Error probability (joint: alpha, beta)"], label="Heart, $G=\sqrt{T}$",   linestyle='-', marker='o', color='blue', markersize=3)
        im5 ,= plt.plot(Heart_T2_G2["SNR"]-G2_2, Heart_T2_G2["Error probability (joint: alpha, beta)"], label="Heart, $G=T/2$",          linestyle='-', marker='o', color='blue', markersize=6)
        im6 ,= plt.plot(Heart_T2_G3["SNR"]-G3_2, Heart_T2_G3["Error probability (joint: alpha, beta)"], label="Heart, $G=T$",        linestyle='-', marker='o', color='blue', markersize=9)
        im7 ,= plt.plot(Cancer_T3_G1["SNR"]-G1_3, Cancer_T3_G1["Error probability (joint: alpha, beta)"], label="Cancer, $G=\sqrt{T}$",    linestyle='--', marker='o', color='green', markersize=3)
        im8 ,= plt.plot(Cancer_T3_G2["SNR"]-G2_3, Cancer_T3_G2["Error probability (joint: alpha, beta)"], label="Cancer, $G=T/2$",           linestyle='--', marker='o', color='green', markersize=6)
        im9 ,= plt.plot(Cancer_T3_G3["SNR"]-G3_3, Cancer_T3_G3["Error probability (joint: alpha, beta)"], label="Cancer, $G=T$",         linestyle='--', marker='o', color='green', markersize=9)

        # create blank rectangle
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        # Create organized list containing all handles for table. Extra represent empty space
        legend_handle = [extra, extra, extra, extra, extra, im1, im2, im3, extra, im4, im5, im6, extra, im7, im8, im9]
        # Define the labels
        label_row_1 = ["", r"$G=\sqrt{T}$", r"$G=T/2$", r"$G=T$"]
        label_j_1 = [r"Parkinson"]
        label_j_2 = [r"Heart disease"]
        label_j_3 = [r"Breast cancer"]
        label_empty = [""]
        # organize labels for table construction
        legend_labels = np.concatenate(
            [label_row_1, label_j_1, label_empty * 3, label_j_2, label_empty * 3, label_j_3, label_empty * 3])
        # Create legend
        axe.legend(legend_handle, legend_labels, fontsize=14,
                  loc='lower left', ncol=4, shadow=False, handletextpad=-2, framealpha=0.8)

        plt.autoscale(tight=True)
        plt.xlabel(r'SNR [dB]', fontsize=20)
        plt.ylabel(r'Error probability [\%]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0,50])
        plt.show()

    # fig.savefig('error_prob.png', format='png', dpi=300)
    fig.savefig('error_prob_a_g.pdf', format='pdf')

if False:
    # Figure: lower and upper bounds on mismatch
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
        label_row_1 = ["", r"Upper bnd (Train)", r"Lower bnd (Train)", r"Optimized (Test)"]
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

        # plt.legend(fontsize=16, ncol=1, loc='upper right', framealpha=0.8)
        plt.xlabel(r'SNR [dB]', fontsize=20)
        plt.ylabel(r'Mismatch probability [\%]', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([0, 50])
        plt.xlim([-25, 10])
        plt.show()

    # fig.savefig('mismatch_bounds.png', format='png', dpi=300)
    fig.savefig('mismatch_bounds.pdf')

if True:
    # Figure: SNR gain
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

    # Parkinson
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

        plt.plot(x11, y11, linestyle='', marker='o', color='blue', markersize=3, label='Parkinson, T=30, $G=\sqrt{T}$')
        plt.plot(x12, y12, linestyle='', marker='o', color='blue', markersize=6, label='Parkinson, T=30, $G=T/2$')
        plt.plot(x13, y13, linestyle='', marker='o', color='blue', markersize=9, label='Parkinson, T=30, $G=T$')

        plt.plot(x21, y21, linestyle='', marker='o', color='green', markersize=3, label='Heart, T=45, $G=\sqrt{T}$')
        plt.plot(x22, y22, linestyle='', marker='o', color='green', markersize=6,  label='Heart, T=45, $G=T/2$')
        plt.plot(x23, y23, linestyle='', marker='o', color='green', markersize=9, label='Heart, T=45, $G=T$')

        plt.plot(x31, y31, linestyle='', marker='o', color='red', markersize=3, label='Cancer, T=60, $G=\sqrt{T}$')
        plt.plot(x32, y32, linestyle='', marker='o', color='red', markersize=6, label='Cancer, T=60, $G=T/2$')
        plt.plot(x33, y33, linestyle='', marker='o', color='red', markersize=9, label='Cancer, T=60, $G=T$')

        plt.legend(fontsize=9.9, ncol=3, loc='lower center', framealpha=0.8)
        plt.xlabel(r'Error probability [\%]', fontsize=15)
        plt.ylabel(r'SNR gain [dB]', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0, 12])
        plt.show()

    # fig.savefig('snr_gain.png', format='png', dpi=300)
    fig.savefig('snr_gain_a_g.pdf', format='pdf')
