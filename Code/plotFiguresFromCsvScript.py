import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# load simulation files
relativeRootPath = "C:\\Users\\Yuval\\Google Drive\\PhD\\Boosting\\Power-constrained weighted combining\\Simulation\\"
resultsFolderPath = relativeRootPath + "Results\\"

# p=1
filenameHeart15 = "Heart Diseases Database_08_11_2021-08_07_01"
filenameHeart30 = "Heart Diseases Database_08_11_2021-08_10_40"
filenameHeart45 = "Heart Diseases Database_08_11_2021-08_14_17"

# p=2
# filenameHeart15 = "Heart Diseases Database_03_11_2021-16_09_11"
# filenameHeart30 = "Heart Diseases Database_03_11_2021-16_09_20"
# filenameHeart45 = "Heart Diseases Database_03_11_2021-16_09_28"


HeartP1 = pd.read_csv(resultsFolderPath + filenameHeart15 + ".csv")
HeartP2 = pd.read_csv(resultsFolderPath + filenameHeart30 + ".csv")
HeartP3 = pd.read_csv(resultsFolderPath + filenameHeart45 + ".csv")

# Figure: lower and upper bounds
fig = plt.figure()
plt.plot(HeartP1["SNR"], HeartP1["Upper bound (mismatch probability)"], label='Heart Disease, T=30, Upper bound', linestyle='--', marker='v', color='blue')
plt.plot(HeartP1["SNR"], HeartP1["Lower bound (mismatch probability)"], label='Heart Disease, T=30, Lower bound', linestyle='--', marker='^', color='blue')
plt.plot(HeartP2["SNR"], HeartP2["Upper bound (mismatch probability)"], label='Heart Disease, T=30, Upper bound', linestyle='--', marker='v', color='green')
plt.plot(HeartP2["SNR"], HeartP2["Lower bound (mismatch probability)"], label='Heart Disease, T=30, Lower bound', linestyle='--', marker='^', color='green')
plt.plot(HeartP3["SNR"], HeartP3["Upper bound (mismatch probability)"], label='Heart Disease, T=30, Upper bound', linestyle='--', marker='v', color='red')
plt.plot(HeartP3["SNR"], HeartP3["Lower bound (mismatch probability)"], label='Heart Disease, T=30, Lower bound', linestyle='--', marker='^', color='red')
plt.legend(loc="upper right", fontsize=12)
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Mismatch probability bound [%]", fontsize=14)
plt.grid()


Cancer15 = pd.read_csv(resultsFolderPath + filenameCancer15 + ".csv")
Cancer30 = pd.read_csv(resultsFolderPath + filenameCancer30 + ".csv")
Cancer45 = pd.read_csv(resultsFolderPath + filenameCancer45 + ".csv")
Cancer60 = pd.read_csv(resultsFolderPath + filenameCancer60 + ".csv")

Parkinson15 = pd.read_csv(resultsFolderPath + filenameParkinson15 + ".csv")
Parkinson30 = pd.read_csv(resultsFolderPath + filenameParkinson30 + ".csv")
Parkinson45 = pd.read_csv(resultsFolderPath + filenameParkinson45 + ".csv")
Parkinson60 = pd.read_csv(resultsFolderPath + filenameParkinson60 + ".csv")

# calculate mismatch probability gain
Heart15Gain = Heart15["Error probability (trivial)"]-Heart15["Error probability (Optimized GD)"]
Heart30Gain = Heart30["Error probability (trivial)"]-Heart30["Error probability (Optimized GD)"]
Heart45Gain = Heart45["Error probability (trivial)"]-Heart45["Error probability (Optimized GD)"]
Heart60Gain = Heart60["Error probability (trivial)"]-Heart60["Error probability (Optimized GD)"]

Cancer15Gain = Cancer15["Error probability (trivial)"]-Cancer15["Error probability (Optimized GD)"]
Cancer30Gain = Cancer30["Error probability (trivial)"]-Cancer30["Error probability (Optimized GD)"]
Cancer45Gain = Cancer45["Error probability (trivial)"]-Cancer45["Error probability (Optimized GD)"]
Cancer60Gain = Cancer60["Error probability (trivial)"]-Cancer60["Error probability (Optimized GD)"]

Parkinson15Gain = Parkinson15["Error probability (trivial)"]-Parkinson15["Error probability (Optimized GD)"]
Parkinson30Gain = Parkinson30["Error probability (trivial)"]-Parkinson30["Error probability (Optimized GD)"]
Parkinson45Gain = Parkinson45["Error probability (trivial)"]-Parkinson45["Error probability (Optimized GD)"]
Parkinson60Gain = Parkinson60["Error probability (trivial)"]-Parkinson60["Error probability (Optimized GD)"]

# Figure: Heart, T=45; Parkinson, T=30; Cancer, T=60
fig = plt.figure()
plt.plot(Parkinson30["SNR"], Parkinson30["Error probability (trivial)"], label='Parkinson, T=30, Trivial', marker='v', color='blue')
plt.plot(Parkinson30["SNR"], Parkinson30["Error probability (Optimized GD)"], label='Parkinson, T=30, Optimal (GD)', marker='^', color='blue')
plt.plot(Heart45["SNR"], Heart45["Error probability (trivial)"], label='Heart Disease, T=45, Trivial', linestyle='--', marker='v', color='green')
plt.plot(Heart45["SNR"], Heart45["Error probability (Optimized GD)"], label='Heart Disease, T=45, Optimal (GD)', linestyle='--', marker='^', color='green')
plt.plot(Cancer60["SNR"], Cancer60["Error probability (trivial)"], label='Breast cancer, T=60, Trivial', linestyle=':', marker='v', color='red')
plt.plot(Cancer60["SNR"], Cancer60["Error probability (Optimized GD)"], label='Breast cancer, T=60, Optimal (GD)', linestyle=':', marker='^', color='red')
plt.legend(loc="upper right", fontsize=12)
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Error probability [%]", fontsize=14)
plt.grid()

# Figure: Overall mismatch improvements plot
fig = plt.figure()
plt.plot(np.tile(Heart15["SNR"], 4), np.concatenate((Heart15Gain, Heart30Gain, Heart45Gain, Heart60Gain), axis=None), label='Heart disease, T=15, 30, 45, 60', linestyle='', marker='o', color='blue')
plt.plot(np.tile(Cancer15["SNR"], 4), np.concatenate((Cancer15Gain, Cancer30Gain, Cancer45Gain, Cancer60Gain), axis=None), label='Breast cancer, T=15, 30, 45, 60', linestyle='', marker='o', color='green')
plt.plot(np.tile(Parkinson15["SNR"], 4), np.concatenate((Parkinson15Gain, Parkinson30Gain, Parkinson45Gain, Parkinson60Gain), axis=None), label='Parkinson\'s disease, T=15, 30, 45, 60', linestyle='', marker='o', color='red')
plt.legend(loc="upper right", fontsize=12)
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Error probability improvement [%]", fontsize=14)
plt.grid()


#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# All error probability improvement vs snr for ascending T plots
fig = plt.figure()
plt.plot(Heart15["SNR"], Heart15Gain, label='T=15', linestyle='', marker='o')
plt.plot(Heart30["SNR"], Heart30Gain, label='T=30', linestyle='', marker='o')
plt.plot(Heart45["SNR"], Heart45Gain, label='T=45', linestyle='', marker='s')
plt.legend(loc="upper right", fontsize=12)
plt.title("Heart disease")
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Mismatch probability improvement [%]", fontsize=14)
plt.grid()

fig = plt.figure()
plt.plot(Cancer15["SNR"], Cancer15Gain, label='Breast cancer, T=15', linestyle='', marker='o')
plt.plot(Cancer30["SNR"], Cancer30Gain, label='Breast cancer, T=30', linestyle='', marker='o')
plt.plot(Cancer45["SNR"], Cancer45Gain, label='Breast cancer, T=45', linestyle='', marker='s')
plt.plot(Cancer60["SNR"], Cancer60Gain, label='Breast cancer, T=60', linestyle='', marker='s')
plt.legend(loc="upper right", fontsize=12)
plt.title("Breast cancer")
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Mismatch probability improvement [%]", fontsize=14)
plt.grid()

fig = plt.figure()
plt.plot(Parkinson15["SNR"], Parkinson15Gain, label='Parkinson\'s disease, T=15', linestyle='', marker='o')
plt.plot(Parkinson30["SNR"], Parkinson30Gain, label='Parkinson\'s disease, T=30', linestyle='', marker='o')
plt.plot(Parkinson45["SNR"], Parkinson45Gain, label='Parkinson\'s disease, T=45', linestyle='', marker='s')
plt.plot(Parkinson60["SNR"], Parkinson60Gain, label='Parkinson\'s disease, T=60', linestyle='', marker='s')
plt.legend(loc="upper right", fontsize=12)
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Mismatch probability improvement [%]", fontsize=14)
plt.grid()

# Table: calculate improvement for each (dataset,T) pair at SNR=-10dB. Add in table to paper.
# targetSNR=-9
# aa=[
# ["Heart", Heart15Gain[Heart15["SNR"]==targetSNR], Heart30Gain[Heart30["SNR"]==targetSNR], Heart45Gain[Heart45["SNR"]==targetSNR], 0],
# ["Cancer", Cancer15Gain[Parkinson15["SNR"]==targetSNR], Cancer30Gain[Parkinson30["SNR"]==targetSNR], Cancer45Gain[Parkinson45["SNR"]==targetSNR], Cancer60Gain[Parkinson45["SNR"]==targetSNR]],
# ["Parkinson", Parkinson15Gain[Parkinson15["SNR"]==targetSNR], Parkinson30Gain[Parkinson30["SNR"]==targetSNR], Parkinson45Gain[Parkinson45["SNR"]==targetSNR], Parkinson60Gain[Parkinson45["SNR"]==targetSNR]]
#     ]

fig = plt.figure()
plt.plot(Heart15["SNR"], Heart15Gain, label='Heart disease, T=15', linestyle='', marker='o', color='blue')
plt.plot(Heart30["SNR"], Heart30Gain, label='Heart disease, T=30', linestyle='', marker='o', color='blue')
plt.plot(Heart45["SNR"], Heart45Gain, label='Heart disease, T=45', linestyle='', marker='o', color='blue')
plt.plot(Heart45["SNR"], Heart60Gain, label='Heart disease, T=45', linestyle='', marker='o', color='blue')
plt.plot(Cancer15["SNR"], Cancer15Gain, label='Breast cancer, T=15', linestyle='', marker='s', color='green')
plt.plot(Cancer30["SNR"], Cancer30Gain, label='Breast cancer, T=30', linestyle='', marker='s', color='green')
plt.plot(Cancer45["SNR"], Cancer45Gain, label='Breast cancer, T=45', linestyle='', marker='s', color='green')
plt.plot(Cancer60["SNR"], Cancer60Gain, label='Breast cancer, T=60', linestyle='', marker='s', color='green')
plt.plot(Parkinson15["SNR"], Parkinson15Gain, label='Parkinson\'s disease, T=15', linestyle='', marker='o', color='red')
plt.plot(Parkinson30["SNR"], Parkinson30Gain, label='Parkinson\'s disease, T=30', linestyle='', marker='o', color='red')
plt.plot(Parkinson45["SNR"], Parkinson45Gain, label='Parkinson\'s disease, T=45', linestyle='', marker='o', color='red')
plt.plot(Parkinson60["SNR"], Parkinson60Gain, label='Parkinson\'s disease, T=60', linestyle='', marker='o', color='red')
plt.legend(loc="upper right", fontsize=12)
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Error probability improvement [%]", fontsize=14)
plt.grid()


# All mismatch probability vs snr plots
fig = plt.figure()
plt.plot(Heart15["SNR"], Heart15["Mismatch probability (trivial)"], label='Trivial, T=15', color='blue')
plt.plot(Heart15["SNR"], Heart15["Mismatch probability (Optimized GD)"], label='Optimal (GD), T=15')
plt.plot(Heart30["SNR"], Heart30["Mismatch probability (trivial)"], label='Trivial, T=30', linestyle='--')
plt.plot(Heart30["SNR"], Heart30["Mismatch probability (Optimized GD)"], label='Optimal (GD), T=30', linestyle='--')
plt.plot(Heart45["SNR"], Heart45["Mismatch probability (trivial)"], label='Trivial, T=45', linestyle='-.')
plt.plot(Heart45["SNR"], Heart45["Mismatch probability (Optimized GD)"], label='Optimal (GD), T=45', linestyle='-.')
plt.plot(Heart60["SNR"], Heart60["Mismatch probability (trivial)"], label='Trivial, T=60', linestyle=':')
plt.plot(Heart60["SNR"], Heart60["Mismatch probability (Optimized GD)"], label='Optimal (GD), T=60', linestyle=':')
plt.legend(loc="upper right", fontsize=12)
# plt.title(str(database) + "\n Number of Estimators:" + str(numBaseClassifiers))
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Mismatch probability [%]", fontsize=14)
plt.grid()

fig = plt.figure()
plt.plot(Heart15["SNR"], Heart15["Error probability (trivial)"], label='Trivial, T=15', color='blue')
plt.plot(Heart15["SNR"], Heart15["Error probability (Optimized GD)"], label='Optimal (GD), T=15')
plt.plot(Heart30["SNR"], Heart30["Error probability (trivial)"], label='Trivial, T=30', linestyle='--')
plt.plot(Heart30["SNR"], Heart30["Error probability (Optimized GD)"], label='Optimal (GD), T=30', linestyle='--')
plt.plot(Heart45["SNR"], Heart45["Error probability (trivial)"], label='Trivial, T=45', linestyle='-.')
plt.plot(Heart45["SNR"], Heart45["Error probability (Optimized GD)"], label='Optimal (GD), T=45', linestyle='-.')
plt.plot(Heart60["SNR"], Heart60["Error probability (trivial)"], label='Trivial, T=60', linestyle=':')
plt.plot(Heart60["SNR"], Heart60["Error probability (Optimized GD)"], label='Optimal (GD), T=60', linestyle=':')
plt.legend(loc="upper right", fontsize=12)
# plt.title(str(database) + "\n Number of Estimators:" + str(numBaseClassifiers))
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Error probability [%]", fontsize=14)
plt.grid()


# plot mismatch probability vs snr
fig = plt.figure()
plt.plot(Cancer15["SNR"], Cancer15["Mismatch probability (trivial)"], label='Trivial', color='blue')
plt.plot(Cancer15["SNR"], Cancer15["Mismatch probability (Optimized GD)"], label='Optimal (GD)')
plt.plot(Cancer30["SNR"], Cancer30["Mismatch probability (trivial)"], label='Trivial', linestyle='--', color='blue')
plt.plot(Cancer30["SNR"], Cancer30["Mismatch probability (Optimized GD)"], label='Optimal (GD)', linestyle='--')
plt.plot(Cancer45["SNR"], Cancer45["Mismatch probability (trivial)"], label='Trivial', linestyle='-.', color='blue')
plt.plot(Cancer45["SNR"], Cancer45["Mismatch probability (Optimized GD)"], label='Optimal (GD)', linestyle='-.')
plt.plot(Cancer60["SNR"], Cancer60["Mismatch probability (trivial)"], label='Trivial', linestyle=':', color='blue')
plt.plot(Cancer60["SNR"], Cancer60["Mismatch probability (Optimized GD)"], label='Optimal (GD)', linestyle=':')
plt.legend(loc="upper right", fontsize=12)
# plt.title(str(database) + "\n Number of Estimators:" + str(numBaseClassifiers))
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Mismatch probability [%]", fontsize=14)
plt.grid()

fig = plt.figure()
plt.plot(Parkinson15["SNR"], Parkinson15["Mismatch probability (trivial)"], label='Trivial', color='blue')
plt.plot(Parkinson15["SNR"], Parkinson15["Mismatch probability (Optimized GD)"], label='Optimal (GD)')
plt.plot(Parkinson30["SNR"], Parkinson30["Mismatch probability (trivial)"], label='Trivial', linestyle='--', color='blue')
plt.plot(Parkinson30["SNR"], Parkinson30["Mismatch probability (Optimized GD)"], label='Optimal (GD)', linestyle='--')
plt.plot(Parkinson45["SNR"], Parkinson45["Mismatch probability (trivial)"], label='Trivial', linestyle='-.', color='blue')
plt.plot(Parkinson45["SNR"], Parkinson45["Mismatch probability (Optimized GD)"], label='Optimal (GD)', linestyle='-.')
plt.plot(Parkinson60["SNR"], Parkinson60["Mismatch probability (trivial)"], label='Trivial', linestyle=':', color='blue')
plt.plot(Parkinson60["SNR"], Parkinson60["Mismatch probability (Optimized GD)"], label='Optimal (GD)', linestyle=':')
plt.legend(loc="upper right", fontsize=12)
# plt.title(str(database) + "\n Number of Estimators:" + str(numBaseClassifiers))
plt.xlabel("SNR [dB]", fontsize=14)
plt.ylabel("Mismatch probability [%]", fontsize=14)
plt.grid()
