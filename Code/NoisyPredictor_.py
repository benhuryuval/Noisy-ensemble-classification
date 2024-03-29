import numpy as np
from auxiliaryFunctions import QFunc

class NoisyPredictor_:
    """class of the a robust predictor that infers using noisy confidence levels"""

    def __init__(self, model, noise_matrix, tie_breaker=1e-8):
        self.model = model
        self.noise_matrix = noise_matrix
        self.tie_breaker = tie_breaker

    def trivialPredict(self, confidenceLevels):
        """predicts the class of the samples using the noisy confidence level, according to the original Real-AdaBoost
        algorithm"""
        noisy_confidence_levels = confidenceLevels + self.noise_matrix
        soft_prediction = np.sum(noisy_confidence_levels, axis=0)
        hard_prediction = (np.sign(soft_prediction + self.tie_breaker) + 1)/2  # outputs 0 or 1
        return hard_prediction

    def optimalPredict(self, alpha, beta, confidence_levels):
        """predicts the class of the samples using the noisy confidence level, according to the modified Real-AdaBoost
        optimal coefficients"""
        noisy_confidence_levels = confidence_levels + self.noise_matrix
        soft_decision = (beta * confidence_levels + self.noise_matrix).T.dot(alpha).T
        hard_decision = 0.5 * (1 + np.sign(soft_decision + self.tie_breaker))
        return hard_decision

    def upperBound(self, confidenceLevels, noiseVars, powerLimit, pNorm):
        """calculates an upper bound on the mismatch probability of the noisy predictor"""
        numBaseClassifiers = np.size(confidenceLevels, 0)
        N = np.size(confidenceLevels, 1)
        ones_vec = np.ones([1, numBaseClassifiers])
        if pNorm == 1:
            coeff = 0 %powerLimit / numBaseClassifiers
        if pNorm == 2:
            coeff = powerLimit / (numBaseClassifiers*np.sum(noiseVars))
        UB = 0
        for i in range(N):
            hi = confidenceLevels[:, i]
            hi = hi.reshape(hi.shape[0], -1)  # 1D array to column vector
            # Hi = hi @ hi.T
            # UB += QFunc(np.sqrt(coeff * ones_vec @ Hi @ ones_vec.T))
            UB += QFunc(np.sqrt(coeff * hi.T @ hi))
        UB = np.squeeze(UB) / N
        return UB

    def lowerBound(self, confidenceLevels, noiseVars, powerLimit, pNorm):
        """calculates a lower bound on the mismatch probability of the noisy predictor"""
        N = np.size(confidenceLevels, 1)
        if pNorm == 1:
            coeff = 0
        if pNorm == 2:
            coeff = powerLimit / noiseVars.min()
        # coeff = powerLimit.__pow__(pNorm) / noiseVars.min()
        LB = 0
        for i in range(N):
            hi = confidenceLevels[:, i]
            hi = hi.reshape(hi.shape[0], -1)  # 1D array to column vector
            LB += QFunc(np.sqrt(coeff * (hi.T @ hi)))
        LB = np.squeeze(LB) / N
        return LB





