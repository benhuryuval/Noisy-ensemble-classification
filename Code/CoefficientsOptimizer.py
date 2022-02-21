import numpy as np
import scipy
from auxiliaryFunctions import QFunc, calc_constraint, calc_uniform, scale_to_constraint
import matplotlib.pyplot as plt


class CoefficientsOptimizer:

    def __init__(self, train_confidence_levels, noise_vars, power_limit=None, p_norm=2):
        self.trainConfidenceLevels = train_confidence_levels
        self.noiseVars = noise_vars
        self.powerLimit = power_limit
        self.pNorm = p_norm

    ''' - - - Unconstrained optimization of weights - - - '''
    # def gradient_alpha(self, alpha):
    #     """ This function calculates the gradient of the cost function"""
    #     n_data_samps = np.size(self.trainConfidenceLevels, 1)
    #     numBaseClassifiers = np.size(self.trainConfidenceLevels, 0)
    #     grad = np.zeros([numBaseClassifiers, 1])
    #     for m in range(numBaseClassifiers):
    #         sign_vec = np.sign(np.sum(self.trainConfidenceLevels, 0))
    #         alpha_h_vec = self.trainConfidenceLevels.T.dot(alpha).T
    #         alpha_sig = np.sqrt(np.power(alpha, 2).T.dot(self.noiseVars))
    #         H_m = self.trainConfidenceLevels[m, :].reshape([1,n_data_samps])
    #         part = (H_m * (alpha_sig ** 2) - alpha[m, 0] * self.noiseVars[m, 0] * alpha_h_vec) / (alpha[m, 0] ** 3)
    #         grad[m, 0] = -1 / np.sqrt((2 * np.pi)) * np.mean(
    #             np.multiply(np.multiply(np.exp(-0.5 * np.power(alpha_h_vec, 2) / (alpha_sig ** 2)), sign_vec),
    #                         part))
    #     return grad

    def gradient_alpha(self, a):
        """ This function calculates the gradient of the cost function"""
        h = self.trainConfidenceLevels
        sigma = self.noiseVars

        h_1 = h.sum(axis=0, keepdims=True)
        a_h = a.T.dot(h)
        a_Sig_a = a.T.dot(np.multiply(sigma, a))
        one_H_one = np.multiply(h_1, h_1)
        a_H_one = np.multiply(a_h, h_1)

        psi = a_H_one / np.sqrt(one_H_one * a_Sig_a)
        psi_tag = (a_Sig_a*h_1*h - a_H_one*(sigma*a)) / (a_Sig_a * np.sqrt(a_Sig_a*one_H_one))
        grad = -1/np.sqrt(2*np.pi) * np.exp(-0.5 * psi**2) * psi_tag

        return np.mean(grad, axis=1, keepdims=True)

        # numBaseClassifiers = np.size(self.trainConfidenceLevels, 0)
        # grad = np.zeros([numBaseClassifiers, 1])
        # for m in range(numBaseClassifiers):
        #     sign_vec = np.sign(np.sum(self.trainConfidenceLevels, 0))
        #     alpha_h_vec = self.trainConfidenceLevels.T.dot(alpha).T
        #     alpha_sig = np.sqrt(np.power(alpha, 2).T.dot(self.noiseVars))
        #     H_m = self.trainConfidenceLevels[m, :].reshape([1,n_data_samps])
        #     part = (H_m * (alpha_sig ** 2) - alpha[m, 0] * self.noiseVars[m, 0] * alpha_h_vec) / (alpha[m, 0] ** 3)
        #     grad[m, 0] = -1 / np.sqrt((2 * np.pi)) * np.mean(
        #         np.multiply(np.multiply(np.exp(-0.5 * np.power(alpha_h_vec, 2) / (alpha_sig ** 2)), sign_vec),
        #                     part))
        # return grad

    ''' - - - Constrained optimization of weights and gains - - - '''
    def calc_mismatch_alpha_beta(self, a, b):
        """ This function calculates the cost function of the optimization"""
        h = self.trainConfidenceLevels
        sigma = self.noiseVars

        h_1 = h.sum(axis=0, keepdims=True)
        a_G_h = a.T.dot(b*h)
        a_Sig_a = a.T.dot(np.multiply(sigma, a))
        one_H_one = np.multiply(h_1, h_1)
        a_G_H_one = np.multiply(a_G_h, h_1)
        psi = a_G_H_one / np.sqrt(one_H_one * a_Sig_a)
        return np.mean(QFunc(psi))

    def gradient_alpha_beta_projection(self, a, b):
        """
        Function that computes the gradient of the mismatch probability \tilde{P}_{\alpha,\beta}(x).
        :param a: coefficients column vector \alpha
        :param b: coefficients column vector \beta
        :param h: matrix of confidence levels column vectors h for all data samples
        :param sigma: noise variance column vector \sigma
        :return: gradient of mismatch probability with respect to \alpha and \beta calculated in (a,b)
        """
        # initializations
        h = self.trainConfidenceLevels
        sigma = self.noiseVars

        # calculate gradient w.r.t alpha
        h_1 = h.sum(axis=0, keepdims=True)
        a_G_h = a.T.dot(b*h)
        a_Sig_a = a.T.dot(np.multiply(sigma, a))
        one_H_one = np.multiply(h_1, h_1)
        a_G_H_one = np.multiply(a_G_h, h_1)
        psi = a_G_H_one / np.sqrt(one_H_one * a_Sig_a)
        psi_tag = (a_Sig_a * h_1 * (b*h) - a_G_H_one * (sigma*a)) / (a_Sig_a * np.sqrt(a_Sig_a * one_H_one))
        grad_alpha = -1/np.sqrt(2*np.pi) * np.mean(np.exp(-0.5 * psi**2) * psi_tag, axis=1, keepdims=True)

        # h_sum = h.sum(axis=0, keepdims=True)
        # # calculate constants
        # sqrt_one_h_ht_one = abs(h_sum)
        # alpha_sigma_alpha = (np.power(a, 2) * sigma).sum()
        # g = ((a * b).T @ h) * h_sum / (sqrt_one_h_ht_one * np.sqrt(alpha_sigma_alpha))
        # # calculate gradient w.r.t alpha
        # parenthesis_term = (b * h) * h_sum / sqrt_one_h_ht_one - g * (a * sigma) / alpha_sigma_alpha
        # grad_alpha = -1/np.sqrt(2*np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1, keepdims=True)

        # calculate gradient w.r.t beta
        A_H_1 = np.multiply(a*h, h_1)
        psi_tag = A_H_1 / np.sqrt(a_Sig_a * one_H_one)
        grad_beta = -1/np.sqrt(2*np.pi) * np.mean(np.exp(-0.5 * psi**2) * psi_tag, axis=1, keepdims=True)

        # parenthesis_term = (a * h) * h_sum / sqrt_one_h_ht_one / np.sqrt(alpha_sigma_alpha)
        # grad_beta = -1/np.sqrt(2*np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1, keepdims=True)

        # get s by projecting gradient for beta on feasible domain
        s = np.zeros([len(grad_beta), 1])
        if grad_beta.__abs__().sum() / b.__abs__().sum() >= 1e-20:  # in case grad_beta is not zero
            s = np.sqrt(self.powerLimit / np.sum(grad_beta**2)) * grad_beta
        return grad_alpha, grad_beta, s

    def gradient_descent(self, alpha_0, beta, max_iter=30000, min_iter=10, tol=1e-5, learn_rate=0.2, decay_rate=0.2):
        """ This function calculates optimal coefficients with gradient descent method using an early stop criteria and
        selecting the minimal value reached throughout the iterations """
        # initializations
        cost_evolution = [None]*max_iter
        alpha_evolution = [None]*max_iter
        eps = 1e-8  # tolerance value for adagrad learning rate update
        # first iteration
        alpha_evolution[0] = alpha_0
        cost_evolution[0] = self.calc_mismatch_alpha_beta(alpha_0, beta)
        step = 0  # initialize gradient-descent step to 0
        i = 0  # iteration index in evolution
        # perform gradient-descent
        for i in range(1, max_iter):
            # calculate grad, update momentum and alpha
            grad = self.gradient_alpha(alpha_evolution[i - 1])
            # update learning rate and advance according to AdaGrad
            learn_rate_upd = np.divide(np.eye(len(alpha_0)) * learn_rate,
                                       np.sqrt(np.diag(np.power(np.squeeze(grad), 2))) + eps)
            step = decay_rate * step - learn_rate_upd.dot(grad)
            alpha_evolution[i] = alpha_evolution[i-1] + step
            # update cost status and history for early stop
            cost_evolution[i] = self.calc_mismatch_alpha_beta(alpha_evolution[i], beta)
            # check convergence
            if i > min_iter and np.abs(cost_evolution[i]-cost_evolution[i-1]) <= tol:
                break
        return cost_evolution, alpha_evolution, i

    def frank_wolfe_alt(self, a0, b0, tol=1e-5, K_max=15000, K_min=10):
        # Initialize
        a, b, rho = [None] * K_max, [None] * K_max, [None] * K_max
        da, db, cost_function = [None] * K_max, [None] * K_max, [None] * K_max
        k = 0
        a[0], b[0] = a0, b0
        cost_function[0] = self.calc_mismatch_alpha_beta(a[0], b[0])
        # apply the Frank-Wolfe algorithm
        for k in range(1, K_max):
            # 1. optimize alpha for previous beta
            cost_ev, a_ev, stop_iter = self.gradient_descent(a[k-1], b[k-1], max_iter=K_min, min_iter=3)
            a_a = a_ev[np.argmin(cost_ev[0:stop_iter+1])]
            cost_a = self.calc_mismatch_alpha_beta(a_a, b[k-1])
            # 2. optimize beta for previous alpha
            tmp_da, tmp_db, db[k] = self.gradient_alpha_beta_projection(a[k-1], b[k-1])  # calculate projected gradient for beta
            rho[k] = 2 / (2 + k)  # determine momentum/step size for beta
            b_b = (1 - rho[k]) * b[k-1] + rho[k] * db[k]  # advance beta
            b_b = scale_to_constraint(b_b, self.powerLimit, self.pNorm)  # normalize beta
            cost_b = self.calc_mismatch_alpha_beta(a[k-1], b_b)
            # 3. optimize alpha for new beta
            cost_ev, a_ev, stop_iter = self.gradient_descent(a[k-1], b_b, max_iter=K_min, min_iter=3)
            a_ba = a_ev[np.argmin(cost_ev[0:stop_iter+1])]
            cost_ba = self.calc_mismatch_alpha_beta(a_ba, b_b)
            # 4. optimize beta for new alpha
            tmp_da, tmp_db, db[k] = self.gradient_alpha_beta_projection(a_a, b[k-1])  # calculate projected gradient for beta
            rho[k] = 2 / (2 + k)  # determine momentum/step size for beta
            b_ab = (1 - rho[k]) * b[k-1] + rho[k] * db[k]  # advance beta
            b_ab = scale_to_constraint(b_ab, self.powerLimit, self.pNorm)  # normalize beta
            cost_ab = self.calc_mismatch_alpha_beta(a_a, b_ab)
            # check new cost functions, and update alpha and beta accordingly
            if   cost_a == np.min([cost_a, cost_b, cost_ba, cost_ab]):
                cost_function[k], a[k], b[k] = cost_a, a_a, b[k-1]
                # print("1")
            elif cost_b == np.min([cost_a, cost_b, cost_ba, cost_ab]):
                cost_function[k], a[k], b[k] = cost_b, a[k-1], b_b
                # print("2")
            elif cost_ba == np.min([cost_a, cost_b, cost_ba, cost_ab]):
                cost_function[k], a[k], b[k] = cost_ba, a_ba, b_b
                # print("3")
            elif cost_ab == np.min([cost_a, cost_b, cost_ba, cost_ab]):
                cost_function[k], a[k], b[k] = cost_ab, a_a, b_ab
                # print("4")
            # recheck convergence of cost function
            if k > K_min and abs(cost_function[k] - cost_function[k-1]) <= tol:
                break
        # return
        return cost_function, a, b, k

    # def frank_wolfe_joint(self, a0, b0, tol=1e-5, learn_rate=0.2, decay_rate=0.2, K_max=15000, K_min=10):
    #     # Initialize
    #     a, b, rho = [None] * K_max, [None] * K_max, [None] * K_max
    #     da, db, cost_function = [None] * K_max, [None] * K_max, [None] * K_max
    #     k = 0
    #     a[0], b[0] = a0, b0
    #     cost_function[0] = self.calc_mismatch_alpha_beta(a[0], b[0])
    #     # apply the Frank-Wolfe algorithm
    #     for k in range(1, K_max):
    #         # 1. optimize alpha for previous beta
    #         cost_ev, a_ev, stop_iter = self.gradient_descent(a[k-1], b[k-1], max_iter=K_max, min_iter=K_min, tol=tol)
    #         a_a = a_ev[np.argmin(cost_ev[0:stop_iter])]
    #         cost_a = self.calc_mismatch_alpha_beta(a_a, b[k-1])
    #         # 2. optimize beta for previous alpha
    #         tmp_da, tmp_db, db[k] = self.gradient_alpha_beta_projection(a[k-1], b[k-1])  # calculate projected gradient for beta
    #         rho[k] = 2 / (2 + k)  # determine momentum/step size for beta
    #         b_b = (1 - rho[k]) * b[k-1] - rho[k] * db[k]  # advance beta
    #         b_b = scale_to_constraint(b_b, self.powerLimit, self.pNorm)  # normalize beta
    #         cost_b = self.calc_mismatch_alpha_beta(a[k-1], b_b)
    #         # 3. check new alpha and new beta
    #         cost_ab = self.calc_mismatch_alpha_beta(a_a, b_b)
    #         # check new cost functions, and update alpha and beta accordingly
    #         if   cost_a == np.min([cost_a, cost_b, cost_ab]):
    #             cost_function[k], a[k], b[k] = cost_a, a_a, b[k-1]
    #         elif cost_b == np.min([cost_a, cost_b, cost_ab]):
    #             cost_function[k], a[k], b[k] = cost_b, a[k-1], b_b
    #         elif cost_ab == np.min([cost_a, cost_b, cost_ab]):
    #             cost_function[k], a[k], b[k] = cost_ab, a_a, b_b
    #         # recheck convergence of cost function
    #         if k > K_min and abs(cost_function[k] - cost_function[k-1]) <= tol:
    #             break
    #     # return
    #     return cost_function, a, b, k

    # - - - Update alpha and beta together
    def frank_wolfe_joint(self, a0, b0, tol=1e-5, learn_rate=0.2, decay_rate=0.2, K_max=15000, K_min=10):
        # Initialize
        eps = 1e-8  # tolerance value for adagrad learning rate update
        a, b, rho = [None] * K_max, [None] * K_max, [None] * K_max
        da, db, cost_function = [None] * K_max, [None] * K_max, [None] * K_max
        k, a[0], b[0] = 0, a0, b0
        step_a, step_b = 0, 0
        cost_function[0] = self.calc_mismatch_alpha_beta(a[0], b[0])
        # apply the Frank-Wolfe algorithm
        for k in range(1, K_max):
            # calculate gradients w.r.t a and b
            da[k-1], tmp_db, db[k-1] = self.gradient_alpha_beta_projection(a[k-1], b[k-1]) # calculate gradient for alpha and projected gradient for beta

            # 1. update momentum and advance alpha
            learn_rate_upd = np.divide(np.eye(len(a0)) * learn_rate,
                                       np.sqrt(np.diag(np.power(np.squeeze(da[k-1]), 2))) + eps)  # update learning rate and advance according to AdaGrad
            step_a = decay_rate * step_a - learn_rate_upd.dot(da[k-1])
            a[k] = a[k-1] + step_a  # advance alpha

            # 2. advance beta
            # learn_rate_upd = np.divide(np.eye(len(b0)) * learn_rate,
            #                            np.sqrt(np.diag(np.power(np.squeeze(db[k-1]), 2))) + eps)  # update learning rate and advance according to AdaGrad
            # step_b = decay_rate * step_b - learn_rate_upd.dot(db[k-1])
            # step_b = (1-2/(2+k)) * step_b - learn_rate_upd.dot(db[k - 1])
            # b[k] = b[k-1] + step_b  # advance beta

            rho[k] = 0.25 * 2 / (2 + k)  # determine momentum/step size for beta
            b[k] = (1 - rho[k]) * b[k-1] - rho[k] * db[k-1]  # advance beta

            # b[k] = b[k-1] + db[k-1]  # advance beta

            b[k] = scale_to_constraint(b[k], self.powerLimit, self.pNorm)  # normalize beta
            # update cost function history and check convergence
            cost_function[k] = self.calc_mismatch_alpha_beta(a[k], b[k])
            if k > K_min and abs(cost_function[k] - cost_function[k-1]) <= tol: break
        # return
        return cost_function, a, b, k

    def optimize_coefficients_power(self, method='Frank-Wolfe', tol=0.0001, max_iter=15000, min_iter=10):
        n_estimators = len(self.noiseVars)
        # initialize coefficients
        a0 = np.ones([n_estimators, 1])  # initialize uniform aggregation coefficients
        # optimize coefficients and power allocation
        if method == 'Alpha-Beta-Joint':  # optimize coefficients and power allocation
            b0 = calc_uniform(n_estimators, self.powerLimit, self.pNorm)  # initialize uniform power subject to constraint
            mismatch_prob, alpha, beta, stop_iter = self.frank_wolfe_joint(a0, b0, tol=tol, K_max=max_iter, K_min=min_iter)
        if method == 'Alpha-Beta-Alternate':  # optimize coefficients and power allocation
            b0 = calc_uniform(n_estimators, self.powerLimit, self.pNorm)  # initialize uniform power per-channel
            mismatch_prob, alpha, beta, stop_iter = self.frank_wolfe_alt(a0, b0, tol=tol, K_max=max_iter, K_min=min_iter)
        elif method == 'Alpha-UniformBeta':  # optimize coefficients with uniform constrained power allocation
            beta = calc_uniform(n_estimators, self.powerLimit, self.pNorm)  # initialize uniform power per-channel
            mismatch_prob, alpha, stop_iter = self.gradient_descent(a0, beta, max_iter=max_iter, min_iter=min_iter)
        elif method == 'Alpha-UnitBeta':  # optimize coefficients with unit power allocation per-channel
            beta = np.ones([n_estimators, 1])  # initialize unit power per-channel
            mismatch_prob, alpha, stop_iter = self.gradient_descent(a0, beta, max_iter=max_iter, min_iter=min_iter)
        # return
        return mismatch_prob, alpha, beta, stop_iter