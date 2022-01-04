import numpy as np
import scipy
from auxiliaryFunctions import QFunc
import matplotlib.pyplot as plt


class CoefficientsOptimizer:

    def __init__(self, train_confidence_levels, noise_vars, power_limit=None, p_norm=2):
        self.trainConfidenceLevels = train_confidence_levels
        self.noiseVars = noise_vars
        self.powerLimit = power_limit
        self.pNorm = p_norm

    ''' - - - Unconstrained optimization of weights - - - '''
    def calc_mismatch_unconstrained(self, alpha):
        """ This function calculates the cost function of the optimization"""
        sign_vec = np.sign(np.sum(self.trainConfidenceLevels, axis=0))
        alpha_h_vec = self.trainConfidenceLevels.T.dot(alpha).T
        alpha_sig = np.sqrt(np.power(alpha, 2).T.dot(self.noiseVars))
        return np.mean(QFunc(np.multiply(sign_vec, alpha_h_vec) / alpha_sig))

    def gradient_alpha(self, alpha):
        """ This function calculates the gradient of the cost function"""
        n_data_samps = np.size(self.trainConfidenceLevels, 1)
        numBaseClassifiers = np.size(self.trainConfidenceLevels, 0)
        grad = np.zeros([numBaseClassifiers, 1])
        for m in range(numBaseClassifiers):
            sign_vec = np.sign(np.sum(self.trainConfidenceLevels, 0))
            alpha_h_vec = self.trainConfidenceLevels.T.dot(alpha).T
            alpha_sig = np.sqrt(np.power(alpha, 2).T.dot(self.noiseVars))
            H_m = self.trainConfidenceLevels[m, :].reshape([1,n_data_samps])
            part = (H_m * (alpha_sig ** 2) - alpha[m, 0] * self.noiseVars[m, 0] * alpha_h_vec) / (alpha[m, 0] ** 3)
            grad[m, 0] = -1 / np.sqrt((2 * np.pi)) * np.sum(
                np.multiply(np.multiply(np.exp(-0.5 * np.power(alpha_h_vec, 2) / (alpha_sig ** 2)), sign_vec),
                            part))
        return grad

    def optimize_coefficients(self, method='BFGS', tol_val=1e-5, max_iter=15000, learn_rate=1e-2,
                    decay_rate=0.9):
        """calculates the optimal unconstrained coefficients (alpha) using a chosen optimization method"""
        n_estimators = len(self.noiseVars)
        x0 = 1 * np.ones([n_estimators, 1])

        if method == 'BFGS':
            max_iter = int(max_iter)
            bfgs_func_args = (self.trainConfidenceLevels, self.noiseVars)
            res_struct = scipy.optimize.minimize(self.calc_mismatch_unconstrained, x0, args=bfgs_func_args,
                                                 method='BFGS', tol=tol_val, options={'maxiter': max_iter})
            optimal_coef = res_struct.x
            cost_list = []
        elif method == 'GD':
            cost_list, alpha, stop_iter = self.gradient_descent(x0, max_iter=max_iter, tol=tol_val,
                                                                    learn_rate=learn_rate, decay_rate=decay_rate)
            optimal_coef = alpha[np.argmin(cost_list[0:stop_iter + 1])]
        else:
            raise NameError('NoSuchMethod')

        return optimal_coef, cost_list

    ''' - - - Constrained optimization of weights and gains - - - '''
    def calc_mismatch_alpha_beta(self, a, b):
        """ This function calculates the cost function of the optimization"""
        h = self.trainConfidenceLevels
        sigma = self.noiseVars
        sqrt_one_h_ht_one = abs(h.sum(axis=0))
        alpha_sigma_alpha = (np.power(a, 2) * sigma).sum()
        g = ((a * b).T @ h) * h.sum(axis=0) / (sqrt_one_h_ht_one * np.sqrt(alpha_sigma_alpha))
        return np.mean(QFunc(g))

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
        h_sum = h.sum(axis=0, keepdims=True)
        # calculate constants
        sqrt_one_h_ht_one = abs(h_sum)
        alpha_sigma_alpha = (np.power(a, 2) * sigma).sum()
        g = ((a * b).T @ h) * h_sum / (sqrt_one_h_ht_one * np.sqrt(alpha_sigma_alpha))
        # calculate gradient w.r.t alpha
        parenthesis_term = (b * h) * h_sum / sqrt_one_h_ht_one + g * (a * sigma) / alpha_sigma_alpha
        grad_alpha = -1/np.sqrt(2*np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1, keepdims=True)
        # calculate gradient w.r.t beta
        parenthesis_term = (a * h) * h_sum / sqrt_one_h_ht_one / np.sqrt(alpha_sigma_alpha)
        grad_beta = -1/np.sqrt(2*np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1, keepdims=True)
        # get s by projecting gradient for beta on feasible domain
        s = np.zeros([len(grad_beta), 1])
        if grad_beta.__abs__().sum() / b.__abs__().sum() >= 1e-12:  # in case grad_beta is not zero
            if self.pNorm == 1:
                s[np.argmax(grad_beta)] = self.powerLimit
            elif self.pNorm == 2:
                s = self.powerLimit / np.sqrt(np.power(grad_beta, 2).sum()) * grad_beta
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

    def frank_wolfe(self, a0, b0, tol=1e-5, K_max=15000, K_min=10):
        # Initialize
        a, b, rho = [None] * K_max, [None] * K_max, [None] * K_max
        da, db, cost_function = [None] * K_max, [None] * K_max, [None] * K_max
        k = 0
        a[0], b[0] = a0, b0
        cost_function[0] = self.calc_mismatch_alpha_beta(a[0], b[0])
        # apply the Frank-Wolfe algorithm
        for k in range(1, K_max):
            # optimize alpha for previous beta
            cost_ev, a_ev, stop_iter = self.gradient_descent(a[k-1], b[k-1], max_iter=K_min, min_iter=K_min)
            a_a = a_ev[np.argmin(cost_ev[0:stop_iter])]
            cost_a = self.calc_mismatch_alpha_beta(a_a, b[k-1])
            # optimize beta for previous alpha
            tmp_da, tmp_db, db[k] = self.gradient_alpha_beta_projection(a[k-1], b[k-1])  # calculate projected gradient for beta
            rho[k] = 2 / (2 + k)  # determine momentum/step size for beta
            b_b = (1 - rho[k]) * b[k-1] - rho[k] * db[k]  # advance beta
            cost_b = self.calc_mismatch_alpha_beta(a[k-1], b_b)
            # optimize alpha for new beta
            cost_ev, a_ev, stop_iter = self.gradient_descent(a[k-1], b_b, max_iter=K_min, min_iter=K_min)
            a_ba = a_ev[np.argmin(cost_ev[0:stop_iter])]
            cost_ba = self.calc_mismatch_alpha_beta(a_ba, b_b)
            # optimize beta for new alpha
            tmp_da, tmp_db, db[k] = self.gradient_alpha_beta_projection(a_a, b[k-1])  # calculate projected gradient for beta
            rho[k] = 2 / (2 + k)  # determine momentum/step size for beta
            b_ab = (1 - rho[k]) * b[k-1] - rho[k] * db[k]  # advance beta
            cost_ab = self.calc_mismatch_alpha_beta(a_a, b_ab)
            # check new cost functions, and update alpha and beta accordingly
            if   cost_a == np.min([cost_a, cost_b, cost_ba, cost_ab]):
                cost_function[k], a[k], b[k] = cost_a, a_a, b[k-1]
            elif cost_b == np.min([cost_a, cost_b, cost_ba, cost_ab]):
                cost_function[k], a[k], b[k] = cost_a, a[k-1], b_b
            elif cost_ba == np.min([cost_a, cost_b, cost_ba, cost_ab]):
                cost_function[k], a[k], b[k] = cost_a, a_ba, b_b
            elif cost_ab == np.min([cost_a, cost_b, cost_ba, cost_ab]):
                cost_function[k], a[k], b[k] = cost_a, a_a, b_ab
            # recheck convergence of cost function
            if k > K_min and abs(cost_function[k] - cost_function[k-1]) <= tol:
                break
        # return
        return cost_function, a, b, k

    def optimize_coefficients_power(self, method='Frank-Wolfe', tol=0.0001, max_iter=15000, min_iter=10):
        n_estimators = len(self.noiseVars)
        # initialize coefficients
        a0 = np.ones([n_estimators, 1])  # initialize uniform aggregation coefficients
        # optimize coefficients and power allocation
        if method == 'Alpha-Beta':  # optimize coefficients and power allocation
            b0 = self.powerLimit * np.ones([n_estimators, 1]) / np.linalg.norm(np.ones([n_estimators, 1]), self.pNorm)  # initialize uniform power subject to constraint
            mismatch_prob, alpha, beta, stop_iter = self.frank_wolfe(a0, b0, tol=tol, K_max=max_iter, K_min=min_iter)
        elif method == 'Alpha-UniformBeta':  # optimize coefficients with uniform constrained power allocation
            beta = self.powerLimit * np.ones([n_estimators, 1]) / np.linalg.norm(np.ones([n_estimators, 1]), self.pNorm)  # initialize uniform power subject to constraint
            mismatch_prob, alpha, stop_iter = self.gradient_descent(a0, beta, max_iter=max_iter, min_iter=min_iter)
        elif method == 'Alpha-UnitBeta':  # optimize coefficients with unit power allocation per-channel
            beta = np.ones([n_estimators, 1])  # initialize unit power per-channel
            mismatch_prob, alpha, stop_iter = self.gradient_descent(a0, beta, max_iter=max_iter, min_iter=min_iter)
        # return
        return mismatch_prob, alpha, beta, stop_iter




