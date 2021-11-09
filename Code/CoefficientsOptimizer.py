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
    def calc_mismatch_unconstrained(self, coefficients):
        """ This function calculates the cost function of the optimization"""
        N = np.size(self.trainConfidenceLevels, 1)
        sign_vec = np.sign(np.sum(self.trainConfidenceLevels, 0))
        alpha_h_vec = np.sum(np.multiply(np.tile(coefficients, (N, 1)).T, self.trainConfidenceLevels), 0)
        alpha_sig = np.sqrt(np.sum((np.multiply(np.power(coefficients, 2), self.noiseVars))))
        return np.sum(QFunc(np.multiply(sign_vec, alpha_h_vec) / alpha_sig))


    def gradient_unconstrained(self, coefficients):
        """ This function calculates the gradient of the cost function"""
        N = np.size(self.trainConfidenceLevels, 1)
        numBaseClassifiers = np.size(self.trainConfidenceLevels, 0)
        val = np.zeros((numBaseClassifiers, 1))
        for m in range(numBaseClassifiers):
            sign_vec = np.sign(np.sum(self.trainConfidenceLevels, 0))
            alpha_h_vec = np.sum(np.multiply(np.tile(coefficients, (N, 1)).T, self.trainConfidenceLevels), 0)
            alpha_sig = np.sqrt(np.sum((np.multiply(np.power(coefficients, 2), self.noiseVars))))
            var_m = self.noiseVars[m]
            H_m = self.trainConfidenceLevels[m, :]
            alpha_m = coefficients[m]

            part = (H_m * (alpha_sig ** 2) - alpha_m * var_m * alpha_h_vec) / (alpha_m ** 3)
            val[m] = -1 / np.sqrt((2 * np.pi)) * np.sum(
                np.multiply(np.multiply(np.exp(-0.5 * np.power(alpha_h_vec, 2) / (alpha_sig ** 2)), sign_vec),
                            part))
        return val


    def gradientDescent_ES_min(self, alpha_0, max_iter, tol, learn_rate, decay_rate,
                               cost_history_max_size=101, history_ratio_th=0.5):
        """ This function calculates optimal coefficients with gradient descent method using an early stop criteria and
        selecting the minimal value reached throughout the iterations """
        numBaseClassifiers = np.size(self.trainConfidenceLevels, 0)
        N = np.size(self.trainConfidenceLevels, 1)
        eps = 1e-8
        coefficients = alpha_0
        diff = 0
        cost_list = np.array([])
        cost_history = np.array([])
        history_stuck_th = int((cost_history_max_size - 1) * history_ratio_th)
        break_counter = 0
        break_counter_th = 10  # int(max_iter)
        min_cost_alpha = alpha_0
        min_cost = self.calc_mismatch_unconstrained(alpha_0)
        for i in range(int(max_iter)):
            # calculate grad, update momentum and alpha
            grad = self.gradient_unconstrained(coefficients)
            learn_rate_upd = np.divide(np.eye(numBaseClassifiers) * learn_rate,
                                       np.sqrt(np.diag(np.power(np.squeeze(grad), 2))) + eps)
            diff = decay_rate * diff - np.squeeze(learn_rate_upd @ grad)
            coefficients = coefficients + diff
            # update cost status and history for early stop
            current_cost = self.calc_mismatch_unconstrained(coefficients)
            cost_list = np.append(cost_list, current_cost)
            cost_history = np.append(cost_history, current_cost)
            cost_history = cost_history[-cost_history_max_size:]
            # check if we found the best spot
            if current_cost <= min_cost:
                min_cost = current_cost
                min_cost_alpha = coefficients
            # update history:
            recent_cost_history = cost_history[1:]
            prev_cost_history = cost_history[:-1]
            # update break counter:
            # maybe change tol into 100*eps:
            if (np.sum(recent_cost_history > prev_cost_history) >= history_stuck_th) or (
                    len(recent_cost_history) >= history_stuck_th and np.max(recent_cost_history) - np.min(
                recent_cost_history) < tol):
                break_counter += 1
            else:
                break_counter = np.max([0, break_counter - 1])
            if break_counter >= break_counter_th:
                break

        # print(f"{i}), cost is: {cost_history[-1]}")
        # plt.plot(cost_list)
        # plt.show()

        # return the best alpha so far:
        return min_cost_alpha


    def optimize_unconstrained_coefficients(self, method='BFGS', tol_val=1e-5, max_iter=15000, learn_rate=1e-2,
                    decay_rate=0.9):
        """calculates the optimal unconstrained coefficients (alpha) using a chosen optimization method"""
        n_estimators = len(self.noiseVars)
        x0 = 1 * np.ones([n_estimators, 1])

        if method == 'BFGS':
            max_iter = int(max_iter)
            bfgs_func_args = (self.trainConfidenceLevels, self.noiseVars)
            res_struct = scipy.optimize.minimize(self.calc_mismatch_unconstrained, x0, args=bfgs_func_args, method='BFGS',
                                                 tol=tol_val,
                                                 options={'maxiter': max_iter})
            optimal_coef = res_struct.x
        elif method == 'GD':
            optimal_coef = self.gradientDescent(np.squeeze(x0), max_iter, tol_val, learn_rate=0.1,
                                                decay_rate=0.01)
        elif method == 'GD_ES':
            optimal_coef = self.gradientDescent_ES(np.squeeze(x0), max_iter, tol_val, learn_rate,
                                                   decay_rate)
        elif method == 'GD_ES_min':
            optimal_coef = self.gradientDescent_ES_min(np.squeeze(x0), max_iter, tol_val, learn_rate,
                                                  decay_rate)
        else:
            raise NameError('NoSuchDatabase')

        return optimal_coef

    ''' - - - Constrained optimization of weights and gains - - - '''
    def calc_mismatch_constrained(self, a, b):
        """ This function calculates the cost function of the optimization"""
        h = self.trainConfidenceLevels
        sigma = self.noiseVars
        sqrt_one_h_ht_one = abs(h.sum(axis=0))
        alpha_sigma_alpha = (np.power(a, 2) * sigma).sum()
        g = ((a * b).T @ h) * h.sum(axis=0) / (sqrt_one_h_ht_one * np.sqrt(alpha_sigma_alpha))
        return np.sum(QFunc(g)) / h.shape[1]

    def gradient_constrained_alpha_beta(self, a, b):
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
        # calculate constants
        sqrt_one_h_ht_one = abs(h.sum(axis=0))
        alpha_sigma_alpha = (np.power(a, 2) * sigma).sum()
        g = ((a * b).T @ h) * h.sum(axis=0) / (sqrt_one_h_ht_one * np.sqrt(alpha_sigma_alpha))
        # calculate gradient w.r.t alpha
        parenthesis_term = (b * h) * h.sum(axis=0) / sqrt_one_h_ht_one + g * (a * sigma.reshape([len(sigma),1])) / alpha_sigma_alpha
        grad_alpha = -1/np.sqrt(2*np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1)
        # calculate gradient w.r.t beta
        parenthesis_term = (a * h) * h.sum(axis=0) / sqrt_one_h_ht_one / np.sqrt(alpha_sigma_alpha)
        grad_beta = -1/np.sqrt(2*np.pi) * np.sum(np.exp(- 1 / 2 * np.power(g, 2)) * parenthesis_term, axis=1)
        # project gradient for beta on feasible domain
        if self.pNorm == 1:
            s = np.zeros([len(grad_beta), 1])
            s[np.argmax(grad_beta)] = self.powerLimit
        elif self.pNorm == 2:
            s = self.powerLimit / np.sqrt(np.power(grad_beta, 2).sum()) * grad_beta
        return grad_alpha.T, grad_beta.T, s

    def frank_wolfe(self, a0, b0, tol=1e-6, K=15000):
        # Initialize
        a, b, rho = [None] * K, [None] * K, [None] * K
        da, db, cost_function = [None] * K, [None] * K, [None] * K
        a[0], b[0] = a0, b0
        cost_function[0] = self.calc_mismatch_constrained(a[0], b[0])
        # Apply the Frank-Wolfe algorithm
        for k in range(1, K):
            rho[k] = 2 / (2 + k)
            da[k], tmp, db[k] = self.gradient_constrained_alpha_beta(a[k-1], b[k-1])
            a[k] = (1 - rho[k]) * a[k - 1] - rho[k] * da[k].reshape((len(da[k]), 1))
            b[k] = (1 - rho[k]) * b[k - 1] - rho[k] * db[k].reshape((len(db[k]), 1))
            cost_function[k] = self.calc_mismatch_constrained(a[k], b[k])
            if k > 100 and tol >= abs(cost_function[k]-cost_function[k-1]):
                break
        # Return
        return cost_function, a, b, k

    def optimize_constrained_coefficients(self, method='Frank-Wolfe', tol=0.0001, K=15000):
        n_estimators = len(self.noiseVars)
        # initialize coefficients
        a0 = np.ones([n_estimators, 1])
        if self.pNorm == 1:
            b0 = np.ones([n_estimators, 1])*self.powerLimit/n_estimators
        elif self.pNorm == 2:
            b0 = np.ones([n_estimators, 1]) * np.sqrt(self.powerLimit / n_estimators)
        # optimize coefficients
        if method == 'Frank-Wolfe':
            mismatch_prob, alpha, beta, stop_iter = self.frank_wolfe(a0, b0, tol, K)
        # return
        return mismatch_prob, alpha, beta, stop_iter




