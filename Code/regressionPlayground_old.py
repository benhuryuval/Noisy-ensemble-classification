# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def predictYb(self, X, noise=0.0, alpha=1.0):
        """Predict regression value for X.
        The predicted regression value of an input sample is computed
        as the weighted median prediction of the regressors in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples for prediction.
        N : {array-like} of shape (n_samples,)
            The additive noise to be added to outputs of regressors.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted regression values.
        """
        # check_is_fitted(self)
        X = self._check_X(X)
        #return self._get_median_predict(X, len(self.estimators_))

        limit = len(self.estimators_)

        # Evaluate predictions of all estimators
        predictions = np.array([est.predict(X) for est in self.estimators_[:limit]]).T

        # Add noise
        noisy_predictions = predictions + noise

        # Sort the predictions
        sorted_idx = np.argsort(noisy_predictions, axis=1)

        # Find index of median prediction for each sample
        est_weights = self.estimator_weights_[sorted_idx]
        est_weights = est_weights * alpha.T
        weight_cdf = sk.utils.extmath.stable_cumsum(est_weights, axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(len(X)), median_idx]

        # Return median predictions
        return noisy_predictions[np.arange(len(X)), median_estimators]

# Create the dataset
rng = np.random.RandomState(1)

n_samples = 100
n_estimators = 300
sigma0 = 1e7

X = np.linspace(0, 6, n_samples)[:, np.newaxis]
f = np.sin(X).ravel() + np.sin(6 * X).ravel()
y = f + rng.normal(0, 0.1, X.shape[0])

# Fit regression model
regr = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4), n_estimators=n_estimators, random_state=rng
)
regr.fit(X, y)

# Predict: clean
y_1 = regr.predict(X)

# Predict: noisy
sigma = sigma0*np.ones([n_estimators, 1])
alpha = np.ones([n_estimators, 1])

sigma = np.zeros([n_estimators, 1])
sigma[0:160] = sigma0
alpha = np.ones([n_estimators, 1])
alpha[0:160] = 0.0

noise = np.array([np.random.normal(0, np.sqrt(sigma_m), n_samples) for sigma_m in sigma]).T
y_2 = predictYb(regr, X, noise, np.ones([n_estimators, 1]))
y_3 = predictYb(regr, X, noise, alpha)

# Plot the results
plt.figure()
plt.plot(X, f, c="k", label="target model", linestyle='-', linewidth=1)
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="r", label="regression model (clean)", linestyle='--', linewidth=2)
plt.plot(X, y_2, c="b", label="regression model (noisy, unweighted)", linestyle=':', linewidth=2)
plt.plot(X, y_3, c="g", label="regression model (noisy, weighted)", linestyle=':', linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression, n_estimators=" + str(n_estimators) + ", sigma=" + str(sigma0))
plt.legend()
# plt.show()

print("MSE:\n\tunweighted: " + str(sk.metrics.mean_squared_error(y, y_2)) + "\n\tweighted: " + str(sk.metrics.mean_squared_error(y, y_3)))