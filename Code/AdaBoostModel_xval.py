import numpy as np
import sklearn as sk
import pandas as pd
from sklearn import preprocessing, model_selection, ensemble

class AdaBoostModel:
    """ Class of real adaboost model - the classifier, train and test set """

    def __init__(self, train_features, train_labels, n_estimators):
        """ Returns Model of real adaboost with M classifiers, trained with database database_name
            with 80% train set, and 20% test set """

        classifier = ensemble.AdaBoostClassifier(sk.tree.DecisionTreeClassifier(max_depth=1),
                                                 n_estimators=n_estimators,
                                                 algorithm='SAMME.R')
        classifier.fit(train_features, train_labels)

        self.train_X = train_features
        self.train_y = train_labels
        self.classifier = classifier

    def calc_confidence_levels(self, X, eps=1e-8):

        estimators_list = self.classifier.estimators_

        X_confidence = np.zeros([len(estimators_list), len(X)])
        for m in range(len(estimators_list)):
            prob = estimators_list[m].predict_proba(X)[:, 0]
            X_confidence[m] = 0.5 * np.log(((1 - prob) + eps) / (prob + eps))

        return X_confidence
