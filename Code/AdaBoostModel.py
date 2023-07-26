import numpy as np
import sklearn as sk
import pandas as pd
from sklearn import preprocessing, model_selection, ensemble
from sklearn.model_selection import KFold

class AdaBoostModel:
    """ Class of real adaboost model - the classifier, train and test set """

    def __init__(self, database_name, database_path, train_size, n_estimators):
        """ Returns Model of real adaboost with M classifiers, trained with database database_name
            with 80% train set, and 20% test set """

        # load database
        if database_name == 'Wisconsin Breast Cancer Database':
            from sklearn.datasets import load_breast_cancer
            dataset = sk.datasets.load_breast_cancer()
            features = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            labels = pd.Categorical.from_codes(dataset.target, dataset.target_names)
        elif database_name == 'Heart Diseases Database':
            dataset = np.genfromtxt(database_path, delimiter=',')
            features = dataset[:, 0:13]
            labels = dataset[:, 13] > 0
        elif database_name == 'Parkinson Database':
            dataset = np.genfromtxt(database_path, delimiter=',')
            features = dataset[1:, 1:23]
            labels = dataset[1:, 23]
        else:
            raise NameError('NoSuchDatabase')

        encoder = preprocessing.LabelEncoder()
        binary_encoded_labels = pd.Series(encoder.fit_transform(labels))
        train_features, test_features, train_labels, test_labels = model_selection.train_test_split(
            features, binary_encoded_labels, train_size=train_size, test_size=1 - train_size)
        classifier = ensemble.AdaBoostClassifier(sk.tree.DecisionTreeClassifier(max_depth=1),
                                                 n_estimators=n_estimators,
                                                 algorithm='SAMME.R')
        classifier.fit(train_features, train_labels)

        self.train_X = train_features
        self.test_X = test_features
        self.train_y = train_labels
        self.test_y = test_labels
        self.classifier = classifier

    def calc_confidence_levels(self, eps=1e-8):

        estimators_list = self.classifier.estimators_

        train_confidence = np.zeros([len(estimators_list), len(self.train_y)])
        for m in range(len(estimators_list)):
            prob = estimators_list[m].predict_proba(self.train_X)[:, 0]
            train_confidence[m] = 0.5 * np.log(((1 - prob) + eps) / (prob + eps))

        test_confidence = np.zeros([len(estimators_list), len(self.test_y)])
        for m in range(len(estimators_list)):
            prob = estimators_list[m].predict_proba(self.test_X)[:, 0]
            test_confidence[m] = 0.5 * np.log(((1 - prob) + eps) / (prob + eps))

        return train_confidence, test_confidence
