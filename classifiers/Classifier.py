import abc

import numpy as np


class ClassifierClass(abc.ABC):
    """
    Abstract class that models a generic classifier.
    """
    class Model(abc.ABC):
        """
        Abstract class that models the model of a generic classifier.
        """
        pass

    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray, **kwargs):
        self.training_data = training_data
        self.training_labels = training_labels

    @abc.abstractmethod
    def train_model(self, **kwargs) -> None:
        """
        Train the classifier using the training data.
        :param kwargs: optional additional arguments.
        """
        pass

    @abc.abstractmethod
    def classify(self, testing_data: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """
        Classifies the samples in :param testing_data.
        :param testing_data: the sample to classify
        :param priors: optional parameter useful to predict class labels.
        :return: the predicted labels
        """
        pass

    @abc.abstractmethod
    def get_scores(self):
        """
        Returns the scores produced by the classifiers.
        :return:
        """
        pass
