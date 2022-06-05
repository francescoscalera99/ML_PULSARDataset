import numpy as np
import scipy.special

from classifiers.Classifier import ClassifierClass
from utils.matrix_utils import covariance_matrix_mean, vrow, vcol


class MVG(ClassifierClass):
    class Model(ClassifierClass.Model):
        def __init__(self,
                     mu0: np.ndarray,
                     c0: np.ndarray,
                     mu1: np.ndarray,
                     c1: np.ndarray = None,
                     variant='full-cov'):
            self.mu0 = mu0
            self.mu1 = mu1
            if variant == 'tied':
                self.c = c0
            elif variant == 'diag':
                self.c0 = c0 * np.eye(c0.shape[0])
                self.c1 = c1 * np.eye(c1.shape[0])
            else:
                self.c0 = c0
                self.c1 = c1

    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray, variant='full-cov'):
        super().__init__(training_data, training_labels)
        self._model = None
        self.variant = variant
        self._score_matrix = None

    def train_model(self) -> None:
        c0, mu0 = covariance_matrix_mean(self.training_data[:, self.training_labels == 0])
        c1, mu1 = covariance_matrix_mean(self.training_data[:, self.training_labels == 1])

        if self.variant == 'tied':
            c = within_class_variability(self.training_data, self.training_labels)
            self._model = MVG.Model(mu0, c, mu1, variant=self.variant)
        else:
            self._model = MVG.Model(mu0, c0, mu1, c1, variant=self.variant)

    def classify(self, testing_data: np.ndarray, priors: np.ndarray) -> np.ndarray:
        if self.variant == 'tied':
            class0_conditional_probability = self.logpdf_GAU_ND(testing_data, vcol(self._model.mu0), self._model.c)
            class1_conditional_probability = self.logpdf_GAU_ND(testing_data, vcol(self._model.mu1), self._model.c)
        else:
            class0_conditional_probability = self.logpdf_GAU_ND(testing_data, self._model.mu0, self._model.c0)
            class1_conditional_probability = self.logpdf_GAU_ND(testing_data, self._model.mu1, self._model.c1)
        logScores = np.vstack((class0_conditional_probability, class1_conditional_probability))
        logJoint = logScores + vcol(np.log(priors))
        logMarginal = vrow(scipy.special.logsumexp(logJoint, axis=0))
        logPost = logJoint - logMarginal
        class_posterior_probabilities = np.exp(logPost)
        predicted_labels = np.argmax(class_posterior_probabilities, axis=0)
        self._score_matrix = logScores
        return predicted_labels

    def logpdf_GAU_1sample(self, x, mu, C):
        M = mu.shape[0]
        logDet_E = np.linalg.slogdet(C)[1]

        return -(M / 2) * np.log(2 * np.pi) - 0.5 * logDet_E - 0.5 * np.dot(np.dot((vcol(x) - vcol(mu)).T, np.linalg.inv(C)),
                                                                            (vcol(x) - vcol(mu)))

    def logpdf_GAU_ND(self, X, mu, C):
        Y = [self.logpdf_GAU_1sample(X[:, i:i + 1], mu, C) for i in range(X.shape[1])]
        return np.array(Y).ravel()

    def get_llrs(self):
        return self._score_matrix[1, :] - self._score_matrix[0, :]


def tied_covariance_matrix(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    C0 = covariance_matrix_mean(D0)[0]
    C1 = covariance_matrix_mean(D1)[0]

    SW = D0.shape[1] * C0 + D1.shape[1] * C1
    return SW / D.shape[1]


def within_class_variability(dataset, labels) -> np.ndarray:
    """
    Evaluates the within class covariance given the labeled dataset

    :param dataset: the dataset (unlabeled)
    :param labels: the labels of :param dataset
    :return: the within class covariance matrix
    """
    num_classes = len(np.unique(labels))
    n_features, n_samples = dataset.shape
    sw = np.zeros((n_features, n_features))

    for i in range(num_classes):
        class_data = dataset[:, labels == i]
        c_mean = class_mean(dataset, labels, i)

        class_data = class_data - vcol(c_mean)
        sw = sw + class_data @ class_data.T

    return sw / n_samples


def class_mean(dataset, labels, c):
    data = dataset[:, labels == c]
    return np.mean(data, axis=1)


def main():
    pass


if __name__ == '__main__':
    main()
