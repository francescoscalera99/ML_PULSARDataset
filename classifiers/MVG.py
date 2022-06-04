import numpy as np
import scipy.special

from classifiers.Classifier import ClassifierClass
from utils import covariance_matrix_mean, vrow, vcol


class MVG(ClassifierClass):
    class Model(ClassifierClass.Model):
        def __init__(self, mu0, c0, mu1, c1=None, variant='full-cov'):
            self.mu0 = mu0
            self.mu1 = mu1
            if variant == 'tied':
                self.c = c0
            else:
                self.c0 = c0
                self.c1 = c1

    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray, variant='full-cov'):
        super().__init__(training_data, training_labels)
        self._model = None
        self.variant = variant

    def train_model(self) -> None:
        c0, mu0 = covariance_matrix_mean(self.training_data[:, self.training_labels == 0])
        c1, mu1 = covariance_matrix_mean(self.training_data[:, self.training_labels == 1])

        if self.variant == 'tied':
            c = tied_covariance_matrix(self.training_data, self.training_labels)
            self._model = MVG.Model(mu0, c, mu1, variant=self.variant)

        elif self.variant == 'diag':
            pass
        else:
            self._model = MVG.Model(mu0, c0, mu1, c1, variant=self.variant)

    def classify(self, testing_data: np.ndarray, priors: np.ndarray) -> np.ndarray:
        class0_conditional_probability = self.logpdf_GAU_ND(testing_data, self._model.mu0, self._model.c0)
        class1_conditional_probability = self.logpdf_GAU_ND(testing_data, self._model.mu1, self._model.c1)
        logScores = np.vstack((class0_conditional_probability.ravel(), class1_conditional_probability.ravel()))
        logJoint = logScores + vcol(np.log(priors))
        logMarginal = vrow(scipy.special.logsumexp(logJoint, axis=0))
        logPost = logJoint - logMarginal
        class_posterior_probabilities = np.exp(logPost)
        predicted_labels = np.argmax(class_posterior_probabilities, axis=0)
        return predicted_labels

    def logpdf_GAU_1sample(self, x, mu, C):
        M = mu.shape[0]
        logDet_E = np.linalg.slogdet(C)[1]

        return -(M / 2) * np.log(2 * np.pi) - 0.5 * logDet_E - 0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(C)),
                                                                            (x - mu))

    def logpdf_GAU_ND(self, X, mu, C):
        Y = [self.logpdf_GAU_1sample(X[:, i:i + 1], mu, C) for i in range(X.shape[1])]
        return np.array(Y).ravel()


def tied_covariance_matrix(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    C0 = covariance_matrix_mean(D0)[0]
    C1 = covariance_matrix_mean(D1)[0]
    C2 = covariance_matrix_mean(D2)[0]

    SW = D0.shape[1] * C0 + D1.shape[1] * C1 + D2.shape[1] * C2
    return SW / D.shape[1]


def main():
    pass


if __name__ == '__main__':
    main()
