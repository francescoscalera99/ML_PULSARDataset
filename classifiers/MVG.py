import numpy as np
import scipy.special

from Classifier import Classifier
from utils import covariance_matrix_mean, vrow


class MVG(Classifier):
    class Model(Classifier.Model):
        def __init__(self, mu0, c0, mu1, c1):
            self.mu0 = mu0
            self.c0 = c0
            self.mu1 = mu1
            self.c1 = c1

    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray):
        super().__init__(training_data, training_labels)
        self._model = None

    def train_model(self) -> None:
        mu0, c0 = covariance_matrix_mean(self.training_data[:, self.training_labels == 0])
        mu1, c1 = covariance_matrix_mean(self.training_data[:, self.training_labels == 1])

        self._model = MVG.Model(mu0, c0, mu1, c1)

    def logpdf_GAU_1sample(self, x, mu, C):
        M = mu.shape[0]
        logDet_E = np.linalg.slogdet(C)[1]

        return -(M / 2) * np.log(2 * np.pi) - 0.5 * logDet_E - 0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x - mu))

    def logpdf_GAU_ND(self, X, mu, C):
        Y = [self.logpdf_GAU_1sample(X[:, i:i + 1], mu, C) for i in range(X.shape[1])]
        return np.array(Y).ravel()

    def classify(self, testing_data: np.ndarray, priors: np.ndarray) -> np.ndarray:
        class0_conditional_probability = self.logpdf_GAU_ND(testing_data, self._model.mu0, self._model.c0)
        class1_conditional_probability = self.logpdf_GAU_ND(testing_data, self._model.mu1, self._model.c1)
        logScores = np.vstack(class0_conditional_probability.ravel(), class1_conditional_probability.ravel())
        logJoint = logScores + np.log(priors)
        logMarginal = vrow(scipy.special.logsumexp(logJoint, axis=0))
        logPost = logJoint - logMarginal
        class_posterior_probabilities = np.exp(logPost)
        predicted_labels = np.argmax(class_posterior_probabilities, axis = 0)
        return predicted_labels


def main():
    pass


if __name__ == '__main__':
    main()
