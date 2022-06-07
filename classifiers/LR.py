import numpy as np
from scipy import optimize as opt

from classifiers.Classifier import ClassifierClass


class LR(ClassifierClass):
    class Model(ClassifierClass.Model):
        def __init__(self, w: np.ndarray, b: float):
            self.w = w
            self.b = b

    def __init__(self, training_data, training_labels, lbd, pi_t: float):
        super().__init__(training_data, training_labels)
        self._lambda = lbd
        self._model = None
        self._pi_t = pi_t
        self._scores = None

    def objective_function(self, v):
        w, b = v[0:-1], v[-1]
        z = 2 * self.training_labels - 1
        x = self.training_data
        regularization_term = (self._lambda / 2) * (w.T @ w)
        nf = len(self.training_labels == 0)
        nt = len(self.training_labels == 1)
        second_term_t = self._pi_t/nt * np.logaddexp(0, -z[self.training_labels == 1] * (w.T @ x[:, self.training_labels == 1] + b)).mean()
        second_term_f = (1-self._pi_t)/nf * np.logaddexp(0, -z[self.training_labels == 0] * (w.T @ x[:, self.training_labels == 0] + b)).mean()
        return regularization_term + second_term_t + second_term_f

    def train_model(self) -> None:
        x0 = np.zeros(self.training_data.shape[0] + 1)
        v, _, _ = opt.fmin_l_bfgs_b(self.objective_function, x0=x0, approx_grad=True)

        w, b = v[0:-1], v[-1]
        self._model = LR.Model(w, b)

    def classify(self, testing_data: np.ndarray, priors: np.ndarray) -> np.ndarray:
        self._scores = np.dot(self._model.w.T, testing_data) + self._model.b
        predicted_labels = (self._scores > 0).astype(int)
        return predicted_labels

    def get_llrs(self):
        return self._scores
