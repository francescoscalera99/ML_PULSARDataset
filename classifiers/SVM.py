import numpy as np
import scipy.optimize

from classifiers.Classifier import ClassifierClass
from utils.matrix_utils import vrow, vcol


class SVM(ClassifierClass):
    class Model:
        def __init__(self, w_star, alpha_star):
            self.w = w_star[:-1]
            self.b = w_star[-1]
            self.alpha = alpha_star

        def __str__(self):
            return f"Parameters:\nw:\n{self.w}\nb:{self.b}\nalpha:{self.alpha}"

        def get(self):
            return self.w, self.b, self.alpha

    class Kernel:
        def __init__(self, parameters, kernel_type, csi=0):
            self._csi = csi
            self._type = kernel_type
            if self._type == 'poly':
                self.d, self.c = parameters
            elif self._type == 'RBF':
                self.gamma = float(parameters)
            else:
                raise TypeError("Argument 'parameters' can either be a tuple of numbers or a number")

        def __str__(self):
            if self._type == 'poly':
                return f"Poly (d={self.d}, c={self.c})"
            elif self._type == 'RBF':
                return f"RBF (γ={self.gamma})"
            else:
                raise RuntimeError("Unexpected value for self._type.\n"
                                   f"Expected either 'poly' or 'RBF', got {self._type} instead.")

        def function(self, x1: np.ndarray, x2: np.ndarray):
            """
            Implementation of the (possibly regularized) kernel function of the SVM,
            chosen depending on self._type parameter.
            :param x1: the first dataset
            :param x2: the second dataset
            :return: the kernel matrix of size NxN
            """
            if self._type == 'poly':
                return (x1.T @ x2 + self.c) ** self.d + self._csi
            elif self._type == 'RBF':
                k = x1.reshape((*x1.shape, 1)) - x2.reshape((x2.shape[0], 1, x2.shape[1]))
                k = -self.gamma * np.linalg.norm(k, axis=0)**2
                return np.exp(k) + self._csi
            else:
                raise RuntimeError("Unexpected value for self._type.\n"
                                   f"Expected either 'poly' or 'RBF', got {self._type} instead.")

    def __init__(self, training_data, training_labels, **kwargs):
        if set(training_labels) == {0, 1}:
            training_labels[training_labels == 0] = - 1
        super().__init__(training_data, training_labels)
        self._k = kwargs['k']
        self._C = kwargs['c']
        self._model = None
        self._kernel = SVM.Kernel(kwargs['kernel_params'], kwargs['kernel_type'], self._k ** 2)
        self._D = np.vstack((self.training_data, self._k * np.ones(self.training_labels.size)))
        self._scores = None
        self._H = ((vcol(self.training_labels)
                   * self._kernel.function(self.training_data, self.training_data))
                   * vrow(self.training_labels))
        print('finished constructor')

    def _primal(self, w, b):
        w_ = np.vstack((vcol(w), b))
        first_term = 0.5 * w_.T @ w_
        second_term = 1 - self.training_labels * (w_.T @ self._D)
        second_term[second_term < 0] = 0
        second_term = self._C * np.sum(second_term)
        return first_term + second_term

    def _neg_dual(self, alpha):
        dual_objective_neg = 0.5 * alpha.T @ self._H @ alpha - np.sum(alpha)
        gradient = self._H @ alpha - 1
        return dual_objective_neg, vrow(gradient)

    def _solve_dual(self, balanced) -> Model:
        num_samples = self.training_labels.size
        alpha0 = np.zeros(num_samples)
        if balanced:
            pi_t_emp = np.sum(self.training_labels == 1) / self.training_labels.size
            pi_f_emp = np.sum(self.training_labels == -1) / self.training_labels.size
            pi_t = 0.5
            c_t = self._C * pi_t / pi_t_emp
            c_f = self._C * (1 - pi_t) / pi_f_emp
            bounds = [(0, c_t) if label == 1 else (0, c_f) for label in self.training_labels]
        else:
            bounds = [(0, self._C)] * num_samples
        alpha_star, _, _ = scipy.optimize.fmin_l_bfgs_b(self._neg_dual, x0=alpha0, bounds=bounds, factr=1.0)
        coefficients = self.training_labels * alpha_star
        w_star = vcol(np.sum(coefficients * self._D, axis=1))
        return self.Model(w_star, alpha_star)

    def train_model(self, balanced=False):
        self._model = self._solve_dual(balanced)

    def classify(self, testing_data, priors: np.ndarray):
        if self._kernel is None:
            w = self._model.w
            b = self._model.b
            self._scores = w.T @ testing_data + self._k * b
        else:
            self._scores = (self._model.alpha * self.training_labels).reshape(1, self.training_data.shape[1]) \
                           @ self._kernel.function(self.training_data, testing_data)

        predictions = np.array(self._scores > 0, dtype=int)
        print("Finished classification")
        return predictions

    def get_llrs(self):
        return self._scores

    def compute_duality_gap(self) -> tuple:
        w = self._model.w
        b = self._model.b
        alpha = self._model.alpha

        if self._kernel is None:
            p = float(self._primal(w, b))
        else:
            p = 0
        d = float(self._neg_dual(alpha)[0])

        return p, d, p + d
