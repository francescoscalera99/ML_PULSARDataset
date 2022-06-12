from itertools import product

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

        def generate_scores(self, alpha_star, dtr, ltr, dte):
            if self._type == 'poly':
                return np.sum((alpha_star * ltr).reshape(1, dtr.shape[1]) @ (dtr.T @ dte+self.c)**self.d+self._csi, axis=0)
            elif self._type == 'RBF':
                pass
            else:
                raise RuntimeError("Unexpected value for self._type.\n"
                                   f"Expected either 'poly' or 'RBF', got {self._type} instead.")

        def function(self, x1: np.ndarray, x2: np.ndarray):
            if self._type == 'poly':
                return (vrow(x1) @ vcol(x2) + self.c)**self.d + self._csi
            elif self._type == 'RBF':
                return np.exp(-self.gamma * (x1-x2).T @ (x1-x2)) + self._csi
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
        self._kernel = SVM.Kernel(kwargs['kernel_params'], kwargs['kernel_type'], self._k**2)
        self._D = np.vstack((self.training_data, self._k * np.ones(self.training_labels.size)))
        self._scores = None
        if self._kernel is None:
            g_matrix = self._D.T @ self._D
        elif kwargs['kernel_type'] == 'poly':
            num_samples = training_labels.size
            g_matrix = (self.training_data.T @ self.training_data + self._kernel.c)**self._kernel.d + self._k**2
        else:
            # TODO: RBF
            pass
        self._H = (vcol(self.training_labels) * g_matrix) * vrow(self.training_labels)
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

    def _solve_dual(self) -> Model:
        num_samples = self.training_labels.size
        alpha0 = np.zeros(num_samples)
        pi_t_emp = np.sum(self.training_labels == 1) / self.training_labels.size
        pi_f_emp = np.sum(self.training_labels == -1) / self.training_labels.size
        pi_t = 0.5
        c_t = self._C * pi_t / pi_t_emp
        c_f = self._C * (1-pi_t) / pi_f_emp
        bounds = [(0, c_t) if label == 1 else (0, c_f) for label in self.training_labels]
        alpha_star, _, _ = scipy.optimize.fmin_l_bfgs_b(self._neg_dual, x0=alpha0, bounds=bounds, factr=1.0, maxiter=10000)
        coefficients = self.training_labels * alpha_star
        w_star = vcol(np.sum(coefficients * self._D, axis=1))
        return self.Model(w_star, alpha_star)

    def train_model(self):
        self._model = self._solve_dual()

    def classify(self, testing_data, priors: np.ndarray):
        if self._kernel is None:
            w = self._model.w
            b = self._model.b
            self._scores = w.T @ testing_data + self._k * b
        else:
            alpha = self._model.alpha
            self._scores = self._kernel.generate_scores(alpha, self.training_data, self.training_labels, testing_data)

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


def getScore_kernelRBFSVM(alpha_star, DTR, LTR, DTE, K, gamma):
  k = np.zeros((DTR.shape[1], DTE.shape[1]))
  for i in range(DTR.shape[1]):
    for j in range(DTE.shape[1]):
      k[i, j] = np.exp(-gamma*(np.linalg.norm(DTR[:, i]- DTE[:, j])**2))+K**2
