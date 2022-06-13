import itertools

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

from PCA import PCA
from classifiers.Classifier import ClassifierClass
from utils.matrix_utils import vrow, vcol
from utils.metrics_utils import compute_min_DCF
from utils.utils import k_fold


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


def tuning_parameters_PolySVM(training_data, training_labels):
    titles_Kfold = ['Gaussianized feature (5-fold, no PCA)', 'Guassianized feature (5-fold, PCA = 7)', 'Gaussianized feature (5-fold, PCA = 5)']

    datasets = []

    training_dataPCA7 = PCA(training_data, 7)
    training_dataPCA5 = PCA(training_data, 5)
    datasets.append(training_data)
    datasets.append(training_dataPCA7)
    datasets.append(training_dataPCA5)
    C_values = np.logspace(-3, 3, 20)
    K_values = [0.0, 1.0]
    c_values = [0, 1, 10, 15]

    hyperparameters = itertools.product(c_values, K_values)
    j = 0
    for dataset in datasets:
        i = 0
        plt.figure()
        plt.rcParams['text.usetex'] = True
        for c, K in hyperparameters:
            DCFs = []
            for C in C_values:
                llrs, evaluationLabels = k_fold(dataset, training_labels, SVM, 5, k=K, c=C, kernel_params=(2, c), kernel_type='poly')
                print(llrs)
                min_dcf = compute_min_DCF(llrs, evaluationLabels, 0.5, 1, 1)
                print("min_DCF for C = ", C, "with c = ", c, "and K =", K, "->", min_dcf )
                DCFs.append(min_dcf)
            # f"prior:0.5, c:{c}, K:{K}"
            plt.plot(C_values, DCFs, color=np.random.rand(3,), label=r"$\pi_{}T=0.5$, c="+str(c)+r", K="+str(K))
        plt.title(titles_Kfold[j])
        j += 1
        plt.legend()
        plt.xscale('log')
        plt.show()


def tuning_parameters_LinearSVMBalanced(training_data, training_labels):
    titles_Kfold = ['Gaussianized feature (5-fold, no PCA)', 'Guassianized feature (5-fold, PCA = 7)',
                    'Gaussianized feature (5-fold, PCA = 5)']

    datasets = []

    training_dataPCA7 = PCA(training_data, 7)
    training_dataPCA5 = PCA(training_data, 5)
    datasets.append(training_data)
    datasets.append(training_dataPCA7)
    datasets.append(training_dataPCA5)
    C_values = np.logspace(-2, 2, 20)
    K_values = [1.0, 10.0]
    priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(K_values, priors)
    j = 0
    for dataset in datasets:
        plt.figure()
        plt.rcParams['text.usetex'] = True
        for K, p in hyperparameters:
            DCFs = []
            for C in C_values:
                llrs, evaluationLabels = k_fold(dataset, training_labels, SVM, 5, k=K, c=C, kernel_params=(2, 0),
                                                kernel_type='poly')
                print(llrs)
                min_dcf = compute_min_DCF(llrs, evaluationLabels, p, 1, 1)
                print("min_DCF for K = ", K, "with prior = ", p, "->", min_dcf)
                DCFs.append(min_dcf)
            # f"prior:0.5, c:{c}, K:{K}"
            plt.plot(C_values, DCFs, color=np.random.rand(3, ), label=r"$\pi_{T}=0.5$, K=" + str(K) + r", $\widetilde(\pi)$=" + str(p))
        plt.title(titles_Kfold[j])
        j += 1
        plt.legend()
        plt.xscale('log')
        plt.show()
