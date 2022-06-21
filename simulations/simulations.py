import itertools
import numpy as np
from prettytable import PrettyTable

from classifiers.Classifier import ClassifierClass
from classifiers.LR import LR
from classifiers.MVG import MVG
from classifiers.SVM import SVM
from utils.metrics_utils import compute_actual_DCF, compute_min_DCF
from utils.utils import k_fold


class Simulator:
    def __init__(self, training_data: np.ndarray,
                 training_labels: np.ndarray,
                 classifier_class: ClassifierClass,
                 hyperparameters: dict,
                 seed=0,
                 **kwargs):
        """
        Class useful to simulate different classifiers with different parameters and hyperparameters

        :param training_data: the training set
        :param training_labels: the training labels
        :param classifier_class: the reference to the classifier class
        :param hyperparameters: the dictionary of all possible hyperparameters for the current classifier
        :param seed: the seed on which is based the shuffle in the k-fold procedure
        :param kwargs: the dictionary of all possible configuration parameters for the current classifier and simulation
        """

        self.training_data = training_data
        self.training_labels = training_labels
        self.classifier = classifier_class
        self.seed = seed
        self.hyperparameters = self._validate_hyperparameters(hyperparameters)
        self.kwargs = self._validate_kwargs(kwargs)

    def _validate_hyperparameters(self, hyperparameters: dict) -> dict:
        """
        Checks if all the passed hyperparameters are ok and returns the same dictionary

         possible hyperparameters:
        * m_pca: int ->               (ALL)       mandatory
        * pi: float                   (ALL)       mandatory
        * pi_tilde: float             (ALL)       mandatory
        * variant: str           (MVG)       optional
        * svm_c: float                (SVM)       optional
        """
        for v in hyperparameters.values():
            if type(v) is not list:
                raise TypeError("Hyperparameters must be lists of values.")

        if 'm_pca' not in hyperparameters.keys() or 'pi' not in hyperparameters.keys() or 'pi_tilde' not in hyperparameters.keys():
            raise KeyError("Parameters 'm_pca' (int or None), 'pi' (float) and 'pi_tilde' (float) are mandatory.")

        if not all(m is None or isinstance(m, int) for m in hyperparameters['m_pca']):
            raise TypeError("Parameters 'm_pca' must be either None or integers.")
        if not all(isinstance(p, float) for p in hyperparameters['pi']):
            raise TypeError("Parameters 'pi' must be floating-point numbers.")
        if not all(isinstance(p, float) for p in hyperparameters['pi_tilde']):
            raise TypeError("Parameters 'pi_tilde' must be floating-point numbers.")

        if self.classifier is MVG:
            if 'variant' not in hyperparameters.keys():
                raise KeyError("Parameter 'variant' (str) is mandatory for MVG simulations.")

            if not all(isinstance(variant, str) and variant in ['full-cov', 'diag', 'tied'] for variant in hyperparameters['variant']):
                raise ValueError("Parameters 'variant' must be strings among ['full-cov', 'diag', 'tied']")

        if self.classifier is SVM:
            if 'svm_c' not in hyperparameters.keys():
                raise KeyError("Parameter 'svm_c' (float) is mandatory for SVM simulations.")
            if not all(isinstance(c, float) for c in hyperparameters['svm_c']):
                raise TypeError("Parameters 'svm_c' must be floating-point numbers.")

        # end of method -> validation OK
        return hyperparameters

    def _validate_kwargs(self, kwargs: dict) -> dict:
        """
        Checks if all the passed kwargs are ok and returns the same dictionary

         possible kwargs:
        * calibrate: bool             (ALL)       mandatory
        * actual_dcf: bool            (ALL)       mandatory
        * num_folds: int              (ALL)       mandatory
        * lr_lambda: float (LR lambda)   (LR)        optional
        * svm_ker_type: str           (SVM ALL)   optional (mandatory for SVM)
        * svm_k: float                (SVM ALL)   optional (mandatory for SVM)
        * svm_c_d: tuple[float, int]  (SVM POLY)  optional
        * svm_gamma: float            (SVM RBF)   optional
        """

        if 'calibrate' not in kwargs.keys() or 'actual_dcf' not in kwargs.keys() or 'num_folds' not in kwargs.keys():
            raise KeyError("kwargs 'calibrate' (bool), 'num_folds' (int) and 'actual_dcf' (bool) are mandatory.")

        if type(kwargs['calibrate']) is not bool:
            raise TypeError("kwarg 'calibrate' must be boolean.")
        if type(kwargs['actual_dcf']) is not bool:
            raise TypeError("kwarg 'actual_dcf' must be boolean.")
        if type(kwargs['num_folds']) is not int:
            raise TypeError("kwarg 'num_folds' must be integer.")

        if self.classifier is LR:
            if 'lr_lambda' not in kwargs.keys():
                raise KeyError("kwarg 'lr_lambda' is mandatory for LR simulations")
            if type(kwargs['lr_lambda']) is not float and type(kwargs['lr_lambda']) is not int:
                raise TypeError("kwarg 'lr_lambda' must be numeric.")

        if self.classifier is SVM:
            if 'svm_ker_type' not in kwargs.keys() or 'svm_k' not in kwargs.keys():
                raise KeyError("kwargs 'svm_ker_type' (str) and 'svm_k' (float) are mandatory for SVM simulations.")
            if type(kwargs['svm_k']) is not float and type(kwargs['svm_k']) is not int:
                raise TypeError("kwarg 'svm_k' must be numeric.")

            if kwargs['svm_ker_type'] == 'poly':
                if 'svm_c_d' not in kwargs.keys():
                    raise KeyError(
                        "kwarg 'svm_c_d' (tuple[float, int]) is mandatory for polynomial-kernel SVM simulations.")
                if type(kwargs['svm_c_d']) is not tuple or \
                        (type(kwargs['svm_c_d'][0]) is not float and type(kwargs['svm_c_d'][0]) is not int) \
                        or type(kwargs['svm_c_d'][1] is not int):
                    raise TypeError("kwarg 'svm_c_d' must be a tuple[float, int]")

            elif kwargs['svm_ker_type'] == 'rbf':
                if 'svm_gamma' not in kwargs.keys():
                    raise KeyError("kwarg 'svm_gamma' (float) is mandatory for RBF-kernel SVM simulations.")
                if type(kwargs['svm_gamma']) is not float and type(kwargs['svm_gamma']) is not int:
                    raise TypeError("kwarg 'svm_gamma' must be a floating-point value.")
            else:
                raise ValueError("kwarg 'svm_ker_type' can only be either 'poly' or 'rbf'")

        # end of method -> validation OK
        return kwargs

    def simulate(self):
        pass


def MVG_simulations(training_data, training_labels, calibrateScore=False, actualDCF=False):
    variants = ['full-cov', 'diag', 'tied']
    m = [False, None, 7, 5, 4]
    pis = [0.1, 0.5, 0.9]

    hyperparameters = itertools.product(variants, m, pis)
    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for variant, m, pi in hyperparameters:
        if m == False:  # raw features
            llrs, labels = k_fold(training_data, training_labels, MVG, 5, seed=0, m=None, raw=True, variant=variant)
        else:
            llrs, labels = k_fold(training_data, training_labels, MVG, 5, seed=0, m=m, raw=False, variant=variant)
        if actualDCF:
            if calibrateScore:
                score = scoreCalibration(llrs, labels)
            else:
                score = llrs
            actDCF = compute_actual_DCF(score, labels, pi, 1, 1)
            table.add_row([f"PCA m={m}, variant={variant}, π_tilde={pi} -> actual dcf", round(actDCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
            table.add_row([f"PCA m={m}, variant={variant}, π_tilde={pi} -> min dcf", round(min_dcf, 3)])
    print(table)


def LR_simulations(training_data, training_labels, lbd, calibrateScore=False, actualDCF=False):
    m_values = [False, None, 7, 5]
    pis_T = [0.5, 0.1, 0.9]
    pis = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m_values, pis, pis_T)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi, pi_T in hyperparameters:
        if m == False:  # raw features
            llrs, labels = k_fold(training_data, training_labels, LR, 5, m=None, raw=True, seed=0, lbd=lbd, pi_T=pi_T)
        else:
            llrs, labels = k_fold(training_data, training_labels, LR, 5, m=m, raw=False, seed=0, lbd=lbd, pi_T=pi_T)
        if actualDCF:
            if calibrateScore:
                score = scoreCalibration(llrs, labels)
            else:
                score = llrs
            actDCF = compute_actual_DCF(score, labels, pi, 1, 1)
            table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T} -> actual dcf:", round(actDCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
            table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T} -> min dcf:", round(min_dcf, 3)])

    print(table)


def SVM_LinearBalancedSimulations(training_data, training_labels, K, C_piT, calibrateScore=False, actualDCF=False):
    m = [None, 7, 5]
    priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, C_piT, priors)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, (C, pi_T), pi in hyperparameters:
        if pi_T is None:
            llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=False, m=m, raw=False,
                                  pi_T=pi_T, k=K, c=C,
                                  kernel_params=(1, 0), kernel_type='poly')
        else:
            llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=True, m=m, raw=False,
                                  pi_T=pi_T, k=K, c=C,
                                  kernel_params=(1, 0), kernel_type='poly')
        if actualDCF:
            if calibrateScore:
                score = scoreCalibration(llrs, labels)
            else:
                score = llrs
            actDCF = compute_actual_DCF(score, labels, pi, 1, 1)
            table.add_row([f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", round(actDCF, 3)])
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        print(f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(min_dcf, 3))
        table.add_row([f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])

    print(table)


def SVM_PolySimulations(training_data, training_labels, K, C, pi_T, c, d):
    m = [None, 7, 5]
    priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, priors)
    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi in hyperparameters:
        llrs, evaluationLabels = k_fold(training_data, training_labels, SVM, 5, m=m, raw=False, k=K, c=C, balanced=True,
                                        pi_T=pi_T,
                                        kernel_params=(d, c), kernel_type='poly')
        min_dcf = compute_min_DCF(llrs, evaluationLabels, 0.5, 1, 1)
        print(f"PCA m={m}, data: gaussianized, π_tilde={pi}, pi_T = 0.5, C ={C} K={K}, c={c}, d={d}", "-->",
              round(min_dcf, 3))
        table.add_row(
            [f"PCA m={m}, data: gaussianized, π_tilde={pi}, pi_T = 0.5, C ={C} K={K}, c={c}, d={d}", round(min_dcf, 3)])

    print(table)


def SVM_RBFSimulations(training_data, training_labels, K, C, pi_T, gamma):
    m = [None, 7, 5]
    priors = [0.5, 0.1, 0.9]
    hyperparameters = itertools.product(m, priors)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi in hyperparameters:
        llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, m=m, raw=False, balanced=True, pi_T=pi_T,
                              k=K, c=C,
                              kernel_params=gamma, kernel_type='RBF')
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        print(f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(min_dcf, 3))
        table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])

    print(table)


if __name__ == '__main__':
    pass
