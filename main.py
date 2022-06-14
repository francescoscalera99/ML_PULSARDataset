import itertools
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from PCA import PCA
from classifiers.LR import LR
from classifiers.MVG import MVG
from classifiers.SVM import SVM
from utils.plot_utils import plot_histogram, create_heatmap
from utils.utils import load_dataset, gaussianize, splitData_SingleFold, k_fold
from utils.metrics_utils import compute_min_DCF


def MVG_simulations(training_data, training_labels):
    variants = ['full-cov', 'diag', 'tied']
    m = [None, 7, 5, 4]
    pis = [0.1, 0.5, 0.9]

    hyperparameters = itertools.product(variants, m, pis)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for variant, m, pi in hyperparameters:
        if m is not None:
            training_data = PCA(training_data, m)

        llrs, labels = k_fold(training_data, training_labels, MVG, 5, seed=0, variant=variant)
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        table.add_row([f"PCA m={m}, data: gaussianized, variant={variant}, π_tilde={pi}", round(min_dcf, 3)])
    print(table)


def LR_simulations(training_data, training_labels, lbd):
    m = [None, 7, 5, 4]
    pis_T = [0.5, 0.1, 0.9]
    pis = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, pis, pis_T)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi, pi_T in hyperparameters:
        if m is not None:
            training_data = PCA(training_data, m)

        llrs, labels = k_fold(training_data, training_labels, LR, 5, seed=0, lbd=lbd, pi_T=pi_T)

        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  λ={lbd}", round(min_dcf, 3)])

    print(table)


def SVM_LinearSimulations(training_data, training_labels, K, C_piT):
    m = [None, 7, 5, 4]
    priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, C_piT, priors)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, (C, pi_T), pi in hyperparameters:
        if m is not None:
            training_data = PCA(training_data, m)
        if pi_T == None:
            llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=False, pi_T=pi_T, k=K, c=C, kernel_params=(1, 0), kernel_type='poly')
        else:
            llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=True, pi_T=pi_T, k=K, c=C, kernel_params=(1, 0), kernel_type='poly')
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        print(f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(min_dcf, 3))
        table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])

    print(table)


def SVM_PolySimulations(training_data, training_labels, K, C, pi_T, c, d):
    m = [None, 7, 5, 4]
    priors = [0.5, 0.1, 0.9]
    hyperparameters = itertools.product(m, priors)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi in hyperparameters:
        if m is not None:
            training_data = PCA(training_data, m)
        llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=True, pi_T=pi_T, k=K, c=C,
                                  kernel_params=(d, c), kernel_type='poly')
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        print(f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(min_dcf, 3))
        table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])

    print(table)


def main():
    (training_data, training_labels), (testing_data, testing_labels) = load_dataset()

    # =============== MULTIVARIATE GAUSSIAN CLASSIFIER ===============
    # MVG_simulations(training_data, training_labels)

    # =============== LOGISTIC REGRESSION ===============
    #find_optLambda(training_data, training_labels)
    #lbd = 1e-3
    #LR_simulations(training_data, training_labels, lbd)

    # =============== SUPPORT VECTOR MACHINE ===============
    #tuning_parameters_PolySVM(training_data, training_labels)
    #tuning_parameters_LinearSVMBalanced(training_data, training_labels)
    K = 1.0 #This values comes from tuning of hyperparameters
    #C_piT = [(1e-2, None), (1e-3, 0.5), (6 * 1e-3, 0.1), (7 * 1e-4, 0.9)] #These values comes from tuning of hyperparameter
    #SVM_LinearSimulations(training_data, training_labels, K, C_piT)
    pi_TPoly = 0.5
    CPoly = 5*1e-5
    c = 15
    d = 2
    SVM_PolySimulations(training_data, training_labels, K, CPoly, pi_TPoly, c, d)


if __name__ == '__main__':
    main()
