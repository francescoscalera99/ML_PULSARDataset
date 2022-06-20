import itertools
import os

import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from PCA import PCA
from classifiers.LR import LR
from classifiers.MVG import MVG
from classifiers.SVM import SVM, tuning_parameters_PolySVM, tuning_parameters_RBFSVM, \
    tuning_parameters_LinearSVMUnbalanced, tuning_parameters_LinearSVMBalanced
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
            dtr = PCA(training_data, m)
        else:
            dtr = training_data
        llrs, labels = k_fold(dtr, training_labels, MVG, 5, seed=0, variant=variant, balanced=False, pi_T=None)
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        table.add_row([f"PCA m={m}, data: gaussianized, variant={variant}, π_tilde={pi}", round(min_dcf, 3)])
    print(table)


def LR_simulations(training_data, training_labels, lbd):
    m = [None, 7, 5]
    pis_T = [0.5, 0.1, 0.9]
    pis = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, pis, pis_T)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi, pi_T in hyperparameters:
        if m is not None:
            dtr = PCA(training_data, m)
        else:
            dtr = training_data

        llrs, labels = k_fold(dtr, training_labels, LR, 5, seed=0, lbd=lbd, pi_T=pi_T)

        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}", round(min_dcf, 3)])

    print(table)


def SVM_LinearSimulations(training_data, training_labels, K, C_piT):
    m = [None, 7, 5]
    priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, C_piT, priors)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, (C, pi_T), pi in hyperparameters:
        if m is not None:
            dtr = PCA(training_data, m)
        else:
            dtr = training_data
        if pi_T == None:
            llrs, labels = k_fold(dtr, training_labels, SVM, 5, seed=0, balanced=False, pi_T=pi_T, k=K, c=C,
                                  kernel_params=(1, 0), kernel_type='poly')
        else:
            llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=True, pi_T=pi_T, k=K, c=C,
                                  kernel_params=(1, 0), kernel_type='poly')
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        print(f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(min_dcf, 3))
        # table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])

    print(table)


# def SVM_PolySimulations(training_data, training_labels, K, C, pi_T, c, d):
#     m = [None, 7, 5]
#     priors = [0.5, 0.1, 0.9]
#     hyperparameters = itertools.product(m, priors)
#
#     table = PrettyTable()
#     table.field_names = ['Hyperparameters', 'min DCF']
#
#     for m, pi in hyperparameters:
#         if m is not None:
#             dtr = PCA(training_data, m)
#         else:
#             dtr = training_data
#         llrs, labels = k_fold(dtr, training_labels, SVM, 5, seed=0, balanced=True, pi_T=pi_T, k=K, c=C,
#                               kernel_params=(d, c), kernel_type='poly')
#         min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
#         print(f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(min_dcf, 3))
#         table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])
#
#     print(table)
#
#
# def SVM_RBFSimulations(training_data, training_labels, K, C, pi_T, gamma):
#     m = [None, 7, 5]
#     priors = [0.5, 0.1, 0.9]
#     hyperparameters = itertools.product(m, priors)
#
#     table = PrettyTable()
#     table.field_names = ['Hyperparameters', 'min DCF']
#
#     for m, pi in hyperparameters:
#         if m is not None:
#             dtr = PCA(training_data, m)
#         else:
#             dtr = training_data
#         llrs, labels = k_fold(dtr, training_labels, SVM, 5, seed=0, balanced=True, pi_T=pi_T, k=K, c=C,
#                               kernel_params=gamma, kernel_type='RBF')
#         min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
#         print(f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(min_dcf, 3))
#         table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])
#
#     print(table)


def main():
    (training_data, training_labels), (testing_data, testing_labels) = load_dataset()
    # titles = ['1. Mean of the integrated profile',
    #           '2. Standard deviation of the integrated profile',
    #           '3. Excess kurtosis of the integrated profile',
    #           '4. Excess kurtosis of the integrated profile',
    #           '5. Mean of the DM-SNR curve',
    #           '6. Standard deviation of the DM-SNR curve',
    #           '7. Excess kurtosis of the DM-SNR curve',
    #           '8. Skewness of the DM-SNR curve']

    # =============== FEATURE ANALYSIS ===============
    # plot_histogram(training_data, training_labels, titles)
    # create_heatmap(training_data, training_labels)

    # =============== MULTIVARIATE GAUSSIAN CLASSIFIER ===============
    # MVG_simulations(training_data, training_labels)

    # =============== LOGISTIC REGRESSION ===============
    # find_optLambda(training_data, training_labels)
    # lbd = 1e-3
    # LR_simulations(training_data, training_labels, lbd)

    # =============== SUPPORT VECTOR MACHINE ===============
    # print("LINEAR SVM - TUNING PARAMETERS")
    # tuning_parameters_LinearSVMUnbalanced(training_data, training_labels)
    # tuning_parameters_LinearSVMBalanced(training_data, training_labels)
    # print("POLY SVM - TUNING PARAMETERS")
    # tuning_parameters_PolySVM(training_data, training_labels)
    # print("RBF SVM - TUNING PARAMETERS")
    tuning_parameters_RBFSVM(training_data, training_labels)
    # tuning_parameters_LinearSVMBalanced(training_data, training_labels)
    # K_Linear = 1.0 #This values comes from tuning of hyperparameters
    # C_piT_Linear = [(1e-2, None), (1e-3, 0.5), (6 * 1e-3, 0.1), (7 * 1e-4, 0.9)] #These values comes from tuning of hyperparameter
    # SVM_LinearSimulations(training_data, training_labels, K_Linear, C_piT_Linear)
    # K_Poly = 1.0
    # pi_TPolyRBF = 0.5
    # CPoly = 5*1e-5
    # c = 15
    # d = 2
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d)
    # K_RBF = 1.0
    # gamma_RBF = 1e-3
    # C_RBF = 1e-1
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF)

    # =============== GAUSSIAN MIXTURE MODELS ===============

    # ****************** TURN OFF PC AT END OF SIMULATION (needs sudo) ******************
    # (windows ?)
    # os.system("shutdown /s /t 1")
    # MAC
    # os.system("shutdown -h now")


def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    _, detC = np.linalg.slogdet(C)
    invC = np.linalg.inv(C)
    return np.diag(
        -(M / 2) * np.log(2 * np.pi) - (1 / 2) * (detC) - (1 / 2) * np.dot(np.dot((x - mu).T, invC), (x - mu)))


if __name__ == '__main__':
    main()
