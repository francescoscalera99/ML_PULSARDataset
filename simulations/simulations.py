import itertools
import numpy as np
from prettytable import PrettyTable

from classifiers.GMM import GMM
from classifiers.LR import LR, calibrateScores
from classifiers.MVG import MVG
from classifiers.SVM import SVM
from utils.metrics_utils import compute_actual_DCF, compute_min_DCF
from utils.utils import k_fold


# def MVG_simulations(training_data, training_labels, calibrateScore=False, actualDCF=False):
#     variants = ['full-cov', 'diag', 'tied']
#     m = [False, None, 7, 5, 4]
#     pis = [0.1, 0.5, 0.9]
#
#     hyperparameters = itertools.product(variants, m, pis)
#     table = PrettyTable()
#     table.field_names = ['Hyperparameters', 'min DCF']
#
#     for variant, m, pi in hyperparameters:
#         if m == False:  # raw features
#             llrs, labels = k_fold(training_data, training_labels, MVG, 5, seed=0, m=None, raw=True, variant=variant)
#         else:
#             llrs, labels = k_fold(training_data, training_labels, MVG, 5, seed=0, m=m, raw=False, variant=variant)
#         if actualDCF:
#             if calibrateScore:
#                 score = scoreCalibration(llrs, labels)
#             else:
#                 score = llrs
#             actDCF = compute_actual_DCF(score, labels, pi, 1, 1)
#             table.add_row([f"PCA m={m}, variant={variant}, π_tilde={pi} -> actual dcf", round(actDCF, 3)])
#         else:
#             min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
#             table.add_row([f"PCA m={m}, variant={variant}, π_tilde={pi} -> min dcf", round(min_dcf, 3)])
#     print(table)


def MVG_simulations(training_data, training_labels, calibrateScore=False, actualDCF=False):
    variants = ['full-cov', 'diag', 'tied']
    m = [7, 5, 4]
    pis = [0.1, 0.5, 0.9]

    hyperparameters = itertools.product(variants, m, pis)
    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for i, (variant, m, pi) in enumerate(hyperparameters):
        print(f"Iteration {i + 1}/27")
        llrs, labels = k_fold(training_data, training_labels, MVG, 5, seed=0, m=m, raw=True, variant=variant)
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        table.add_row([f"Raw features, PCA m={m}, variant={variant}, π_tilde={pi} -> min dcf", round(min_dcf, 3)])
    print(table)


def LR_simulations(training_data, training_labels, lbd, calibratedScore=False, actualDCF=False):
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
            if calibratedScore:
                score = calibrateScores(llrs, labels)
            else:
                score = llrs
            actDCF = compute_actual_DCF(score, labels, pi, 1, 1)
            table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T} -> actual dcf:", round(actDCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
            table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T} -> min dcf:", round(min_dcf, 3)])

    print(table)


def SVM_LinearUnbalancedSimulations(training_data, training_labels, K, C, calibratedScore=False, actualDCF=False):
    m = [False, None, 7, 5]
    priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, priors)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi in hyperparameters:
        if m == False:
            llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=False, m=None, raw=True,
                                  pi_T=None, k=K, c=C,
                                  kernel_params=(1, 0), kernel_type='poly')
        else:
            llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=False, m=m, raw=False,
                                  pi_T=None, k=K, c=C,
                                  kernel_params=(1, 0), kernel_type='poly')
        if actualDCF:
            if calibratedScore:
                score = calibrateScores(llrs, labels)
            else:
                score = llrs
            actDCF = compute_actual_DCF(score, labels, pi, 1, 1)
            table.add_row([f"PCA m={m}, π_tilde={pi},  C ={C}", round(actDCF, 3)])
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        print(f"PCA m={m} π_tilde={pi}, C ={C}, K{K} --> ", round(min_dcf, 3))
        table.add_row([f"PCA m={m}, π_tilde={pi}, C ={C}", round(min_dcf, 3)])

    print(table)


def SVM_PolySimulations(training_data, training_labels, K, C, pi_T, c, d):
    m = [False, None, 7, 5]
    priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, priors)
    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi in hyperparameters:
        if m == False:
            llrs, evaluationLabels = k_fold(training_data, training_labels, SVM, 5, seed=0, m=None, raw=True, k=K, c=C,
                                            balanced=True, pi_T=pi_T,
                                            kernel_params=(d, c), kernel_type='poly')
        else:
            llrs, evaluationLabels = k_fold(training_data, training_labels, SVM, 5, seed=0, m=m, raw=False, k=K, c=C,
                                            balanced=True, pi_T=pi_T,
                                            kernel_params=(d, c), kernel_type='poly')
        min_dcf = compute_min_DCF(llrs, evaluationLabels, 0.5, 1, 1)
        print(f"PCA m={m}, π_tilde={pi}, pi_T = 0.5, C ={C} K={K}, c={c}, d={d}", "-->",
              round(min_dcf, 3))
        table.add_row(
            [f"PCA m={m}, π_tilde={pi}, pi_T = 0.5, C ={C} K={K}, c={c}, d={d}", round(min_dcf, 3)])

    print(table)


def SVM_RBFSimulations(training_data, training_labels, K, C, pi_T, gamma):
    ms = [None, 7, 5]
    effective_priors = [0.5, 0.1, 0.9]
    hyperparameters = itertools.product(ms, effective_priors)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi in hyperparameters:
        if m == False:
            llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, m=None, raw=True, balanced=True,
                                  pi_T=pi_T,
                                  k=K, c=C, kernel_params=gamma, kernel_type='RBF')
        else:
            llrs, labels = k_fold(training_data, training_labels, SVM, 5, seed=0, m=m, raw=False, balanced=True,
                                  pi_T=pi_T,
                                  k=K, c=C, kernel_params=gamma, kernel_type='RBF')
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        print(f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(min_dcf, 3))
        table.add_row([f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])

    print(table)


def GMM_Simulations(training_data, training_labels, alpha, psi):
    variants = ['full-cov', 'diag', 'tied']
    effective_priors = [0.5, 0.1, 0.9]
    ms = [None, 7, 5]
    raws = [True, False]
    num_components = [2, 4]

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    hyperparameters = itertools.product(ms, effective_priors, variants, raws, num_components)
    n_iter = len(variants) * len(effective_priors) * len(ms) * len(raws) * len(num_components)

    for i, (m, pi, variant, raw, g) in enumerate(hyperparameters):
        print(f"Iteration {i + 1}/{n_iter}: ", end="")
        llrs, labels = k_fold(training_data, training_labels, GMM, nFold=5, seed=0, m=m, raw=raw, type=variant,
                              alpha=alpha, psi=psi, G=g)
        min_dcf = compute_min_DCF(np.array(llrs), labels, pi, 1, 1)
        print(f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", "-->", round(min_dcf, 3))
        table.add_row([f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", round(min_dcf, 3)])

    print(table)
