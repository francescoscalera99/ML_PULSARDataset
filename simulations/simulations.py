import itertools
import numpy as np
from prettytable import PrettyTable

from classifiers.GMM2 import GMM
from classifiers.LR import LR, calibrateScores
from classifiers.MVG import MVG
from classifiers.SVM import SVM
from utils.metrics_utils import compute_actual_DCF, compute_min_DCF
from utils.utils import k_fold


def MVG_simulations(training_data, training_labels, calibratedScore=False, actualDCF=False):
    variants = ['full-cov', 'diag', 'tied']
    ms = [False, None, 7, 5, 4]
    effective_priors = [0.1, 0.5, 0.9]

    hyperparameters = itertools.product(variants, ms, effective_priors)
    table = PrettyTable()
    if actualDCF:
        if calibratedScore:
            table.field_names = ['Hyperparameters', 'act DCF (0.5)', 'actDCF (0.1)', 'act DCF (0.9)']
        else:
            table.field_names = ['Hyperparameters', 'act DCF']
    else:
        table.field_names = ['Hyperparameters', 'min DCF']

    for i, (variant, m, pi) in enumerate(hyperparameters):
        print(f"Iteration {i + 1}/{len(variants) * len(ms) * len(effective_priors)}")
        if m == False:
            llrs, evaluationLabels = k_fold(training_data, training_labels, MVG, 5, seed=0, m=None, raw=True, variant=variant)
        else:
            llrs, evaluationLabels = k_fold(training_data, training_labels, MVG, 5, seed=0, m=m, raw=False,
                                            variant=variant)
        if actualDCF:
            if calibratedScore:
                actDCF_cal = []
                priors_T_logReg = [0.5, 0.1, 0.9]
                for prior in priors_T_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-4, prior=prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                table.add_row([f"PCA m={m}, variant={variant}, π_tilde={pi}", *actDCF_cal])
                print(f"PCA m={m}, variant={variant}, π_tilde={pi}", "-->", *actDCF_cal)
            else:
                score = llrs
                actDCF = compute_actual_DCF(score, evaluationLabels, pi, 1, 1)
                table.add_row([f"Gaussianized features, PCA m={m}, variant={variant}, π_tilde={pi}", round(actDCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            table.add_row([f"Gaussianzed features, PCA m={m}, variant={variant}, π_tilde={pi}", round(min_dcf, 3)])
    print(table)
    with open(f"results/actDCF/MVG_ACT-{actualDCF}_calibrated-{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def LR_simulations(training_data, training_labels, lbd, calibratedScore=False, actualDCF=False):
    m_values = [False, None, 7, 5]
    pis_T = [0.5, 0.1, 0.9]
    effective_priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m_values, effective_priors, pis_T)

    table = PrettyTable()
    if actualDCF:
        if calibratedScore:
            table.field_names = ['Hyperparameters', 'act DCF (0.5)', 'actDCF (0.1)', 'act DCF (0.9)']
        else:
            table.field_names = ['Hyperparameters', 'act DCF']
    else:
        table.field_names = ['Hyperparameters', 'min DCF']

    for i, (m, pi, pi_T) in enumerate(hyperparameters):
        print(f"Iteration {i + 1}/{len(m_values) * len(effective_priors) * len(pis_T)}")
        if m == False:  # raw features
            llrs, evaluationLabels = k_fold(training_data, training_labels, LR, 5, m=None, raw=True, seed=0, lbd=lbd,
                                            pi_T=pi_T)
        else:
            llrs, evaluationLabels = k_fold(training_data, training_labels, LR, 5, m=m, raw=False, seed=0, lbd=lbd,
                                            pi_T=pi_T)
        if actualDCF:
            if calibratedScore:
                actDCF_cal = []
                priors_logReg = [0.5, 0.1, 0.9]
                for prior in priors_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-4, prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                print(f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}", "-->", *actDCF_cal)
                table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}", *actDCF_cal])
            else:
                score = llrs
                actDCF = compute_actual_DCF(score, evaluationLabels, pi, 1, 1)
                table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}", round(actDCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}", round(min_dcf, 3)])

    print(table)
    with open(f"results/actDCF/LR_ACT-{actualDCF}_calibrated-{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def SVM_LinearUnbalancedSimulations(training_data, training_labels, K, C, calibratedScore=False, actualDCF=False):
    ms = [False, None, 7, 5]
    effective_priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(ms, effective_priors)

    table = PrettyTable()
    if actualDCF:
        if calibratedScore:
            table.field_names = ['Hyperparameters', 'act DCF (0.5)', 'actDCF (0.1)', 'act DCF (0.9)']
        else:
            table.field_names = ['Hyperparameters', 'act DCF']
    else:
        table.field_names = ['Hyperparameters', 'min DCF']

    for i, (m, pi) in enumerate(hyperparameters):
        print(f"Iteration {i + 1}/{len(ms) * len(effective_priors)}")
        if m == False:
            llrs, evaluationLabels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=False, m=None,
                                            raw=True,
                                            pi_T=None, k=K, c=C,
                                            kernel_params=(1, 0), kernel_type='poly')
        else:
            llrs, evaluationLabels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=False, m=m,
                                            raw=False,
                                            pi_T=None, k=K, c=C,
                                            kernel_params=(1, 0), kernel_type='poly')
        if actualDCF:
            if calibratedScore:
                actDCF_cal = []
                priors_logReg = [0.5, 0.1, 0.9]
                for prior in priors_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-4, prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                table.add_row([f"PCA m={m} π_tilde={pi}, C ={C}, K{K} --> ", *actDCF_cal])
            else:
                score = llrs
                act_DCF = compute_actual_DCF(score, evaluationLabels, pi, 1, 1)
                print(f"PCA m={m} π_tilde={pi}, C ={C}, K{K} --> ", round(act_DCF, 3))
                table.add_row(
                    [f"PCA m={m} π_tilde={pi}, C ={C}, K{K} --> ", round(act_DCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            print(f"PCA m={m} π_tilde={pi}, C ={C}, K{K} --> ", round(min_dcf, 3))
            table.add_row([f"PCA m={m}, π_tilde={pi}, C ={C}", round(min_dcf, 3)])

    print(table)
    with open(f"results/actDCF/SVM_linear_unbalanced_ACT-{actualDCF}_calibrated-{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def SVM_LinearBalancedSimulations(training_data, training_labels, K, C, calibratedScore=False, actualDCF=False):
    m = [False, None, 7, 5]
    pi_T_values = [0.5, 0.1, 0.9]
    effective_priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, effective_priors, pi_T_values)

    table = PrettyTable()
    if actualDCF:
        if calibratedScore:
            table.field_names = ['Hyperparameters', 'act DCF (0.5)', 'actDCF (0.1)', 'act DCF (0.9)']
        else:
            table.field_names = ['Hyperparameters', 'act DCF']
    else:
        table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi, pi_T in hyperparameters:
        if m == False:
            llrs, evaluationLabels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=True, m=None,
                                            raw=True,
                                            pi_T=pi_T, k=K, c=C,
                                            kernel_params=(1, 0), kernel_type='poly')
        else:
            llrs, evaluationLabels = k_fold(training_data, training_labels, SVM, 5, seed=0, balanced=True, m=m,
                                            raw=False,
                                            pi_T=pi_T, k=K, c=C,
                                            kernel_params=(1, 0), kernel_type='poly')
        if actualDCF:
            if calibratedScore:
                actDCF_cal = []
                priors_logReg = [0.5, 0.1, 0.9]
                for prior in priors_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-4, prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                table.add_row([f"PCA m={m}, pi_T={pi_T}, π_tilde={pi}, C ={C}", *actDCF_cal])
            else:
                score = llrs
                act_DCF = compute_actual_DCF(score, evaluationLabels, pi, 1, 1)
                print(f"PCA m={m}, pi_T={pi_T}, π_tilde={pi}, C ={C}", "-->", round(act_DCF, 3))
                table.add_row(
                    [f"PCA m={m}, pi_T={pi_T}, π_tilde={pi}, C ={C}", round(act_DCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            print(f"PCA m={m} pi_T={pi_T} π_tilde={pi}, C ={C}, K{K} --> ", round(min_dcf, 3))
            table.add_row([f"PCA m={m}, pi_T={pi_T}, π_tilde={pi}, C ={C}", round(min_dcf, 3)])

    print(table)
    with open(f"results/actDCF/SVM_linear_balanced_ACT-{actualDCF}_calibrated{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def SVM_PolySimulations(training_data, training_labels, K, C, pi_T, c, d, actualDCF=False, calibratedScore=False):
    m = [False, None, 7, 5]
    effective_priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(m, effective_priors)

    table = PrettyTable()
    if actualDCF:
        if calibratedScore:
            table.field_names = ['Hyperparameters', 'act DCF (0.5)', 'actDCF (0.1)', 'act DCF (0.9)']
        else:
            table.field_names = ['Hyperparameters', 'act DCF']
    else:
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
        if actualDCF:
            if calibratedScore:
                actDCF_cal = []
                priors_logReg = [0.5, 0.1, 0.9]
                for prior in priors_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-4, prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                table.add_row([f"PCA m={m}, π_tilde={pi}, pi_T = 0.5, C ={C} K={K}, c={c}, d={d}", *actDCF_cal])
            else:
                score = llrs
                act_DCF = compute_actual_DCF(score, evaluationLabels, pi, 1, 1)
                print(f"PCA m={m}, π_tilde={pi}, pi_T = 0.5, C ={C} K={K}, c={c}, d={d}", "-->", round(act_DCF, 3))
                table.add_row(
                    [f"PCA m={m}, π_tilde={pi}, pi_T = 0.5, C ={C} K={K}, c={c}, d={d}", round(act_DCF, 3)])
        else:
            min_dcf = compute_min_DCF(llrs, evaluationLabels, pi, 1, 1)
            print(f"PCA m={m}, π_tilde={pi}, pi_T = 0.5, C ={C} K={K}, c={c}, d={d}", "-->",
                  round(min_dcf, 3))
            table.add_row(
                [f"PCA m={m}, π_tilde={pi}, pi_T = 0.5, C ={C} K={K}, c={c}, d={d}", round(min_dcf, 3)])

    print(table)
    with open(f"results/actDCF/SVM_Poly_ACT-{actualDCF}_calibrated{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def SVM_RBFSimulations(training_data, training_labels, K, C, pi_T, gamma, actualDCF=False, calibratedScore=False):
    ms = [None, 7, 5]
    effective_priors = [0.5, 0.1, 0.9]
    hyperparameters = itertools.product(ms, effective_priors)

    table = PrettyTable()
    if actualDCF:
        if calibratedScore:
            table.field_names = ['Hyperparameters', 'act DCF (0.5)', 'actDCF (0.1)', 'act DCF (0.9)']
        else:
            table.field_names = ['Hyperparameters', 'act DCF']
    else:
        table.field_names = ['Hyperparameters', 'min DCF']

    for m, pi in hyperparameters:
        if m == False:
            llrs, evaluationLabels = k_fold(training_data, training_labels, SVM, 5, seed=0, m=None, raw=True,
                                            balanced=True,
                                            pi_T=pi_T,
                                            k=K, c=C, kernel_params=gamma, kernel_type='RBF')
        else:
            llrs, evaluationLabels = k_fold(training_data, training_labels, SVM, 5, seed=0, m=m, raw=False,
                                            balanced=True,
                                            pi_T=pi_T,
                                            k=K, c=C, kernel_params=gamma, kernel_type='RBF')
        if actualDCF:
            if calibratedScore:
                actDCF_cal = []
                priors_logReg = [0.5, 0.1, 0.9]
                for prior in priors_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-4, prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                table.add_row([f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", *actDCF_cal])
            else:
                score = llrs
                act_DCF = compute_actual_DCF(score, evaluationLabels, pi, 1, 1)
                print(f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(act_DCF, 3))
                table.add_row(
                    [f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", round(act_DCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            table.add_row([f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])

    print(table)
    with open(f"results/actDCF/SVM_RBF_ACT-{actualDCF}_calibrated{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def GMM_Simulations(training_data, training_labels, g, alpha, psi, actualDCF=False, calibratedScore=False):
    variants = ['full-cov', 'diag', 'tied']
    effective_priors = [0.5, 0.1, 0.9]
    ms = [None, 7]
    raws = [True, False]

    table = PrettyTable()
    if actualDCF:
        if calibratedScore:
            table.field_names = ['Hyperparameters', 'act DCF (0.5)', 'actDCF (0.1)', 'act DCF (0.9)']
        else:
            table.field_names = ['Hyperparameters', 'act DCF']
    else:
        table.field_names = ['Hyperparameters', 'min DCF']

    hyperparameters = itertools.product(ms, effective_priors, variants, raws)
    n_iter = len(variants) * len(effective_priors) * len(ms) * len(raws)

    for i, (m, pi, variant, raw) in enumerate(hyperparameters):
        print(f"Iteration {i + 1}/{n_iter}: ", end="")
        llrs, evaluationLabels = k_fold(training_data, training_labels, GMM, nFold=5, seed=0, m=m, raw=raw,
                                        type=variant,
                                        alpha=alpha, psi=psi, G=g)
        if actualDCF:
            if calibratedScore:
                actDCF_cal = []
                priors_logReg = [0.5, 0.1, 0.9]
                for prior in priors_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-4, prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                print(f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", "-->", *actDCF_cal)
                table.add_row([f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", *actDCF_cal])
            else:
                score = llrs
                act_DCF = compute_actual_DCF(score, evaluationLabels, pi, 1, 1)
                print(f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", "-->", round(act_DCF, 3))
                table.add_row([f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", round(act_DCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            print(f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", "-->", round(min_dcf, 3))
            table.add_row([f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", round(min_dcf, 3)])

    print(table)
    with open(f"results/actDCF/GMM_ACT-{actualDCF}_calibrated{calibratedScore}.txt", 'w') as f:
        f.write(str(table))
