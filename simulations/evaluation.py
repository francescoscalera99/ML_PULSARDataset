import itertools

import numpy as np
from prettytable import PrettyTable

from classifiers.GMM2 import GMM
from classifiers.LR import calibrateScores, LR
from classifiers.MVG import MVG
from classifiers.SVM import SVM
from preprocessing.preprocessing import PCA, gaussianize
from utils.metrics_utils import compute_actual_DCF, compute_min_DCF


def MVG_evaluation(training_data, ltr, testing_data, evaluationLabels, actualDCF=False, calibratedScore=False):
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
            dtr = training_data
            dte = testing_data
        else:
            dtr_gauss = gaussianize(training_data, training_data)
            dte_gauss = gaussianize(training_data, testing_data)
            if m is not None:
                dtr = PCA(dtr_gauss, dtr_gauss, m)
                dte = PCA(dte_gauss, dtr_gauss, m)
            else:
                dtr = dtr_gauss
                dte = dte_gauss

        mvg = MVG(dtr, ltr, variant=variant)
        mvg.train_model()
        mvg.classify(dte, None)

        llrs = mvg.get_llrs()
        del mvg

        if actualDCF:
            if calibratedScore:

                priors_T_logReg = [0.5, 0.1, 0.9]

                actDCF_cal = []
                for prior in priors_T_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-4, prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                table.add_row([f"PCA m={m}, variant={variant}, π_tilde={pi}", *actDCF_cal])
                print(f"PCA m={m}, variant={variant}, π_tilde={pi}", "-->", *actDCF_cal)
            else:
                actDCF = compute_actual_DCF(llrs, evaluationLabels, pi, 1, 1)
                table.add_row([f"PCA m={m}, variant={variant}, π_tilde={pi}", round(actDCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            table.add_row([f"PCA m={m}, variant={variant}, π_tilde={pi}", round(min_dcf, 3)])
    print(table)
    with open(f"results/evaluation/MVG_EVAL_ACT-{actualDCF}_calibrated-{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def LR_evaluation(training_data, ltr, testing_data, evaluationLabels, lbd, actualDCF=False, calibratedScore=False):
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

        if m == False:
            dtr = training_data
            dte = testing_data
        else:
            dtr_gauss = gaussianize(training_data, training_data)
            dte_gauss = gaussianize(training_data, testing_data)
            if m is not None:
                dtr = PCA(dtr_gauss, dtr_gauss, m)
                dte = PCA(dte_gauss, dtr_gauss, m)
            else:
                dtr = dtr_gauss
                dte = dte_gauss

        logReg = LR(dtr, ltr, lbd=lbd, pi_T=pi_T)
        logReg.train_model()
        logReg.classify(dte, None)

        llrs = logReg.get_llrs()
        del logReg

        if actualDCF:
            if calibratedScore:
                actDCF_cal = []
                priors_logReg = [0.5, 0.1, 0.9]
                for prior in priors_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-4, prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                print(f"PCA m={m}, π_tilde={pi}, π_T={pi_T}", "-->", *actDCF_cal)
                table.add_row([f"PCA m={m}, π_tilde={pi}, π_T={pi_T}", *actDCF_cal])
            else:
                actDCF = compute_actual_DCF(llrs, evaluationLabels, pi, 1, 1)
                table.add_row([f"PCA m={m}, π_tilde={pi}, π_T={pi_T}", round(actDCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            table.add_row([f"PCA m={m}, π_tilde={pi}, π_T={pi_T}", round(min_dcf, 3)])

    print(table)
    with open(f"results/evaluation/LR_EVAL_ACT-{actualDCF}_calibrated-{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def SVM_LinearUnbalanced_evaluation(training_data, ltr, testing_data, evaluationLabels, K, C, calibratedScore=False, actualDCF=False):
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
            dtr = training_data
            dte = testing_data
        else:
            dtr_gauss = gaussianize(training_data, training_data)
            dte_gauss = gaussianize(training_data, testing_data)
            if m is not None:
                dtr = PCA(dtr_gauss, dtr_gauss, m)
                dte = PCA(dte_gauss, dtr_gauss, m)
            else:
                dtr = dtr_gauss
                dte = dte_gauss

        svm = SVM(dtr, ltr, k=K, c=C, kernel_params=(1, 0), kernel_type='poly')
        svm.train_model(balanced=False, pi_T=None)
        svm.classify(dte, None)

        llrs = svm.get_llrs()
        del svm

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
                act_DCF = compute_actual_DCF(llrs, evaluationLabels, pi, 1, 1)
                print(f"PCA m={m} π_tilde={pi}, C ={C}, K{K} --> ", round(act_DCF, 3))
                table.add_row(
                    [f"PCA m={m} π_tilde={pi}, C ={C}, K{K} --> ", round(act_DCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            print(f"PCA m={m} π_tilde={pi}, C ={C}, K{K} --> ", round(min_dcf, 3))
            table.add_row([f"PCA m={m}, π_tilde={pi}, C ={C}", round(min_dcf, 3)])

    print(table)
    with open(f"results/evaluation/SVM_linear_unbalanced_EVAL_ACT-{actualDCF}_calibrated-{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def SVM_LinearBalanced_evaluation(training_data, ltr, testing_data, evaluationLabels, K, C, calibratedScore=False,
                                    actualDCF=False):
    ms = [False, None, 7, 5]
    pi_T_values = [0.5, 0.1, 0.9]
    effective_priors = [0.5, 0.1, 0.9]

    hyperparameters = itertools.product(ms, effective_priors, pi_T_values)

    table = PrettyTable()
    if actualDCF:
        if calibratedScore:
            table.field_names = ['Hyperparameters', 'act DCF (0.5)', 'actDCF (0.1)', 'act DCF (0.9)']
        else:
            table.field_names = ['Hyperparameters', 'act DCF']
    else:
        table.field_names = ['Hyperparameters', 'min DCF']

    for i, (m, pi, pi_T) in enumerate(hyperparameters):
        print(f"Iteration {i + 1}/{len(ms) * len(effective_priors)}")

        if m == False:
            dtr = training_data
            dte = testing_data
        else:
            dtr_gauss = gaussianize(training_data, training_data)
            dte_gauss = gaussianize(training_data, testing_data)
            if m is not None:
                dtr = PCA(dtr_gauss, dtr_gauss, m)
                dte = PCA(dte_gauss, dtr_gauss, m)
            else:
                dtr = dtr_gauss
                dte = dte_gauss

        svm = SVM(dtr, ltr, k=K, c=C, kernel_params=(1, 0), kernel_type='poly')
        svm.train_model(balanced=True, pi_T=pi_T)
        svm.classify(dte, None)

        llrs = svm.get_llrs()
        del svm

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
                act_DCF = compute_actual_DCF(llrs, evaluationLabels, pi, 1, 1)
                print(f"PCA m={m}, pi_T={pi_T}, π_tilde={pi}, C ={C}", "-->", round(act_DCF, 3))
                table.add_row(
                    [f"PCA m={m}, pi_T={pi_T}, π_tilde={pi}, C ={C}", round(act_DCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            print(f"PCA m={m} pi_T={pi_T} π_tilde={pi}, C ={C}, K{K} --> ", round(min_dcf, 3))
            table.add_row([f"PCA m={m}, pi_T={pi_T}, π_tilde={pi}, C ={C}", round(min_dcf, 3)])

    print(table)
    with open(f"results/evaluation/SVM_linear_balanced_EVAL_ACT-{actualDCF}_calibrated-{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def SVM_Poly_evaluation(training_data, ltr, testing_data, evaluationLabels, K, C, pi_T, c, d, actualDCF=False, calibratedScore=False):
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
            dtr = training_data
            dte = testing_data
        else:
            dtr_gauss = gaussianize(training_data, training_data)
            dte_gauss = gaussianize(training_data, testing_data)
            if m is not None:
                dtr = PCA(dtr_gauss, dtr_gauss, m)
                dte = PCA(dte_gauss, dtr_gauss, m)
            else:
                dtr = dtr_gauss
                dte = dte_gauss

        svm = SVM(dtr, ltr, k=K, c=C, kernel_params=(d, c), kernel_type='poly')
        svm.train_model(balanced=True, pi_T=pi_T)
        svm.classify(dte, None)

        llrs = svm.get_llrs()
        del svm

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
                act_DCF = compute_actual_DCF(llrs, evaluationLabels, pi, 1, 1)
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
    with open(f"results/evaluation/SVM_Poly_ACT-{actualDCF}_calibrated{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def SVM_RBF_evaluation(training_data, ltr, testing_data, evaluationLabels, K, C, pi_T, gamma, actualDCF=False, calibratedScore=False):
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
            dtr = training_data
            dte = testing_data
        else:
            dtr_gauss = gaussianize(training_data, training_data)
            dte_gauss = gaussianize(training_data, testing_data)
            if m is not None:
                dtr = PCA(dtr_gauss, dtr_gauss, m)
                dte = PCA(dte_gauss, dtr_gauss, m)
            else:
                dtr = dtr_gauss
                dte = dte_gauss

        svm = SVM(dtr, ltr, k=K, c=C, kernel_params=gamma, kernel_type='RBF')
        svm.train_model(balanced=True, pi_T=pi_T)
        svm.classify(dte, None)

        llrs = svm.get_llrs()
        del svm

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
                act_DCF = compute_actual_DCF(llrs, evaluationLabels, pi, 1, 1)
                print(f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", "-->", round(act_DCF, 3))
                table.add_row(
                    [f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", round(act_DCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            table.add_row([f"PCA m={m}, π_tilde={pi}, π_T={pi_T}  C ={C}", round(min_dcf, 3)])

    print(table)
    with open(f"results/evaluation/SVM_RBF_EVAL_ACT-{actualDCF}_calibrated{calibratedScore}.txt", 'w') as f:
        f.write(str(table))


def GMM_evaluation(training_data, ltr, testing_data, evaluationLabels, g, alpha, psi, actualDCF=False, calibratedScore=False):
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
        print(f"Iteration {i + 1}/{n_iter}: ")

        if raw:
            if m is not None:
                dtr = PCA(training_data, training_data, m)
                dte = PCA(testing_data, training_data, m)
            else:
                dtr = training_data
                dte = testing_data
        else:
            dtr_gauss = gaussianize(training_data, training_data)
            dte_gauss = gaussianize(training_data, testing_data)
            if m is not None:
                dtr = PCA(dtr_gauss, dtr_gauss, m)
                dte = PCA(dte_gauss, dtr_gauss, m)
            else:
                dtr = dtr_gauss
                dte = dte_gauss

        gmm = GMM(dtr, ltr, type=variant)
        gmm.train_model(alpha=alpha, psi=psi, G=g)
        gmm.classify(dte, None)

        llrs = gmm.get_llrs()
        del gmm

        if actualDCF:
            if calibratedScore:
                actDCF_cal = []
                priors_logReg = [0.5, 0.1, 0.9]
                for prior in priors_logReg:
                    score, labels = calibrateScores(llrs, evaluationLabels, 1e-7, prior)
                    act_DCF = compute_actual_DCF(score, labels, pi, 1, 1)
                    actDCF_cal.append(round(act_DCF, 3))
                print(f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", "-->", *actDCF_cal)
                table.add_row([f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", *actDCF_cal])
            else:
                act_DCF = compute_actual_DCF(llrs, evaluationLabels, pi, 1, 1)
                print(f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", "-->", round(act_DCF, 3))
                table.add_row([f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", round(act_DCF, 3)])
        else:
            min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
            print(f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", "-->", round(min_dcf, 3))
            table.add_row([f"PCA m={m}, raw data: {raw}, π_tilde={pi}, variant: {variant}, G={g}", round(min_dcf, 3)])

    print(table)
    with open(f"results/evaluation/GMM_EVAL_ACT-{actualDCF}_calibrated{calibratedScore}.txt", 'w') as f:
        f.write(str(table))