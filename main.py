#!/Users/riccardo/VENV/python3_9/bin/python3.9

import itertools

import numpy as np

from classifiers.GMM2 import tuning_componentsGMM, GMM
# from classifiers.LR import tuning_lambda
from classifiers.LR import LR
from classifiers.MVG import MVG
from classifiers.SVM import tuning_parameters_PolySVM, tuning_parameters_RBFSVM, tuning_parameters_LinearSVMBalanced, \
    SVM
from preprocessing.preprocessing import PCA, gaussianize
from simulations.evaluation import MVG_evaluation, LR_evaluation, SVM_LinearUnbalanced_evaluation, \
    SVM_LinearBalanced_evaluation, SVM_Poly_evaluation, SVM_RBF_evaluation, GMM_evaluation
from simulations.simulations import MVG_simulations, GMM_Simulations, SVM_LinearUnbalancedSimulations, \
    SVM_PolySimulations, SVM_RBFSimulations, SVM_LinearBalancedSimulations, LR_simulations
from simulations.tuning import tuning_parameters_LinearSVMUnbalanced_evaluation, \
    tuning_parameters_LinearSVMBalanced_evaluation, tuning_parameters_PolySVM_evaluation, \
    tuning_parameters_RBFSVM_evaluation, tuning_componentsGMM_evaluation, tuning_lambda_evaluation
from utils.metrics_utils import bayes_error_plots_data, bayes_error_plots_data_evaluation
from utils.plot_utils import create_scatterplots, bayes_error_plots, plot_lambda_evaluation, \
    plot_tuningPolySVM_evaluation, plot_tuningRBFSVM_evaluation, plot_tuningGMM_evaluation, \
    ROC_curve, ROC_curve_evaluation, generate_ROC_data
from utils.utils import load_dataset


def get_same_distrib_partition(dtr, ltr, perc=0.1, num_samples=1000):
    idx = np.random.permutation(dtr.shape[1])
    idx_t = idx[:num_samples]
    dtr2 = dtr[:, idx_t]
    ltr2 = ltr[idx_t]

    return dtr2, ltr2


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

    # create_scatterplots(training_data, training_labels)
    # data = PCA(training_data, training_data, 7)
    # create_scatterplots(training_data, training_labels)

    # =============== MULTIVARIATE GAUSSIAN CLASSIFIER ===============
    # MVG_simulations(training_data, training_labels)

    # =============== LOGISTIC REGRESSION ===============
    # tuning_lambda(training_data, training_labels)
    lbd = 1e-7
    # LR_simulations(training_data, training_labels, lbd)

    # =============== SUPPORT VECTOR MACHINE ===============
    # print("LINEAR SVM - TUNING PARAMETERS")
    # tuning_parameters_LinearSVMUnbalanced(training_data, training_labels)
    # tuning_parameters_LinearSVMBalanced(training_data, training_labels)
    # print("POLY SVM - TUNING PARAMETERS")
    # tuning_parameters_PolySVM(training_data, training_labels)
    # print("RBF SVM - TUNING PARAMETERS")
    # tuning_parameters_RBFSVM(training_data, training_labels)
    # tuning_parameters_LinearSVMUnBalanced(training_data, training_labels)

    # print(" ---------- SVM LINEAR UNBALANCED SIMULATION ----------")
    K_LinearUnb = 1.0  # This values comes from tuning of hyperparameters
    C_LinearUnb = 1
    # SVM_LinearUnbalancedSimulations(training_data, training_labels, K_LinearUnb, C_LinearUnb)

    # print(" ---------- SVM LINEAR BALANCED SIMULATION ----------")
    K_LinearB = 1.0  # This values comes from tuning of hyperparameters
    C_LinearB = 2e-2
    # SVM_LinearBalancedSimulations(training_data, training_labels, K_LinearB, C_LinearB)

    # print(" ---------- SVM POLY SIMULATION ----------")
    K_Poly = 1.0
    pi_TPolyRBF = 0.5
    CPoly = 1e-2
    c = 15
    d = 2
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d)

    # print(" ---------- SVM RBF SIMULATION ----------")
    K_RBF = 0
    gamma_RBF = 1e-3
    C_RBF = 1e-1
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF)

    # =============== GAUSSIAN MIXTURE MODELS ===============
    # tuning_componentsGMM(training_data, training_labels, psi=0.01)
    g = 16
    # GMM_Simulations(training_data, training_labels, g, alpha=0.1, psi=0.01)

    # =============== COMPUTING ACTUAL DCF ===============
    # MVG_simulations(training_data, training_labels, actualDCF=True, calibrateScore=False)
    # LR_simulations(training_data, training_labels, lbd, actualDCF=True)
    # SVM_LinearUnbalancedSimulations(training_data, training_labels, K_LinearUnb, C_LinearUnb, actualDCF=True, calibratedScore=False)
    # SVM_LinearBalancedSimulations(training_data, training_labels, K_LinearB, C_LinearB, actualDCF=True, calibratedScore=False)
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d, actualDCF=True, calibratedScore=False)
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF, actualDCF=True, calibratedScore=False)
    # GMM_Simulations(training_data, training_labels, g, alpha=0.1, psi=0.01, actualDCF=True)

    # =============== SCORE CALIBRATION ===============
    # print("============== MVG - SCORE CALIBRATION =============== ")
    # MVG_simulations(training_data, training_labels, actualDCF=True, calibratedScore=True)
    # print("============== LR - SCORE CALIBRATION ===============")
    # LR_simulations(training_data, training_labels, lbd, actualDCF=True, calibratedScore=True)
    # print("============== SVM LINEAR UNBALANCED - SCORE CALIBRATION ===============")
    # SVM_LinearUnbalancedSimulations(training_data, training_labels, K_LinearUnb, C_LinearUnb, actualDCF=True, calibratedScore=True )
    # print("============== SVM LINEAR BALANCED - SCORE CALIBRATION ===============")
    # SVM_LinearBalancedSimulations(training_data, training_labels, K_LinearB, C_LinearB, actualDCF=True, calibratedScore=True)
    # print("============== SVM POLY - SCORE CALIBRATION ===============")
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d, actualDCF=True, calibratedScore=True)
    # print("============== SVM RBF BALANCED - SCORE CALIBRATION ===============")
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF, actualDCF=True, calibratedScore=True)
    # print("============== GMM - SCORE CALIBRATION ===============")
    # GMM_Simulations(training_data, training_labels, g, alpha=0.1, psi=0.01, actualDCF=True, calibratedScore=True)

    # =============== ROC CURVE (BEST CLASSIFIER) ===============
    classifiers = [MVG, LR, SVM, GMM]
    args = [
        {"raw": False,
         "m": 7,
         "variant": "tied"},
        {"raw": False,
         "m": 7,
         "lbd": lbd,
         "pi_T": 0.5},
        {"raw": False,
         "m": 7,
         "k": K_LinearB,
         "c": C_LinearB,
         "pi_T": 0.5,
         "balanced": True,
         "kernel_type": "poly",
         "kernel_params": (1, 0)},
        {"raw": False,
         "m": 7,
         "G": g,
         "type": "full-cov",
         "alpha": 0.1,
         "psi": 0.1}]

    # ROC_curve(training_data, training_labels, classifiers, args)

    # =============== BAYES ERROR PLOT ==================
    # for i, classifier in enumerate(classifiers):
    #     print(f"{'*'*30} bep {i+1}/{len(classifiers)} {'*'*30}")
    #     bayes_error_plots_data(training_data, training_labels, classifier, **args[i])
    #
    # for i, classifier in enumerate(classifiers):
    #     print(f"{'*'*30} bep {i+1}/{len(classifiers)} {'*'*30}")
    #     bayes_error_plots_data(training_data, training_labels, classifier, **args[i])
    # print(f"plotting...")
    # for a in [True, False]:
    #     bayes_error_plots2(classifiers, after=a)

    # =============== EXPERIMENTAL RESULT ===============
    # MVG_evaluation(training_data, training_labels, testing_data, testing_labels)
    # LR_evaluation(training_data, training_labels, testing_data, testing_labels, lbd)
    # SVM_LinearUnbalanced_evaluation(training_data, training_labels, testing_data, testing_labels, K_LinearUnb, C_LinearUnb)
    # SVM_LinearBalanced_evaluation(training_data, training_labels, testing_data, testing_labels, K_LinearB, C_LinearB)
    # SVM_Poly_evaluation(training_data, training_labels, testing_data, testing_labels, K_Poly, CPoly, pi_TPolyRBF, c, d)
    # SVM_RBF_evaluation(training_data, training_labels, testing_data, testing_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF)
    # GMM_evaluation(training_data, training_labels, testing_data, testing_labels, g, alpha=0.1, psi=0.01)

    # =============== TUNING HYPERPARAMETERS - EXPERIMENTAL RESULT ===============
    # tuning_lambda_evaluation(training_data, training_labels, testing_data, testing_labels)
    # plot_lambda_evaluation()
    # tuning_parameters_LinearSVMUnbalanced_evaluation(training_data, training_labels, testing_data, testing_labels)
    # plot_tuningLinearSVMUnbalanced_evaluation()
    # tuning_parameters_LinearSVMBalanced_evaluation(training_data, training_labels, testing_data, testing_labels)
    # plot_tuning_LinearSVMBalanced_evaluation()
    # tuning_parameters_PolySVM_evaluation(training_data, training_labels, testing_data, testing_labels)
    # plot_tuningPolySVM_evaluation()
    # tuning_parameters_RBFSVM_evaluation(training_data, training_labels, testing_data, testing_labels)
    # plot_tuningRBFSVM_evaluation()
    # tuning_componentsGMM_evaluation(training_data, training_labels, testing_data, testing_labels)
    # plot_tuningGMM_evaluation()

    lbd = 1e-7
    K_LinearB = 1.0  # This values comes from tuning of hyperparameters
    C_LinearB = 2e-2

    K_Poly = 1.0
    CPoly = 1e-2
    c = 15
    d = 2

    K_RBF = 0
    gamma_RBF = 1e-3
    C_RBF = 1e-1

    g = 16
    # =============== ACTUAL DCF - EXPERIMENTAL RESULT ===============
    # MVG_evaluation(training_data, training_labels, testing_data, testing_labels, actualDCF=True)
    # LR_evaluation(training_data, training_labels, testing_data, testing_labels, lbd, actualDCF=True)
    # SVM_LinearUnbalanced_evaluation(training_data, training_labels, testing_data, testing_labels, K_LinearUnb, C_LinearUnb, actualDCF=True)
    # SVM_LinearBalanced_evaluation(training_data, training_labels, testing_data, testing_labels, K_LinearB, C_LinearB, actualDCF=True)
    # SVM_Poly_evaluation(training_data, training_labels, testing_data, testing_labels, K_Poly, CPoly, pi_TPolyRBF, c, d, actualDCF=True)
    # SVM_RBF_evaluation(training_data, training_labels, testing_data, testing_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF, actualDCF=True)
    # GMM_evaluation(training_data, training_labels, testing_data, testing_labels, g, alpha=0.1, psi=0.01, actualDCF=True)

    # =============== SCORE CALIBRATION - EXPERIMENTAL RESULT ===============
    # MVG_evaluation(training_data, training_labels, testing_data, testing_labels, actualDCF=True, calibratedScore=True)
    # LR_evaluation(training_data, training_labels, testing_data, testing_labels, lbd, actualDCF=True, calibratedScore=True)
    # SVM_LinearBalanced_evaluation(training_data, training_labels, testing_data, testing_labels, K_LinearB, C_LinearB, actualDCF=True, calibratedScore=True)
    # GMM_evaluation(training_data, training_labels, testing_data, testing_labels, g, alpha=0.1, psi=0.01, actualDCF=True, calibratedScore=True)

    # =============== ROC - EXPERIMENTAL RESULT ===============
    # classifiers2 = list(reversed(classifiers))
    # args2 = list(reversed(args))
    # dtr = gaussianize(training_data, training_data)
    # dte = gaussianize(training_data, testing_data)
    #
    # dtr7 = PCA(dtr, dtr, 7)
    # dte7 = PCA(dte, dtr, 7)

    # for i, c in enumerate(classifiers):
    #     print(f"Starting {c.__name__}...")
    #     generate_ROC_data(dtr7, training_labels, dte7, testing_labels, c, args[i])

    # ROC_curve_evaluation(classifiers)

    # =============== BAYES ERROR PLOT - EXPERIMENTAL RESULT ===============
    # for i, classifier in enumerate(classifiers):
    #     print(f"{'*'*30} bep {i+1}/{len(classifiers)} {'*'*30}")
    #     bayes_error_plots_data_evaluation(training_data, training_labels, testing_data, testing_labels, classifier, **args[i])
    # print(f"plotting...")
    for a in [True, False]:
        bayes_error_plots(classifiers, after=a, evaluation=True)

    # ****************** TURN OFF PC AT END OF SIMULATION (needs sudo) ******************
    # (windows ?)
    # os.system("shutdown /s /t 1")
    # MAC
    # os.system("shutdown -h now")


if __name__ == '__main__':
    main()
