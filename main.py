import itertools

import numpy as np

from classifiers.GMM2 import tuning_componentsGMM
from classifiers.LR import tuning_lambda
from classifiers.SVM import tuning_parameters_PolySVM, tuning_parameters_RBFSVM, tuning_parameters_LinearSVMBalanced
from preprocessing.preprocessing import PCA
from simulations.simulations import MVG_simulations, GMM_Simulations, SVM_LinearUnbalancedSimulations, \
    SVM_PolySimulations, SVM_RBFSimulations, SVM_LinearBalancedSimulations, LR_simulations
from utils.plot_utils import create_scatterplots
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
    lbd = 1e-1
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
    # SVM_LinearSimulations(training_data, training_labels, K_Linear, C_piT_Linear, actualDCF=True, calibratedScore=False)
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d, actualDCF=True, calibratedScore=False)
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF, actualDCF=True, calibratedScore=False)
    # GMM_Simulations(training_data, training_labels, g, alpha=0.1, psi=0.01, actualDCF=True)

    # =============== SCORE CALIBRATION ===============
    # print("============== MVG - SCORE CALIBRATION =============== ")
    # MVG_simulations(training_data, training_labels, actualDCF=True, calibratedScore=True)
    # print("============== SVM LINEAR UNBALANCED - SCORE CALIBRATION ===============")
    # SVM_LinearUnbalancedSimulations(training_data, training_labels, K_LinearUnb, C_LinearUnb, actualDCF=True, calibratedScore=True )
    # print("============== SVM LINEAR BALANCED - SCORE CALIBRATION ===============")
    # SVM_LinearBalancedSimulations(training_data, training_labels, K_LinearB, C_LinearB, actualDCF=True, calibratedScore=True)
    # print("============== SVM POLY - SCORE CALIBRATION ===============")
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d, actualDCF=True, calibratedScore=True)
    # print("============== SVM RBF BALANCED - SCORE CALIBRATION ===============")
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF, actualDCF=True, calibratedScore=True)
    # print("============== LR - SCORE CALIBRATION ===============")
    # LR_simulations(training_data, training_labels, lbd, actualDCF=True, calibratedScore=True)
    # print("============== GMM - SCORE CALIBRATION ===============")
    # GMM_Simulations(training_data, training_labels, g, alpha=0.1, psi=0.01, actualDCF=True, calibratedScore=True)

    # =============== EXPERIMENTAL RESULT ===============

    # ****************** TURN OFF PC AT END OF SIMULATION (needs sudo) ******************
    # (windows ?)
    # os.system("shutdown /s /t 1")
    # MAC
    # os.system("shutdown -h now")


if __name__ == '__main__':
    main()
