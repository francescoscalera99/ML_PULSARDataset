import numpy as np
from utils.utils import load_dataset


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
    # tuning_lamba(training_data, training_labels)
    # lbd = 1e-3
    # LR_simulations(training_data, training_labels, lbd)

    # =============== SUPPORT VECTOR MACHINE ===============
    # print("LINEAR SVM - TUNING PARAMETERS")
    # tuning_parameters_LinearSVMUnbalanced(training_data, training_labels)
    # tuning_parameters_LinearSVMBalanced(training_data, training_labels)
    # print("POLY SVM - TUNING PARAMETERS")
    # tuning_parameters_PolySVM(training_data, training_labels)
    # print("RBF SVM - TUNING PARAMETERS")
    # tuning_parameters_RBFSVM(training_data, training_labels)
    # tuning_parameters_LinearSVMBalanced(training_data, training_labels)
    # K_Linear = 1.0 #This values comes from tuning of hyperparameters
    # C_piT_Linear = [(1e-2, None), (1e-3, 0.5), (6 * 1e-3, 0.1), (7 * 1e-4, 0.9)] #These values comes from tuning of hyperparameter
    # SVM_LinearSimulations(training_data, training_labels, K_Linear, C_piT_Linear)
    # K_Poly = 1.0
    # pi_TPolyRBF = 0.5
    # CPoly = 5*1e-1
    # c = 15
    # d = 2
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d)
    # K_RBF = 1.0
    # gamma_RBF = 1e-3
    # C_RBF = 1e-1
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF)

    # =============== GAUSSIAN MIXTURE MODELS ===============

    # =============== COMPUTING ACTUAL DCF ===============
    # MVG_simulations(training_data, training_labels, actualDCF=True, calibrateScore=False)
    # SVM_LinearSimulations(training_data, training_labels, K_Linear, C_piT_Linear, actualDCF=True, calibrateScore=False)
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d, actualDCF=True, calibrateScore=False)
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF, actualDCF=True, calibrateScore=False)
    # LR_simulations(training_data, training_labels, lbd)

    # =============== SCORE CALIBRATION ===============
    # MVG_simulations(training_data, training_labels, actualDCF=True, calibrateScore=True)
    # SVM_LinearSimulations(training_data, training_labels, K_Linear, C_piT_Linear, actualDCF=True, calibrateScore=True)
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d, actualDCF=True, calibrateScore=True)
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF, actualDCF=True, calibrateScore=True)
    # LR_simulations(training_data, training_labels, lbd, , actualDCF=True, calibrateScore=False)

    # =============== EXPERIMENTAL RESULT ===============

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
