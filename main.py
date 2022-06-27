import itertools
import os

import numpy as np

from classifiers.GMM import tuning_componentsGMM
from classifiers.LR import tuning_lambda
from classifiers.SVM import tuning_parameters_PolySVM, tuning_parameters_RBFSVM, tuning_parameters_LinearSVMBalanced
from preprocessing.preprocessing import PCA
from simulations.simulations import MVG_simulations, GMM_Simulations, SVM_LinearUnbalancedSimulations, \
    SVM_PolySimulations, SVM_RBFSimulations
from utils.plot_utils import create_scatterplots
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

    # create_scatterplots(training_data, training_labels)
    # data = PCA(training_data, training_data, 7)
    # create_scatterplots(training_data, training_labels)
    # =============== MULTIVARIATE GAUSSIAN CLASSIFIER ===============
    # MVG_simulations(training_data, training_labels)

    # =============== LOGISTIC REGRESSION ===============
    # tuning_lambda(training_data, training_labels)
    # lbd = 1e-1
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

    # C_piT_LinearBalanced = [(1e-2, None), (1e-3, 0.5), (6 * 1e-3, 0.1), (7 * 1e-4, 0.9)] #These values comes from tuning of hyperparameter

    # print(" ---------- SVM LINEAR UNBALANCED SIMULATION ----------")
    # K_LinearUnb = 1.0  # This values comes from tuning of hyperparameters
    # C_LinearUnb = 1
    # SVM_LinearUnbalancedSimulations(training_data, training_labels, K_LinearUnb, C_LinearUnb)

    # print(" ---------- SVM POLY SIMULATION ----------")
    # K_Poly = 1.0
    # pi_TPolyRBF = 0.5
    # CPoly = 1e-2
    # c = 15
    # d = 2
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d)

    # print(" ---------- SVM RBF SIMULATION ----------")
    # K_RBF = 0
    # gamma_RBF = 1e-3
    # C_RBF = 1e-1
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF)

    # =============== GAUSSIAN MIXTURE MODELS ===============
    # print("GMM TUNING")
    # tuning_componentsGMM(training_data, training_labels, psi=0.1)
    # print("GMM SIMULATIONS")
    # GMM_Simulations(training_data, training_labels, alpha=0.1, psi=0.01)

    # =============== COMPUTING ACTUAL DCF ===============
    # MVG_simulations(training_data, training_labels, actualDCF=True, calibrateScore=False)
    # LR_simulations(training_data, training_labels, lbd)
    # SVM_LinearSimulations(training_data, training_labels, K_Linear, C_piT_Linear, actualDCF=True, calibratedScore=False)
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d, actualDCF=True, calibratedScore=False)
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF, actualDCF=True, calibratedScore=False)

    # =============== SCORE CALIBRATION ===============
    # MVG_simulations(training_data, training_labels, actualDCF=True, calibrateScore=True)
    # SVM_LinearSimulations(training_data, training_labels, K_Linear, C_piT_Linear, actualDCF=True, calibratedScore=True)
    # SVM_PolySimulations(training_data, training_labels, K_Poly, CPoly, pi_TPolyRBF, c, d, actualDCF=True, calibratedScore=True)
    # SVM_RBFSimulations(training_data, training_labels, K_RBF, C_RBF, pi_TPolyRBF, gamma_RBF, actualDCF=True, calibratedScore=True)
    # LR_simulations(training_data, training_labels, lbd, , actualDCF=True, calibrateScore=False)

    # =============== EXPERIMENTAL RESULT ===============

    # ****************** TURN OFF PC AT END OF SIMULATION (needs sudo) ******************
    # (windows ?)
    # os.system("shutdown /s /t 1")
    # MAC
    # os.system("shutdown -h now")

    # (dtr, ltr), (dte, lte) = splitData_SingleFold(training_data, training_labels, seed=0)
    # gmm_classifier = GMM(dtr, ltr, type='diag')
    # g = 1
    # while g <= 16:
    #     gmm_classifier.train_model(alpha=0.1, psi=0.01, G=4)
    #     num_classes = len(set(training_labels))
    #     priors = np.array([1 / num_classes] * num_classes)
    #     predictions = gmm_classifier.classify(dte, priors)
    #
    #     acc, err = compute_accuracy(predictions, lte)
    #     print(f"Error rate for type {'diag'} and G={1}: {round(err * 100, 2)}%")
    #     g += 2


def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    _, detC = np.linalg.slogdet(C)
    invC = np.linalg.inv(C)
    return np.diag(
        -(M / 2) * np.log(2 * np.pi) - (1 / 2) * (detC) - (1 / 2) * np.dot(np.dot((x - mu).T, invC), (x - mu)))


def find_already_done():
    K_values = [1.0, 10.0]
    priors = [0.5, 0.1, 0.9]
    pi_T_values = [0.5, 0.1, 0.9]
    ms = [False, None, 7, 5]
    C_values = np.logspace(-2, 2, 20)
    done = []
    import os
    print(os.path.abspath("."))
    for m, pi_T,  K, p in itertools.product(ms, pi_T_values, K_values, priors):
        try:
            np.load(f"simulations/linearSVM/balanced/K{str(K).replace('.', '-')}_p{str(p).replace('.', '-')}_pT{str(pi_T).replace('.', '-')}_PCA{m}.npy")
            done.append((m, pi_T, K, p))
        except FileNotFoundError:
            pass
    print(done)

# i = 0
# fig, axs = plt.subplots(1, 4)
# fig.suptitle('Tuning hyperparameter λ')
# plt.rcParams['text.usetex'] = True
# for m in m_values:
#   for pi in prior:
#     DCFs = np.load(f"/content/drive/My Drive/Pulsar/tuningLambda/LR_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")
#     axs[i].plot(lbd_values, DCFs, color=np.random.rand(3,), label=r"$\widetilde{\pi}=$")
#     if m == False:
#       axs[i].set_title(f'5-fold, Raw features')
#     else:
#       axs[i].set_title(f'5-fold, PCA (m={m})')
#     axs[i].legend()
#     axs[i].set_xlabel('λ')
#     axs[i].set_ylabel('minDCF')
#     axs[i].set_xscale('log')
#   i+=1
# fig.set_size_inches(20, 5)
# fig.tight_layout()
# fig.subplots_adjust(top=0.88)
# fig.show()


if __name__ == '__main__':
    main()
