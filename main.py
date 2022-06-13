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


def tuning_parameters_PolySVM(training_data, training_labels):
    titles_Kfold = ['Gaussianized feature (5-fold, no PCA)', 'Guassianized feature (5-fold, PCA = 7)', 'Gaussianized feature (5-fold, PCA = 5)']

    datasets = []

    training_dataPCA7 = PCA(training_data, 7)
    training_dataPCA5 = PCA(training_data, 5)
    datasets.append(training_data)
    datasets.append(training_dataPCA7)
    datasets.append(training_dataPCA5)
    C_values = np.logspace(-3, 3, 20)
    K_values = [0.0, 1.0]
    c_values = [0, 1, 10, 15]

    hyperparameters = itertools.product(c_values, K_values)
    j = 0
    for dataset in datasets:
        i = 0
        plt.figure()
        plt.rcParams['text.usetex'] = True
        for c, K in hyperparameters:
            DCFs = []
            for C in C_values:
                llrs, evaluationLabels = k_fold(dataset, training_labels, SVM, 5, k=K, c=C, kernel_params=(2, c), kernel_type='poly')
                print(llrs)
                min_dcf = compute_min_DCF(llrs, evaluationLabels, 0.5, 1, 1)
                print("min_DCF for C = ", C, "with c = ", c, "and K =", K, "->", min_dcf )
                DCFs.append(min_dcf)
            # f"prior:0.5, c:{c}, K:{K}"
            plt.plot(C_values, DCFs, color=np.random.rand(3,), label=r"$\pi_{}T=0.5$, c="+str(c)+r", K="+str(K))
        plt.title(titles_Kfold[j])
        j += 1
        plt.legend()
        plt.xscale('log')
        plt.show()


def main():
    (training_data, training_labels), (testing_data, testing_labels) = load_dataset()

    # =============== MULTIVARIATE GAUSSIAN CLASSIFIER ===============
    MVG_simulations(training_data, training_labels)

    # =============== LOGISTIC REGRESSION ===============
    #find_optLambda(training_data, training_labels)
    #lbd = 1e-3
    #LR_simulations(training_data, training_labels, lbd)

    # =============== SUPPORT VECTOR MACHINE ===============
    #tuning_parameters_PolySVM(training_data, training_labels)
    #tuning_parameters_LinearSVMBalanced(training_data, training_labels)


if __name__ == '__main__':
    main()
