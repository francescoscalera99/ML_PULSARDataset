import itertools
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from PCA import PCA
from classifiers.LR import LR
from classifiers.MVG import MVG
from classifiers.SVM import SVM
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

        (dtr, ltr), (dte, lte) = splitData_SingleFold(training_data, training_labels, seed=0)
        dtr_gaussianized = gaussianize(dtr, dtr)
        dte_gaussianized = gaussianize(dtr, dte)
        titles = ['1. Mean of the integrated profile',
                  '2. Standard deviation of the integrated profile',
                  '3. Excess kurtosis of the integrated profile',
                  '4. Excess kurtosis of the integrated profile',
                  '5. Mean of the DM-SNR curve',
                  '6. Standard deviation of the DM-SNR curve',
                  '7. Excess kurtosis of the DM-SNR curve',
                  '8. Skewness of the DM-SNR curve']

        mvg_raw = MVG(dtr, ltr, variant=variant)
        mvg_raw.train_model()
        mvg_raw.classify(dte, np.array([1 - pi, pi]))

        mvg_gauss = MVG(dtr_gaussianized, ltr, variant=variant)
        mvg_gauss.train_model()
        mvg_gauss.classify(dte_gaussianized, np.array([1 - pi, pi]))
        # cm = build_confusion_matrix(lte, predictions)
        # print(cm)

        llrs_raw = mvg_raw.get_llrs()
        min_dcf_raw = compute_min_DCF(llrs_raw, lte, pi, 1, 1)

        llrs_gauss = mvg_gauss.get_llrs()
        min_dcf_gauss = compute_min_DCF(llrs_gauss, lte, pi, 1, 1)
        # print(min_dcf)
        table.add_row([f'{variant}, m={m}, data: raw, π={pi}', round(min_dcf_raw, 3)])
        table.add_row([f'{variant}, m={m}, data: gaussianized, π={pi}', round(min_dcf_gauss, 3)])

    print(table)


def LR_simulations(training_data, training_labels):
    # datas = [training_data, z_normalized_training_data, z_gauss_training_data]
    # data_types = ['raw', 'z-normalized', 'z-normalized + gaussianized']
    # ds = list(range(3))
    # m = [None, 7, 5, 4]
    m = [None, 7, 5, 4]
    pis = [0.1, 0.5, 0.9]
    pis_T = [0.5, 0.1, 0.9]
    pis = [0.5, 0.1, 0.9]
    lbd = 1e-3

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


def LR_simulations(training_data, training_labels):
    # datas = [training_data, z_normalized_training_data, z_gauss_training_data]
    # data_types = ['raw', 'z-normalized', 'z-normalized + gaussianized']
    # ds = list(range(3))
    # m = [None, 7, 5, 4]
    m = [None, 7, 5, 4]
    pis = [0.1, 0.5, 0.9]
    pis_T = [0.5, 0.1, 0.9]
    pis = [0.5, 0.1, 0.9]
    lbd = 1e-3

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
    K_values = [0.0, 1.0, 10.0]
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
    (training_data, training_labels), _ = load_dataset()
    # find_optLambda(training_data, training_labels)
    # LR_simulations(training_data, training_labels)
    # for k, v in dcfs.items():
    #     np.save(f"pi{k}", np.array(v))
    # #
    # plt.figure()
    # colors = ['red', 'green', 'blue']
    # for i, (k, v) in enumerate(dcfs.items()):
    #     plt.plot(lambdas, v, color=colors[i], label=f'π={k}')
    # plt.legend()
    # plt.xticks()
    # plt.show()
    #
    # MVG_simulations(training_data, training_labels)
    # (dtr, ltr), (dte, lte) = splitData_SingleFold(training_data, training_labels, seed=0)
    #
    # dtr = gaussianize(dtr, dtr)
    # dte = gaussianize(dtr, dte)
    #
    # classifier = SVM(dtr, ltr, k=1, c=5e-5, kernel_params=(2, 15), kernel_type='poly')
    # classifier.train_model()
    # classifier.classify(dte, None)
    # llrs = classifier.get_llrs()
    # min_dcf = compute_min_DCF(np.array(llrs), lte, 0.5, 1, 1)
    # print(min_dcf)
    tuning_parameters_PolySVM(training_data, training_labels)


if __name__ == '__main__':
    main()
