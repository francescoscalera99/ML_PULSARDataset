import itertools
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from PCA import PCA
from classifiers.LR import LR
from utils.metrics_utils import build_confusion_matrix, compute_min_DCF
from utils.utils import load_dataset, \
    z_normalization, \
    gaussianize, \
    compute_accuracy, splitData_SingleFold
from classifiers.MVG import MVG


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
        mvg_raw.classify(dte, np.array([1-pi, pi]))

        mvg_gauss = MVG(dtr_gaussianized, ltr, variant=variant)
        mvg_gauss.train_model()
        mvg_gauss.classify(dte_gaussianized, np.array([1-pi, pi]))
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
    m = [None]
    pis = [0.1, 0.5, 0.9]
    lambdas = np.logspace(-5, 5, 50)
    # datas = [training_data, z_normalized_training_data, z_gauss_training_data]
    # data_types = ['raw', 'z-normalized', 'z-normalized + gaussianized']
    # ds = list(range(3))
    # m = [None, 7, 5, 4]
    # pis = [0.1, 0.5, 0.9]
    # lambdas = np.logspace(-5, 5, 50)

    hyperparameters = itertools.product(m, pis, lambdas)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    dcfs = dict.fromkeys(pis, [])

    for m, pi, lbd in hyperparameters:
        if m is not None:
            training_data = PCA(training_data, m)
        (dtr, ltr), (dte, lte) = splitData_SingleFold(training_data, training_labels, seed=0)
        dtr_gaussianized = gaussianize(dtr, dtr)
        dte_gaussianized = gaussianize(dte, dtr)

        lr_raw = LR(dtr, ltr, lbd, pi)
        lr_raw.train_model()
        lr_raw.classify(dte, np.array([1-pi, pi]))
        llrs_raw = lr_raw.get_llrs()
        min_dcf_raw = compute_min_DCF(llrs_raw, lte, pi, 1, 1)

        lr_gauss = LR(dtr_gaussianized, ltr, lbd, pi)
        lr_gauss.train_model()
        lr_gauss.classify(dte_gaussianized, np.array([1-pi, pi]))
        llrs_gauss = lr_raw.get_llrs()
        min_dcf_gauss = compute_min_DCF(llrs_gauss, lte, pi, 1, 1)

        table.add_row([f"PCA m={m}, data: raw, π={pi}, λ={lbd}", round(min_dcf_raw, 3)])
        table.add_row([f"PCA m={m}, data: gaussianized, π={pi}, λ={lbd}", round(min_dcf_gauss, 3)])
        # dcfs[pi].append(min_dcf_raw)

    return dcfs, lambdas


def main():
    (training_data, training_labels), _ = load_dataset()

    # dcfs, lambdas = LR_simulations(training_data, training_labels)
    # for k, v in dcfs.items():
    #     np.save(f"pi{k}", np.array(v))
    # # print(table)
    # plt.figure()
    # colors = ['red', 'green', 'blue']
    # for i, (k, v) in enumerate(dcfs.items()):
    #     plt.plot(lambdas, v, color=colors[i], label=f'π={k}')
    # plt.legend()
    # plt.xticks()
    # plt.show()

    MVG_simulations(training_data, training_labels)


if __name__ == '__main__':
    main()
