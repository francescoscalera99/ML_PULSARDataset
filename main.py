import itertools
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from PCA import PCA
from classifiers.LR import LR, find_optLambda
from classifiers.MVG import MVG
from utils.utils import load_dataset, gaussianize, splitData_SingleFold, kFold
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
        allKFolds, evaluationLabels = kFold(training_data, training_labels)
        llrs = []
        for singleKFold in allKFolds:
            dtr_gaussianized = gaussianize(singleKFold[1], singleKFold[1])
            dte_gaussianized = gaussianize(singleKFold[1], singleKFold[2])
            lr = LR(dtr_gaussianized, singleKFold[0], lbd, pi_T)
            lr.train_model()
            lr.classify(dte_gaussianized, np.array([0.5, 0.5]))
            llr = lr.get_llrs()
            llr = llr.tolist()
            llrs.extend(llr)
        min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
        table.add_row([f"PCA m={m}, data: gaussianized, π_tilde={pi}, π_T={pi_T}  λ={lbd}", round(min_dcf, 3)])

    print(table)

def main():
    (training_data, training_labels), _ = load_dataset()
    #find_optLambda(training_data, training_labels)
    LR_simulations(training_data, training_labels)
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


if __name__ == '__main__':
    main()
