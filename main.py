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


def MVG_simulations(training_data, z_normalized_training_data, z_gauss_training_data, training_labels):
    datas = [training_data, z_normalized_training_data, z_gauss_training_data]
    data_types = ['raw', 'z-normalized', 'z-normalized + gaussianized']
    ds = list(range(3))
    variants = ['full-cov', 'diag', 'tied']
    m = [None, 7, 5, 4]
    pis = [0.1, 0.5, 0.9]

    hyperparameters = itertools.product(variants, m, ds, pis)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    for variant, m, d, pi in hyperparameters:
        dtr = datas[d]
        if m is not None:
            dtr = PCA(dtr, m)
        (dtr, ltr), (dte, lte) = splitData_SingleFold(dtr, training_labels, seed=0)

        titles = ['1. Mean of the integrated profile',
                  '2. Standard deviation of the integrated profile',
                  '3. Excess kurtosis of the integrated profile',
                  '4. Excess kurtosis of the integrated profile',
                  '5. Mean of the DM-SNR curve',
                  '6. Standard deviation of the DM-SNR curve',
                  '7. Excess kurtosis of the DM-SNR curve',
                  '8. Skewness of the DM-SNR curve']

        mvg = MVG(dtr, ltr, variant=variant)
        mvg.train_model()
        mvg.classify(dte, np.array([1-pi, pi]))
        # cm = build_confusion_matrix(lte, predictions)
        # print(cm)

        llrs = mvg.get_llrs()
        min_dcf = compute_min_DCF(llrs, lte, pi, 1, 1)
        # print(min_dcf)
        table.add_row([f'{variant}, m={m}, data:{data_types[d]}, π={pi}', round(min_dcf, 3)])

    print(table)


def LR_simulations(training_data, z_normalized_training_data, z_gauss_training_data, training_labels):
    datas = [z_normalized_training_data]
    data_types = ['z-normalized']
    ds = list(range(len(datas)))
    m = [None]
    pis = [0.1, 0.5, 0.9]
    lambdas = np.logspace(-5, 5, 50)
    # datas = [training_data, z_normalized_training_data, z_gauss_training_data]
    # data_types = ['raw', 'z-normalized', 'z-normalized + gaussianized']
    # ds = list(range(3))
    # m = [None, 7, 5, 4]
    # pis = [0.1, 0.5, 0.9]
    # lambdas = np.logspace(-5, 5, 50)

    hyperparameters = itertools.product(m, ds, pis, lambdas)

    table = PrettyTable()
    table.field_names = ['Hyperparameters', 'min DCF']

    dcfs = dict.fromkeys(pis, [])

    for m, d, pi, lbd in hyperparameters:
        dtr = datas[d]
        if m is not None:
            dtr = PCA(dtr, m)
        (dtr, ltr), (dte, lte) = splitData_SingleFold(dtr, training_labels, seed=0)

        lr = LR(dtr, ltr, lbd, pi)
        lr.train_model()
        lr.classify(dte, np.array([1-pi, pi]))

        llrs = lr.get_llrs()
        min_dcf = compute_min_DCF(llrs, lte, pi, 1, 1)

        # table.add_row([f"PCA m={m}, data:{data_types[d]}, π={pi}, λ={lbd}", round(min_dcf, 3)])
        dcfs[pi].append(min_dcf)

    return dcfs, lambdas


def main():
    (training_data, training_labels), _ = load_dataset()

    z_normalized_training_data = z_normalization(training_data)

    z_gauss_training_data = gaussianize(z_normalized_training_data, z_normalized_training_data)

    dcfs, lambdas = LR_simulations(training_data, z_normalized_training_data, z_gauss_training_data, training_labels)
    for k, v in dcfs.items():
        np.save(f"pi{k}", np.array(v))
    # print(table)
    plt.figure()
    colors = ['red', 'green', 'blue']
    for i, (k, v) in enumerate(dcfs.items()):
        plt.plot(lambdas, v, color=colors[i], label=f'π={k}')
    plt.legend()
    plt.xticks()
    plt.show()



if __name__ == '__main__':
    main()
