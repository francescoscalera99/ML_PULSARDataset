import itertools
import os

import distinctipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager


# from preprocessing.preprocessing import gaussianize


def plot_histogram(array, labels, titles, nbins: int = 10) -> None:
    """
    Plots the histogram of each feature of the given array
    :param array: the (training) dataset
    :param labels: the labels of parameter array
    :param titles: the array for the titles of the histograms (the names of each feature)
    :param nbins: the number of bins for the histograms
    """
    fig, axs = plt.subplots(2, 4)
    hs = []
    for j in range(array.shape[0]):
        # for j in range(1):
        f = plt.gcf()
        for i in range(len(set(labels))):
            h = axs[j // 4, j % 4].hist(array[j, labels == i], label=f'Class {bool(i)}', bins=nbins, density=True,
                                        alpha=0.7)
            hs.append(h)
        axs[j // 4, j % 4].set_title(titles[j], size='medium')
    fig.set_size_inches((15, 5))
    fig.legend([hs[0], hs[1]],
               labels=['Class False', 'Class True'],
               bbox_to_anchor=(0.47, 0.4, 0.5, 0.5),
               borderaxespad=0.1,
               prop={'size': 12}
               )
    fig.tight_layout()
    fig.show()
    fig.savefig(fname=f'outputs/gauss_features')


def create_heatmap(dataset, labels, cmap='Reds', title=None):
    """
    :param dataset:
    :param cmap:
    :param title:
    :return:
    """
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches((15, 5))
    heatmap = np.abs(np.corrcoef(dataset))
    axs[0].set_title("Whole dataset", size='xx-large')
    sns.heatmap(heatmap, cmap='Greys', annot=True, ax=axs[0])

    heatmap = np.abs(np.corrcoef(dataset[:, labels == 1]))
    axs[1].set_title("Positive class", size='xx-large')
    sns.heatmap(heatmap, cmap='Oranges', annot=True, ax=axs[1])

    heatmap = np.abs(np.corrcoef(dataset[:, labels == 0]))
    axs[2].set_title("Negative class", size='xx-large')
    sns.heatmap(heatmap, cmap='Blues', annot=True, ax=axs[2])
    fig.tight_layout()
    fig.show()
    fig.savefig(fname=f'outputs/gauss_heatmap')


def create_scatterplots(training_data, training_labels, datatype=None):
    num_features = training_data.shape[0]
    num_classes = len(set(training_labels))
    colors = ['red', 'blue']

    titles = ['1. Mean of the integrated profile',
              '2. Standard deviation of the integrated profile',
              '3. Excess kurtosis of the integrated profile',
              '4. Excess kurtosis of the integrated profile',
              '5. Mean of the DM-SNR curve',
              '6. Standard deviation of the DM-SNR curve',
              '7. Excess kurtosis of the DM-SNR curve',
              '8. Skewness of the DM-SNR curve']

    for n, (i, j) in enumerate(itertools.combinations(range(num_features), 2)):
        for k in range(num_classes):
            plt.scatter(training_data[i, training_labels == k], training_data[j, training_labels == k], color=colors[k],
                        label=f"Class {int(k)}")
        plt.xlabel(titles[i])
        plt.ylabel(titles[j])
        plt.legend()
        plt.title(f"Plot {n + 1}")
        # plt.show()
        plt.savefig(fname=f'outputs/scatter/{datatype}Figure{n + 1}')
        plt.cla()


def create_scatterplots2(training_data, training_labels):
    num_features = training_data.shape[0]
    num_classes = len(set(training_labels))
    colors = ['red', 'blue']

    raw_data = training_data
    gaussianized_data = 0  # gaussianize(training_data, training_data)

    titles = ['1. Mean of the integrated profile',
              '2. Standard deviation of the integrated profile',
              '3. Excess kurtosis of the integrated profile',
              '4. Excess kurtosis of the integrated profile',
              '5. Mean of the DM-SNR curve',
              '6. Standard deviation of the DM-SNR curve',
              '7. Excess kurtosis of the DM-SNR curve',
              '8. Skewness of the DM-SNR curve']

    for n, (i, j) in enumerate(itertools.combinations(range(num_features), 2)):
        f, axs = plt.subplots(1, 2)
        f.set_size_inches(20, 10)
        for k in range(num_classes):
            axs[0].scatter(raw_data[i, training_labels == k], raw_data[j, training_labels == k], color=colors[k],
                           label=f"Class {int(k)}")
            axs[1].scatter(gaussianized_data[i, training_labels == k], gaussianized_data[j, training_labels == k],
                           color=colors[k], label=f"Class {int(k)}")
        axs[0].set_xlabel(titles[i])
        axs[1].set_xlabel(titles[i])
        axs[0].set_ylabel(titles[j])
        axs[1].set_ylabel(titles[j])
        f.legend()
        f.suptitle(f"Plot {n + 1}")
        # plt.show()
        f.tight_layout()
        f.savefig(fname=f'outputs/scatter/Figure{n + 1}')
        plt.close(f)


def plot_lambda():
    lbd_values = np.logspace(-8, -5, 20)
    lbd_values = np.array([0, *lbd_values])
    lbd_values2 = np.logspace(-5, 5, 50)

    lbd_values = np.array([*lbd_values, *lbd_values2[1:]])
    m_values = [False, None, 7, 5]
    prior = [0.5, 0.1, 0.9]

    i = 0
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 8)
    # fig.suptitle('Tuning hyperparameter λ')
    plt.rcParams['text.usetex'] = True

    colors = ['red', 'blue', 'green']
    for m in m_values:
        for j, pi in enumerate(prior):
            DCF1 = np.load(
                f"../simulations/LR/LR_0_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")

            DCF2 = np.load(
                f"../simulations/LR/LR_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")

            DCFs = np.array([*DCF1, *DCF2[1:]])

            axs[i // 2, i % 2].plot(lbd_values, DCFs, color=colors[j], label=r"$\widetilde{\pi}=$" + f"{pi}")

            if m == False:
                axs[i // 2, i % 2].set_title(f'5-fold, Raw features')
            elif m is None:
                axs[i // 2, i % 2].set_title(f'5-fold, no PCA')
            else:
                axs[i // 2, i % 2].set_title(f'5-fold, PCA (m={m})')

            axs[i // 2, i % 2].set_xlabel('λ')
            axs[i // 2, i % 2].set_ylabel('minDCF')
            axs[i // 2, i % 2].set_xscale('log')

            xticks = [1.e-09, 1.e-08, 1.e-06, 1.e-04, 1.e-02, 1.e+00, 1.e+02, 1.e+04, 1.e+06]
            xlabels = [r"$0$", r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^0$", r"$10^2$", r"$10^4$",
                       r"$10^6$"]

            axs[i // 2, i % 2].set_xticks(xticks, xlabels)
            axs[i // 2, i % 2].get_xaxis().get_major_formatter().labelOnlyBase = False
        i += 1
    # fig.set_size_inches(10, 10)
    # fig.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:3], labels[:3], loc=10, prop={'size': 10})
    # fig.legend(lines[:3], labels[:3], loc=10, prop={'size': 10})
    # fig.subplots_adjust(wspace=0.3, hspace=0.6)
    # fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    fig.show()


def plot_tuningPolySVM():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7, 5]
    K_values = [0.0, 1.0]
    c_values = [0, 1, 10, 15]

    num_colors = len(K_values) * len(c_values)
    # colors = distinctipy.get_colors(num_colors, pastel_factor=0.7)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 28,
        "axes.labelsize": 30,
        "legend.fontsize": 15,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
    })

    i = 0
    fig, axs = plt.subplots(1, 4, sharey="row")
    for m in m_values:
        hyperparameters = itertools.product(K_values, c_values)
        for j, (K, c) in enumerate(hyperparameters):
            DCFs = np.load(
                f"../simulations/polySVM/K{str(K).replace('.', '-')}_c{str(c).replace('.', '-')}_PCA{str(m)}.npy")
            axs[i].plot(C_values, DCFs, color=colors8[j], label=rf"$K={K}$, $c={c}$", linewidth=2.5)
            if (m == False):
                axs[i].set_title(f'5-fold, Raw features')
            else:
                axs[i].set_title(f'5-fold, PCA (m = {m})')
            if i == 0:
                axs[i].legend(ncol=1, loc="upper right")
            axs[i].set_xlabel(r"$C$")
            axs[i].set_ylabel(r"$minDCF$")
            axs[i].set_xscale('log')
            axs[i].set_xticks([10 ** i for i in range(-2, 3, 2)])
        i += 1
    fig.set_size_inches(20, 5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.show()

    label_params = axs[0].get_legend_handles_labels()
    figl, axl = plt.subplots(figsize=(6.5, 7))
    axl.axis(False)
    axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl.show()


def plot_tuningRBFSVM():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7]
    K_values = [0.0, 1.0]
    # gamma_values = [1e-2, 1e-3, 1e-4]
    gamma_exp = [-2, -3, -4]
    i = 0
    plt.clf()

    # fig.suptitle('RBF SVM')
    print(plt.rcParams)
    plt.rcParams.update({
        "text.usetex": True,
        "axes.titlesize": 28,
        "axes.labelsize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
    })
    print("****************")
    print(plt.rcParams)

    fig, axs = plt.subplots(1, 3, sharey="all")

    # num_colors = len(K_values) * len(gamma_values)
    # colors = distinctipy.get_colors(num_colors, pastel_factor=0.7)

    for m in m_values:
        hyperparameters = itertools.product(K_values, gamma_exp)
        for j, (K, g_exp) in enumerate(hyperparameters):
            gamma = 10 ** g_exp
            DCFs = np.load(
                f"../simulations/RBF/RBF_K{str(K).replace('.', '-')}_c{str(gamma).replace('.', '-')}_PCA{str(m)}.npy")
            lb = r"$\gamma = 10^{" + str(g_exp) + "}$"
            axs[i].plot(C_values, DCFs, color=colors6[j], label=rf"$K={K}$, " + lb, linewidth=2.5)
            if m == False:
                axs[i].set_title(f'5-fold, Raw features')
            else:
                axs[i].set_title(rf"5-fold, PCA ($m = {m}$)")
            if i == 0:
                axs[i].legend(ncol=1)

            axs[i].set_xlabel(r"$C$")
            axs[i].set_ylabel(r"$minDCF$")
            axs[i].set_xscale('log')
            axs[i].set_xticks([10 ** i for i in range(-2, 3, 2)])
        i += 1

    fig.set_size_inches(15, 5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.show()

    # label_params = axs[0].get_legend_handles_labels()
    # figl, axl = plt.subplots(figsize=(6.5, 7))
    # axl.axis(False)
    # axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    # figl.show()


def plot_tuningLinearSVMUnbalanced():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7, 5]
    K_values = [1.0, 10.0]
    priors = [0.5, 0.1, 0.9]

    i = 0

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 22,
        "axes.labelsize": 30,
        "legend.fontsize": 20,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
    })

    fig, axs = plt.subplots(1, 4, sharey='row')
    # colors = distinctipy.get_colors(6, pastel_factor=0.7)
    for m in m_values:
        hyperparameters = itertools.product(K_values, priors)
        for j, (K, p) in enumerate(hyperparameters):
            DCFs = np.load(
                f"../simulations/linearSVM/unbalanced/new/K{str(K).replace('.', '-')}_p{str(p).replace('.', '-')}_PCA{str(m)}.npy")
            axs[i].plot(C_values, DCFs, color=colors[j], label=r"$K=" + str(K) + r",\;\widetilde{\pi}=" + str(p) + r"$",
                        linewidth=3)
            if m is None:
                axs[i].set_title(rf"Gau, no PCA, $\pi_T=0.5$")
            elif m == False:
                axs[i].set_title(rf"Raw, no PCA, $\pi_T=0.5$")
            else:
                axs[i].set_title(rf"Gau, PCA ($m = {m}$), $\pi_T=0.5$")
            # axs[i].legend()
            axs[i].set_xlabel('$C$')
            axs[i].set_ylabel('$minDCF$')
            axs[i].set_xscale('log')
            axs[i].yaxis.set_tick_params(labelbottom=True)
            axs[i].set_xticks([10 ** i for i in range(-2, 3, 2)])
        i += 1
    fig.set_size_inches(20, 5)
    fig.tight_layout()
    fig.show()

    label_params = axs[0].get_legend_handles_labels()
    figl, axl = plt.subplots(figsize=(6.5, 5))
    axl.axis(False)
    axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl.show()


def plot_tuning_LinearSVMBalanced():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7, 5]
    pi_T_values = [0.5, 0.1, 0.9]
    K_values = [1.0, 10.0]
    prior = [0.5, 0.1, 0.9]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "legend.fontsize": 20,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
    })

    i = 0
    fig, axs = plt.subplots(4, 3, sharey='all')

    for m in m_values:
        j = 0
        for pi_T in pi_T_values:
            hyperparameters = itertools.product(K_values, prior)
            for idx, (K, pi) in enumerate(hyperparameters):
                DCFs = np.load(
                    f"../simulations/linearSVM/balanced/K{str(K).replace('.', '-')}_p{str(pi).replace('.', '-')}_pT{str(pi_T).replace('.', '-')}_PCA{str(m)}.npy")
                axs[i, j].plot(C_values, DCFs, color=colors[idx],
                               label=rf"$K={K}$,\;" + r"$\widetilde{\pi}=$" + f"{pi}", linewidth=3)
                if m is None:
                    axs[i, j].set_title('Gau, no PCA' + rf", $\pi_T={pi_T}$")
                elif m == False:
                    axs[i, j].set_title('Raw, no PCA, ' + rf"$\pi_T={pi_T}$")
                else:
                    axs[i, j].set_title('Gau, PCA ' + rf"($m = {m}$), $\pi_T={pi_T}$")
                # axs[i, j].legend()
                axs[i, j].set_xlabel(r"$C$")
                axs[i, j].set_ylabel(r"$minDCF$")
                axs[i, j].set_xscale('log')
                axs[i, j].yaxis.set_tick_params(labelbottom=True)
            j += 1
        i += 1

    fig.set_size_inches(20, 20)
    fig.tight_layout()

    label_params = axs[0, 0].get_legend_handles_labels()

    figl, axl = plt.subplots(figsize=(6.5, 5))
    axl.axis(False)
    axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl.show()

    fig.show()


def plot_tuningGMM():
    variants = ['full-cov', 'diag', 'tied']
    raw = [True, False]
    m_values = [None, 7]
    pis = [0.1, 0.5, 0.9]
    components_values = [rf"${2 ** i}$" for i in range(9)]

    # the datasets are
    # * raw, no PCA
    # * raw, PCA7
    # * gau, no PCA
    # * gau, PCA7

    # in each figure, given PCA and variant, we will have raw and gaussian for every value of G
    # => in each plot there are 8*2 = 16 bars
    # we will have 6 plots

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 22,
        "xtick.labelsize": 30,
        "ytick.labelsize": 20,
        "legend.fontsize": 15
    })

    n = len(raw)  # Number of bars to plot
    w = .3  # With of each column
    x = np.arange(len(components_values))  # Center position of group on x axis
    y = np.arange(0.0, 1.1, 0.2)
    print(y)

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(27, 18)

    for j, (variant, m) in enumerate(itertools.product(variants, m_values)):
        for i, r in enumerate(raw):
            DCFs = np.load(f"simulations/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}.npy")
            position = x + (w * (1 - n) / 2) + i * w
            axs[j // 3, j % 3].bar(position, DCFs, width=w, edgecolor='black')
        axs[j // 3, j % 3].set_xticks(x, components_values)
        axs[j // 3, j % 3].set_yticks(y)

    plt.tight_layout()
    plt.show()
    # plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.2f'))


def plot_tuningGMM2():
    variants = ['full-cov', 'diag', 'tied']
    raw = [True, False]
    m_values = [None, 7]
    pis = [0.5, 0.1, 0.9]
    components_values = [2 ** i for i in range(9)]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "legend.fontsize": 20,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
    })

    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(15, 10)
    # colors = distinctipy.get_colors(6, pastel_factor=0.7)
    y = np.arange(0.0, 1.1, 0.2)

    for i, (variant, m) in enumerate(itertools.product(variants, m_values)):
        for j, (p, r) in enumerate(itertools.product(pis, raw)):
            pp = '' if p == 0.5 else "_pi" + str(p).replace('.', '-')
            DCFs = np.load(f"../simulations/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}{pp}.npy")
            label = r"$\widetilde{\pi}=$" + f"{p}, {'raw' if r else 'gau'}"
            axs[i // 2, i % 2].plot(components_values, DCFs, label=label, color=colors6[j], linewidth=2)
            axs[i // 2, i % 2].set_xscale('log', base=2)
            axs[i // 2, i % 2].set_xticks(components_values)
            axs[i // 2, i % 2].set_yticks(y)
            if i // 2 == 2:
                axs[i // 2, i % 2].set_xlabel("Number of components")

            v = 'tied full-cov' if variant == 'tied' else variant

            pca = f"PCA ($m={m}$)" if m is not None else "no PCA"
            axs[i // 2, i % 2].set_title(rf"{v}, {pca}", size=20)

        axs[i // 2, i % 2].legend(loc='upper right', framealpha=0.5)

    fig.tight_layout()
    plt.show()


def bayes_error_plots(classifier):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    minDCF = np.load(f"simulations/bayesErrorPlot/{classifier.__name__}_minDCF.npy")
    actDCF = np.load(f"simulations/bayesErrorPlot/{classifier.__name__}_actDCF.npy")
    # actDCF_cal = np.load(f"simulations/bayesErrorPlot/{classifier.__name__}_actDCF_Calibrated.npy")

    plt.clf()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        # "axes.titlesize": 22,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
        "legend.fontsize": 12
    })

    plt.plot(effPriorLogOdds, minDCF, label=r"$minDCF$", color="red")
    plt.plot(effPriorLogOdds, actDCF, label=r"$actDCF$", color="blue")
    # plt.plot(effPriorLogOdds, actDCF_cal, label=r"$actDCF$ (cal.)", color="blue", linestyle="dashed")
    plt.legend()
    # plt.title(classifier.__name__)
    plt.xlabel(r"$\log{\frac{\widetilde{\pi}}{(1 - \widetilde{\pi})}}$")
    plt.ylabel(r"$DCF$")
    plt.savefig(fname=f"outputs/bayes_error_plots/beforecal_{classifier.__name__}")
    plt.tight_layout()
    plt.show()


def bayes_error_plots2(classifiers, after=False):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    plt.clf()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
        "legend.fontsize": 12
    })

    fig, axs = plt.subplots(2, 2, sharex="col", sharey="row")

    fig.set_size_inches(10, 8)

    for i, classifier in enumerate(classifiers):
        minDCF = np.load(f"simulations/bayesErrorPlot/{classifier.__name__}_minDCF.npy")
        actDCF = np.load(f"simulations/bayesErrorPlot/{classifier.__name__}_actDCF.npy")
        axs[i // 2, i % 2].plot(effPriorLogOdds, minDCF, label=r"$minDCF$", color="orange", linewidth=2)
        axs[i // 2, i % 2].plot(effPriorLogOdds, actDCF, label=r"$actDCF$", color="dodgerblue", linewidth=2)
        if after:
            actDCF_cal = np.load(f"simulations/bayesErrorPlot/{classifier.__name__}_actDCF_Calibrated.npy")
            axs[i // 2, i % 2].plot(effPriorLogOdds, actDCF_cal, label=r"$actDCF$ (cal.)", linestyle="dashed",
                                    color="dodgerblue", linewidth=2)
        axs[i // 2, i % 2].legend()
        axs[i // 2, i % 2].set_title(classifier.__name__)
        axs[i // 2, i % 2].set_xticks(list(range(-3, 4)))
        axs[i // 2, i % 2].set_yticks(np.arange(0, 1.1, 0.2))
        axs[i // 2, i % 2].xaxis.set_tick_params(labelbottom=True)
        axs[i // 2, i % 2].yaxis.set_tick_params(labelbottom=True)
        if i > 1:
            axs[i // 2, i % 2].set_xlabel(r"$\log{\frac{\widetilde{\pi}}{(1 - \widetilde{\pi})}}$")
        if i % 2 == 0:
            axs[i // 2, i % 2].set_ylabel(r"$DCF$")

    fname = "aftercal" if after else "beforecal"

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    fig.savefig(fname=f"outputs/bayes_error_plots/bep_{fname}")
    plt.show()

def plot_lambda_evaluation():
    lbd_values = np.logspace(-8, -5, 20)
    lbd_values = np.array([0, *lbd_values])
    lbd_values2 = np.logspace(-5, 5, 50)

    lbd_values = np.array([*lbd_values, *lbd_values2[1:]])
    m_values = [False, None, 7, 5]
    prior = [0.5, 0.1, 0.9]

    i = 0
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 8)
    # fig.suptitle('Tuning hyperparameter λ')
    plt.rcParams['text.usetex'] = True

    colors = ['red', 'blue', 'green']
    for m in m_values:
        for j, pi in enumerate(prior):
            DCF1 = np.load(
                f"./../simulations/LR/LR_0_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")

            DCF2 = np.load(
                f"./../simulations/LR/LR_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")

            DCFs = np.array([*DCF1, *DCF2[1:]])
            DCFs_evaluation = np.load(f"./../simulations/evaluation/LR/LR_EVAL_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")
            axs[i // 2, i % 2].plot(lbd_values, DCFs, color=colors[j], label=r"$\widetilde{\pi}=$" + f"{pi}")
            axs[i // 2, i % 2].plot(lbd_values, DCFs_evaluation, color=colors[j], label=r"$\widetilde{\pi}=$" + f"{pi} (eval.)")
            if m == False:
                axs[i // 2, i % 2].set_title(f'5-fold, Raw features')
            elif m is None:
                axs[i // 2, i % 2].set_title(f'5-fold, no PCA')
            else:
                axs[i // 2, i % 2].set_title(f'5-fold, PCA (m={m})')

            axs[i // 2, i % 2].set_xlabel('λ')
            axs[i // 2, i % 2].set_ylabel('minDCF')
            axs[i // 2, i % 2].set_xscale('log')

            xticks = [1.e-09, 1.e-08, 1.e-06, 1.e-04, 1.e-02, 1.e+00, 1.e+02, 1.e+04, 1.e+06]
            xlabels = [r"$0$", r"$10^{-8}$", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^0$", r"$10^2$", r"$10^4$",
                       r"$10^6$"]

            axs[i // 2, i % 2].set_xticks(xticks, xlabels)
            axs[i // 2, i % 2].get_xaxis().get_major_formatter().labelOnlyBase = False
        i += 1
    # fig.set_size_inches(10, 10)
    # fig.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:3], labels[:3], loc=10, prop={'size': 10})
    # fig.legend(lines[:3], labels[:3], loc=10, prop={'size': 10})
    # fig.subplots_adjust(wspace=0.3, hspace=0.6)
    # fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    # colors8 = distinctipy.get_colors(8, pastel_factor=1, colorblind_type='Deuteranomaly')
    # print(colors8)
    colors6 = [(0.48702807223549177, 0.4242891647177821, 0.9480975665882982), (0.9146761531779931, 0.4970424422244128, 0.41460357267068376), (0.843602824944377, 0.6031154951690304, 0.9802318468625552), (0.5887251240359368, 0.9624135405893406, 0.4585532945832182), (0.422567523593921, 0.44218101996887993, 0.5516040738892886), (0.43399916426535, 0.7098723267606655, 0.6255076508970907)]
    # colors8 = [(0.5450484248310105, 0.5130972742328073, 0.5102488831581509),
    #            (0.6109330873928905, 0.7193582681286009, 0.9814590256707204),
    #            (0.9727770320054765, 0.7854905796839438, 0.5145282365057959),
    #            (0.9806065670005477, 0.5066792697066322, 0.7311620666921056),
    #            (0.565920914907729, 0.9141080668353584, 0.7641066636691687),
    #            (0.5114677713143507, 0.5061193495393317, 0.9951605179132765),
    #            (0.5830073483609048, 0.5244350779880778, 0.7931264147573027),
    #            (0.5692188040526873, 0.7826586898074446, 0.5098679245540738)]
    # plot_lambda()
    # plot_tuningPolySVM()
    # plot_tuningRBFSVM()
    # plot_tuningLinearSVMUnbalanced()
    # plot_tuning_LinearSVMBalanced()

    # print(os.path.abspath("."))

    plot_tuningGMM2()
    pass
