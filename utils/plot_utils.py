import itertools
import os

import distinctipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from utils.metrics_utils import compute_FPRs_TPRs
from utils.utils import k_fold
from classifiers.Classifier import ClassifierClass



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
    fig.savefig(fname=f'plots/gauss_features')


def create_heatmap(dataset, labels):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 25,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
        "legend.fontsize": 18,
        "figure.dpi": 180
    })

    datasets = (dataset, gaussianize(dataset, dataset))

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    for i, d in enumerate(datasets):
        data_map1 = np.abs(np.corrcoef(d))
        im1, cbar1 = heatmap(data_map1, [i for i in range(8)], [i for i in range(8)], ax=ax[i, 0], cmap="Greys")
        annotate_heatmap(im1, size=13, fontweight="bold")

        data_map2 = np.abs(np.corrcoef(d[:, labels == 1]))
        im2, cbar2 = heatmap(data_map2, [i for i in range(8)], [i for i in range(8)], ax=ax[i, 1], cmap="Oranges")
        annotate_heatmap(im2, size=13, fontweight="bold")

        data_map3 = np.abs(np.corrcoef(d[:, labels == 0]))
        im3, cbar3 = heatmap(data_map3, [i for i in range(8)], [i for i in range(8)], ax=ax[i, 2], cmap="Blues")
        annotate_heatmap(im3, size=13, fontweight="bold")

        if i == 0:
            ax[i, 0].set_title("Whole dataset")
            ax[i, 1].set_title("Class True")
            ax[i, 2].set_title("Class False")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    fig.show()
    fig.savefig(fname=f"plots/dataset/heatmaps")


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if cbar_kw is None:
        cbar_kw = {}
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, va="bottom")
    cbar.outline.set_linewidth(0)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def create_scatterplots(training_data, training_labels):
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
        f.savefig(fname=f'plots/scatter/Figure{n + 1}')
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
            axs[i].plot(C_values, DCFs, color=colors6[j],
                        label=r"$K=" + str(K) + r",\;\widetilde{\pi}=" + str(p) + r"$",
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
                axs[i, j].plot(C_values, DCFs, color=colors6[idx],
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
    pis = [0.5, 0.1, 0.9]
    components_values = [2 ** i for i in range(9)]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 30,
        "axes.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(12, 10)
    # colors = distinctipy.get_colors(6, pastel_factor=0.7)
    y = np.arange(0.0, 1.1, 0.2)

    for i, (variant, m) in enumerate(itertools.product(variants, m_values)):
        for j, (p, r) in enumerate(itertools.product(pis, raw)):
            pp = '' if p == 0.5 else "_pi" + str(p).replace('.', '-')
            DCFs = np.load(f"../simulations/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}{pp}.npy")
            label = r"$\widetilde{\pi}=$" + f"{p}, {'raw' if r else 'gau'}"
            axs[i // 2, i % 2].plot(components_values, DCFs, label=label, color=colors6[j], linewidth=3)
            axs[i // 2, i % 2].set_xscale('log', base=2)
            axs[i // 2, i % 2].set_xticks(components_values)
            axs[i // 2, i % 2].set_yticks(y)
            if i // 2 == 2:
                axs[i // 2, i % 2].set_xlabel("Number of components")

            v = 'tied full-cov' if variant == 'tied' else variant

            pca = f"PCA ($m={m}$)" if m is not None else "no PCA"
            axs[i // 2, i % 2].set_title(rf"{v}, {pca}", size=20)

        # axs[i // 2, i % 2].legend(loc='upper right', framealpha=0.5)

    fig.tight_layout()
    # plt.show()
    fig.savefig(fname="../plots/tuning_GMM2", dpi=180)

    label_params = axs[0, 0].get_legend_handles_labels()
    figl, axl = plt.subplots(figsize=(6.5, 5))
    axl.axis(False)
    axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    # figl.show()
    figl.savefig(fname="../plots/tuning_GMM_legend")


def bayes_error_plots(classifiers, after=False, evaluation=False):
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
        "legend.fontsize": 18,
        "figure.dpi": 180
    })

    fig, axs = plt.subplots(2, 2, sharex="col", sharey="row")

    fig.set_size_inches(10, 8)

    folder = "/evaluation" if evaluation else ""

    for i, classifier in enumerate(classifiers):
        minDCF = np.load(f"simulations{folder}/bayesErrorPlot/{classifier.__name__}_minDCF.npy")
        actDCF = np.load(f"simulations{folder}/bayesErrorPlot/{classifier.__name__}_actDCF.npy")
        axs[i // 2, i % 2].plot(effPriorLogOdds, minDCF, label=r"$minDCF$", color="orange", linewidth=2)
        axs[i // 2, i % 2].plot(effPriorLogOdds, actDCF, label=r"$actDCF$", color="dodgerblue", linewidth=2)
        if after:
            actDCF_cal = np.load(f"simulations{folder}/bayesErrorPlot/{classifier.__name__}_actDCF_Calibrated.npy")
            axs[i // 2, i % 2].plot(effPriorLogOdds, actDCF_cal, label=r"$actDCF$ (cal.)", linestyle="dashed",
                                    color="dodgerblue", linewidth=2)
        axs[i // 2, i % 2].legend()
        axs[i // 2, i % 2].set_title(classifier.__name__)
        axs[i // 2, i % 2].set_xticks(list(range(-3, 4)))
        #axs[i // 2, i % 2].set_yticks(np.arange(0, 1.1, 0.2))
        axs[i // 2, i % 2].xaxis.set_tick_params(labelbottom=True)
        axs[i // 2, i % 2].yaxis.set_tick_params(labelbottom=True)
        if i > 1:
            axs[i // 2, i % 2].set_xlabel(r"$\log{\frac{\widetilde{\pi}}{(1 - \widetilde{\pi})}}$")
        if i % 2 == 0:
            axs[i // 2, i % 2].set_ylabel(r"$DCF$")

    fname = "aftercal" if after else "beforecal"

    if evaluation:
        fname += "_eval"

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    fig.savefig(fname=f"plots/bayes_error_plots/bep_{fname}", dpi=180)
    fig.show()


def plot_lambda_evaluation():
    lbd_values = np.logspace(-8, 5, 70)
    lbd_values = np.array([0, *lbd_values])

    m_values = [False, None, 7, 5]
    prior = [0.5, 0.1, 0.9]

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 8)

    colors = ['red', 'blue', 'green']
    for i, m in enumerate(m_values):
        for j, pi in enumerate(prior):
            DCFs = np.load(
                f"./../simulations/LR/LR_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")

            DCFs_evaluation = np.load(
                f"./../simulations/evaluation/LR/LR_EVAL_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")
            axs[i // 2, i % 2].plot(lbd_values, DCFs, color=colors[j], label=r"$\widetilde{\pi}=" + f"{pi}$",
                                    linestyle="dashed")
            axs[i // 2, i % 2].plot(lbd_values, DCFs_evaluation, color=colors[j],
                                    label=r"$\widetilde{\pi}=" + f"{pi}$ (eval.)")
            if m == False:
                axs[i // 2, i % 2].set_title(f'5-fold, Raw features')
            elif m is None:
                axs[i // 2, i % 2].set_title(f'5-fold, no PCA')
            else:
                axs[i // 2, i % 2].set_title(f'5-fold, PCA (m={m})')

            axs[i // 2, i % 2].set_xlabel(r'$\lambda$')
            axs[i // 2, i % 2].set_ylabel(r'$minDCF$')
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
    fig.legend(lines[:6], labels[:6], loc=10, ncol=3)
    # fig.legend(lines[:3], labels[:3], loc=10, prop={'size': 10})

    fig.tight_layout()
    # fig.subplots_adjust(top=0.88)
    fig.subplots_adjust(hspace=0.7)
    fig.show()

    fig.savefig(fname="../plots/evaluation/lambda", dpi=180)


def plot_tuningLinearSVMUnbalanced_evaluation():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7, 5]
    K_values = [1.0]
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
                f"./../simulations/linearSVM/unbalanced/new/K{str(K).replace('.', '-')}_p{str(p).replace('.', '-')}_PCA{str(m)}.npy")
            DCFs_evaluation = np.load(
                f"../simulations/evaluation/linearSVM/unbalanced/K{str(K).replace('.', '-')}_p{str(p).replace('.', '-')}_PCA{str(m)}.npy")
            axs[i].plot(C_values, DCFs, color=colors6[j], linestyle="dashed", label=r"$K=" + str(K) + r",\;\widetilde{\pi}=" + str(p) + r"$",
                        linewidth=3)
            axs[i].plot(C_values, DCFs_evaluation, color=colors6[j],
                        label=r"$K=" + str(K) + r",\;\widetilde{\pi}=" + str(p) + r"$" + "(eval.)",
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


def plot_tuning_LinearSVMBalanced_evaluation():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7, 5]
    pi_T_values = [0.5, 0.1, 0.9]
    K_values = [1.0]
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
                DCFs_evaluation = np.load(
                    f"../simulations/evaluation/linearSVM/balanced/K{str(K).replace('.', '-')}_p{str(pi).replace('.', '-')}_pT{str(pi_T).replace('.', '-')}_PCA{str(m)}.npy")
                axs[i, j].plot(C_values, DCFs,  linestyle="dashed", color=colors6[idx],
                               label=rf"$K={K}$,\;" + r"$\widetilde{\pi}=$" + f"{pi}", linewidth=3)
                axs[i, j].plot(C_values, DCFs_evaluation, color=colors6[idx],
                               label=rf"$K={K}$,\;" + r"$\widetilde{\pi}=$" + f"{pi} (eval.)", linewidth=3)
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
    fig.savefig(fname="../plots/evaluation/tuning_SVM_linear_balanced", dpi=180)


def plot_tuningPolySVM_evaluation():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7, 5]
    K_values = [1.0]
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
            DCFs_evaluation = np.load(
                f"../simulations/evaluation/polySVM/K{str(K).replace('.', '-')}_c{str(c).replace('.', '-')}_PCA{str(m)}.npy")

            axs[i].plot(C_values, DCFs, color=colors8[j], linestyle="dashed",label=rf"$K={K}$, $c={c}$", linewidth=2.5)
            axs[i].plot(C_values, DCFs_evaluation, color=colors8[j], label=rf"$K={K}$, $c={c}$ (eval.)", linewidth=2.5)
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

    fig.savefig(fname="../plots/evaluation/tuning_PolySVM_evaluation", dpi=180)


def plot_tuningRBFSVM_evaluation():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7]
    K_values = [0.0]
    gamma_exp = [-2, -3, -4]
    i = 0
    plt.clf()

    # fig.suptitle('RBF SVM')
    plt.rcParams.update({
        "text.usetex": True,
        "axes.titlesize": 28,
        "axes.labelsize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
    })

    fig, axs = plt.subplots(1, 3, sharey="all")

    # num_colors = len(K_values) * len(gamma_values)
    # colors = distinctipy.get_colors(num_colors, pastel_factor=0.7)

    for m in m_values:
        hyperparameters = itertools.product(K_values, gamma_exp)
        for j, (K, g_exp) in enumerate(hyperparameters):
            gamma = 10 ** g_exp
            DCFs = np.load(
                f"../simulations/RBF/RBF_K{str(K).replace('.', '-')}_c{str(gamma).replace('.', '-')}_PCA{str(m)}.npy")
            DCFs_evaluation = np.load(
                f"../simulations/evaluation/RBF/RBF_K{str(K).replace('.', '-')}_gamma{str(gamma).replace('.', '-')}_PCA{str(m)}.npy")
            lb = r"$\gamma = 10^{" + str(g_exp) + "}$"
            axs[i].plot(C_values, DCFs, color=colors6[j], linestyle="dashed", label=rf"$K={K}$, " + lb, linewidth=2.5)
            axs[i].plot(C_values, DCFs_evaluation, color=colors6[j], label=rf"$K={K}$, " + lb + " (eval.)", linewidth=2.5)
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
    fig.savefig(fname="../plots/evaluation/tuning_RBFSVM_evaluation", dpi=180)
    # label_params = axs[0].get_legend_handles_labels()
    # figl, axl = plt.subplots(figsize=(6.5, 7))
    # axl.axis(False)
    # axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    # figl.show()


def plot_tuningGMM_evaluation():
    variants = ['full-cov', 'diag', 'tied']
    raw = [True, False]
    m_values = [7]
    pis = [0.5, 0.1, 0.9]
    components_values = [2 ** i for i in range(9)]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 30,
        "axes.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    fig1, axs = plt.subplots(3, 2)
    fig1.set_size_inches(12, 10)

    y = np.arange(0.0, 1.1, 0.2)

    for i, (variant, m) in enumerate(itertools.product(variants, m_values)):
        for j, (p, r) in enumerate(itertools.product(pis, raw)):
            pp = '' if p == 0.5 else "_pi" + str(p).replace('.', '-')
            DCFs = np.load(f"../simulations/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}{pp}.npy")
            DCFs_evaluation = np.load(f"../simulations/evaluation/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}_pi{str(p).replace('.', '-')}.npy")
            label = r"$\widetilde{\pi}=$" + f"{p} "
            axs[i, 1-int(r)].plot(components_values, DCFs, linestyle="dashed", label=label, color=colors3[j//2], linewidth=3)
            axs[i, 1-int(r)].plot(components_values, DCFs_evaluation, label=label+"(eval.)", color=colors3[j//2], linewidth=3)
            axs[i, 1-int(r)].set_xscale('log', base=2)
            axs[i, 1-int(r)].set_xticks(components_values)
            axs[i, 1-int(r)].set_yticks(y)
            if i // 2 == 2:
                axs[i, 1-int(r)].set_xlabel("Number of components")
            if i % 2 == 0:
                axs[i, 1-int(r)].set_ylabel(r"$DCF$")
            v = 'tied full-cov' if variant == 'tied' else variant

            pca = f"PCA ($m={m}$)" if m is not None else "no PCA"
            axs[i, 1-int(r)].set_title(rf"{v}, {pca}, {'raw' if r else 'gau'}", size=20)

        # axs[i // 2, i % 2].legend(loc='upper right', framealpha=0.5)

    fig1.tight_layout()
    fig1.show()
    fig1.savefig(fname="../plots/evaluation/tuning_GMM2_PCA7", dpi=180)

    label_params1 = axs[0, 0].get_legend_handles_labels()
    figl1, axl1 = plt.subplots(figsize=(5.5, 5.2))
    axl1.axis(False)
    axl1.legend(*label_params1, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl1.show()
    figl1.savefig(fname="../plots/evaluation/tuning_GMM_legend_PCA7")


def ROC_curve(training_data, training_labels, classifiers, args):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
        "legend.fontsize": 18,
        "figure.dpi": 180
    })
    f, ax = plt.subplots()
    colors = distinctipy.get_colors(len(classifiers))
    for i, classifier in enumerate(classifiers):
        score, labels = k_fold(training_data, training_labels, classifier, 5, **args[i])
        FPRs, TPRs = compute_FPRs_TPRs(score, labels)
        ax.plot(FPRs, TPRs, color=colors[i], label=f"{classifier.__name__}")
    ax.set_title("ROC curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.legend()
    plt.grid()
    f.show()
    f.savefig('plots/ROC/ROC_training.png')


def generate_ROC_data(training_data, training_labels, testing_data, testing_labels, classifier, args):
    c = classifier(training_data, training_labels, **args)
    c.train_model(**args)
    c.classify(testing_data, None)
    score = c.get_llrs()
    FPRs, TPRs = compute_FPRs_TPRs(score, testing_labels)
    np.save(f"simulations/evaluation/ROC/{classifier.__name__}_TPRs", TPRs)
    np.save(f"simulations/evaluation/ROC/{classifier.__name__}_FPRs", FPRs)


def ROC_curve_evaluation(classifiers):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
        "legend.fontsize": 18,
        "figure.dpi": 180
    })
    f, ax = plt.subplots()
    colors = distinctipy.get_colors(len(classifiers))
    for i, classifier in enumerate(classifiers):
        FPRs = np.load(f"simulations/evaluation/ROC/{classifier.__name__}_FPRs.npy")
        TPRs = np.load(f"simulations/evaluation/ROC/{classifier.__name__}_TPRs.npy")
        ax.plot(FPRs, TPRs, color=colors[i], label=f"{classifier.__name__}")
    ax.set_title("ROC curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.legend()
    plt.grid()
    f.show()
    f.savefig('plots/ROC/ROC_evaluation.png')


def plot_tuningGMM_evaluation2():
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
        "axes.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    fig1, axs1 = plt.subplots(3, 2)
    fig1.set_size_inches(12, 10)

    fig2, axs2 = plt.subplots(3, 2)
    fig2.set_size_inches(12, 10)

    fig3, axs3 = plt.subplots(3, 2)
    fig3.set_size_inches(12, 10)

    y = np.arange(0.0, 1.1, 0.2)

    axs = [axs1, axs2, axs3]

    for i, (variant, m) in enumerate(itertools.product(variants, m_values)):
        for j, (p, r) in enumerate(itertools.product(pis, raw)):
            pp = '' if p == 0.5 else "_pi" + str(p).replace('.', '-')
            DCFs = np.load(f"../simulations/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}{pp}.npy")
            DCFs_evaluation = np.load(f"../simulations/evaluation/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}_pi{str(p).replace('.', '-')}.npy")
            label = r"$\widetilde{\pi}=$" + f"{p}, {'raw' if r else 'gau'}"
            axs[pis.index(p)][i // 2, i % 2].plot(components_values, DCFs, label=label, linestyle="dashed", color=colors6[j], linewidth=3)
            axs[pis.index(p)][i // 2, i % 2].plot(components_values, DCFs_evaluation, label=label+"(eval.)", color=colors6[j], linewidth=3)
            axs[pis.index(p)][i // 2, i % 2].set_xscale('log', base=2)
            axs[pis.index(p)][i // 2, i % 2].set_xticks(components_values)
            axs[pis.index(p)][i // 2, i % 2].set_yticks(y)
            if i // 2 == 2:
                axs[pis.index(p)][i // 2, i % 2].set_xlabel("Number of components")
            if i % 2 == 0:
                axs[pis.index(p)][i // 2, i % 2].set_ylabel(r"$DCF$")
            v = 'tied full-cov' if variant == 'tied' else variant

            pca = f"PCA ($m={m}$)" if m is not None else "no PCA"
            axs[pis.index(p)][i // 2, i % 2].set_title(rf"{v}, {pca}", size=20)

        # axs[i // 2, i % 2].legend(loc='upper right', framealpha=0.5)

    fig1.tight_layout()
    fig1.show()
    fig1.savefig(fname="../plots/evaluation/tuning_GMM2_pi0-5", dpi=180)

    fig2.tight_layout()
    fig2.show()
    fig2.savefig(fname="../plots/evaluation/tuning_GMM2_pi0-1", dpi=180)

    fig3.tight_layout()
    fig3.show()
    fig3.savefig(fname="../plots/evaluation/tuning_GMM2_pi0-9", dpi=180)

    label_params1 = axs1[0, 0].get_legend_handles_labels()
    figl1, axl1 = plt.subplots(figsize=(6.5, 10))
    axl1.axis(False)
    axl1.legend(*label_params1, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl1.show()
    figl1.savefig(fname="../plots/evaluation/tuning_GMM_legend_pi0-5")

    label_params2 = axs2[0, 0].get_legend_handles_labels()
    figl2, axl2 = plt.subplots(figsize=(6.5, 10))
    axl2.axis(False)
    axl2.legend(*label_params2, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl2.show()
    figl2.savefig(fname="../plots/evaluation/tuning_GMM_legend_pi0-1")

    label_params3 = axs3[0, 0].get_legend_handles_labels()
    figl3, axl3 = plt.subplots(figsize=(6.5, 10))
    axl3.axis(False)
    axl3.legend(*label_params3, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size": 40})
    figl3.show()
    figl3.savefig(fname="../plots/evaluation/tuning_GMM_legend_pi0-9")


if __name__ == '__main__':
    # DO NOT COMMENT
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"],
        "axes.titlesize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 18,
        "legend.fontsize": 18,
        "figure.dpi": 180
    })

    # colors3 = distinctipy.get_colors(3, pastel_factor=1, colorblind_type='Deuteranomaly')
    # print(colors3)

    colors3 = [(0.5348547306212659, 0.5139339239248601, 0.5268686469375292),
               (0.9927895402526214, 0.7352147015471453, 0.5581049747178046),
               (0.524907193614528, 0.7671952975947081, 0.976588025007669)]

    colors6 = [(0.48702807223549177, 0.4242891647177821, 0.9480975665882982),
               (0.9146761531779931, 0.4970424422244128, 0.41460357267068376),
               (0.843602824944377, 0.6031154951690304, 0.9802318468625552),
               (0.5887251240359368, 0.9624135405893406, 0.4585532945832182),
               (0.422567523593921, 0.44218101996887993, 0.5516040738892886),
               (0.43399916426535, 0.7098723267606655, 0.6255076508970907)]
    colors8 = [(0.5450484248310105, 0.5130972742328073, 0.5102488831581509),
               (0.6109330873928905, 0.7193582681286009, 0.9814590256707204),
               (0.9727770320054765, 0.7854905796839438, 0.5145282365057959),
               (0.9806065670005477, 0.5066792697066322, 0.7311620666921056),
               (0.565920914907729, 0.9141080668353584, 0.7641066636691687),
               (0.5114677713143507, 0.5061193495393317, 0.9951605179132765),
               (0.5830073483609048, 0.5244350779880778, 0.7931264147573027),
               (0.5692188040526873, 0.7826586898074446, 0.5098679245540738)]
    # plot_lambda()
    # plot_tuningPolySVM()
    # plot_tuningRBFSVM()
    # print(os.path.abspath("."))

    # plot_tuningLinearSVMUnbalanced_evaluation()
    # plot_tuning_LinearSVMBalanced_evaluation()
    # plot_tuningPolySVM_evaluation()
    # plot_tuningRBFSVM_evaluation()
    # plot_lambda_evaluation()
    # plot_tuningGMM_evaluation()
    # plot_tuningGMM2()
    pass
