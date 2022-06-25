import itertools
import distinctipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_lambda():
    lbd_values = np.logspace(-5, 5, 50)
    m_values = [False, None, 7, 5]
    prior = [0.5, 0.1, 0.9]

    i = 0
    fig, axs = plt.subplots(2, 2)
    # fig.suptitle('Tuning hyperparameter λ')
    plt.rcParams['text.usetex'] = True

    colors = ['red', 'blue', 'green']
    for m in m_values:
        for j, pi in enumerate(prior):
            DCFs = np.load(
                f"../simulations/LR/LR_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")
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
        i += 1
    # fig.set_size_inches(10, 10)
    # fig.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:3], labels[:3], loc=10, prop={'size': 10})
    fig.subplots_adjust(wspace=0.3, hspace=0.6)
    fig.subplots_adjust(top=0.88)
    fig.show()


def plot_tuningPolySVM():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7, 5]
    K_values = [0.0, 1.0]
    c_values = [0, 1, 10, 15]

    num_colors = len(K_values) * len(c_values)
    colors = distinctipy.get_colors(num_colors, pastel_factor=0.7)

    i = 0
    fig, axs = plt.subplots(1, 4)
    # fig.suptitle('Poly SVM')
    plt.rcParams['text.usetex'] = True
    for m in m_values:
        hyperparameters = itertools.product(K_values, c_values)
        for j, (K, c) in enumerate(hyperparameters):
            DCFs = np.load(
                f"../simulations/polySVM/K{str(K).replace('.', '-')}_c{str(c).replace('.', '-')}_PCA{str(m)}.npy")
            axs[i].plot(C_values, DCFs, color=colors[j], label=rf"$K={K}$, $c={c}$")
            if (m == False):
                axs[i].set_title(f'5-fold, Raw features')
            else:
                axs[i].set_title(f'5-fold, PCA (m = {m})')
            axs[i].legend()
            axs[i].set_xlabel(r'$C$')
            axs[i].set_ylabel(r'$minDCF$')
            axs[i].set_xscale('log')
        i += 1
    fig.set_size_inches(20, 5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.show()


def plot_tuningRBFSVM():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7]
    K_values = [0.0, 1.0]
    gamma_values = [1e-2, 1e-3, 1e-4]
    i = 0
    fig, axs = plt.subplots(1, 3)
    # fig.suptitle('RBF SVM')
    plt.rcParams['text.usetex'] = True

    num_colors = len(K_values) * len(gamma_values)
    colors = distinctipy.get_colors(num_colors, pastel_factor=0.7)

    for m in m_values:
        hyperparameters = itertools.product(K_values, gamma_values)
        for j, (K, gamma) in enumerate(hyperparameters):
            DCFs = np.load(
                f"../simulations/RBF/RBF_K{str(K).replace('.', '-')}_c{str(gamma).replace('.', '-')}_PCA{str(m)}.npy")
            axs[i].plot(C_values, DCFs, color=colors[j], label=rf"$K={K}$, $\gamma={gamma}$")
            if (m == False):
                axs[i].set_title(f'5-fold, Raw features')
            else:
                axs[i].set_title(f'5-fold, PCA (m = {m})')
            axs[i].legend()
            axs[i].set_xlabel(r'$C$')
            axs[i].set_ylabel(r'$minDCF$')
            axs[i].set_xscale('log')
        i += 1

    fig.set_size_inches(15, 5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.show()


def plot_tuningLinearSVMUnbalanced():
    C_values = np.logspace(-3, 3, 20)
    m_values = [False, None, 7, 5]
    K_values = [1.0, 10.0]
    priors = [0.5, 0.1, 0.9]

    i = 0
    fig, axs = plt.subplots(1, 4, sharey='row')
    colors = distinctipy.get_colors(6, pastel_factor=0.7)
    for m in m_values:
        hyperparameters = itertools.product(K_values, priors)
        for j, (K, p) in enumerate(hyperparameters):
            DCFs = np.load(
                f"../simulations/linearSVM/unbalanced/new/K{str(K).replace('.', '-')}_p{str(p).replace('.', '-')}_PCA{str(m)}.npy")
            axs[i].plot(C_values, DCFs, color=colors[j], label=f"K={K}, prior={p}")
            if (m == False):
                axs[i].set_title(f'5-fold, Raw features')
            else:
                axs[i].set_title(f'5-fold, PCA (m = {m})')
            axs[i].legend()
            axs[i].set_xlabel('C')
            axs[i].set_ylabel('minDCF')
            axs[i].set_xscale('log')
            axs[i].yaxis.set_tick_params(labelbottom=True)
        i += 1
    fig.set_size_inches(20, 5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.show()

if __name__ == '__main__':
    # plot_lambda()
    # plot_tuningPolySVM()
    # plot_tuningRBFSVM()
    # plot_tuningLinearSVMUnbalanced()
    pass
