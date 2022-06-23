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
                f"/Users/riccardo/PycharmProjects/ML_PULSARDataset/simulations/LR/LR_prior_{str(pi).replace('.', '-')}_PCA{str(m)}.npy")
            axs[i//2, i%2].plot(lbd_values, DCFs, color=colors[j], label=r"$\widetilde{\pi}=$"+f"{pi}")

            if m == False:
                axs[i//2, i%2].set_title(f'5-fold, Raw features')
            elif m is None:
                axs[i//2, i%2].set_title(f'5-fold, no PCA')
            else:
                axs[i//2, i%2].set_title(f'5-fold, PCA (m={m})')

            axs[i//2, i%2].set_xlabel('λ')
            axs[i//2, i%2].set_ylabel('minDCF')
            axs[i//2, i%2].set_xscale('log')
        i += 1
    # fig.set_size_inches(10, 10)
    # fig.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:3], labels[:3], loc=10, prop={'size': 10})
    fig.subplots_adjust(wspace=0.3, hspace=0.6)
    fig.subplots_adjust(top=0.88)
    fig.show()


if __name__ == '__main__':
    plot_lambda()
