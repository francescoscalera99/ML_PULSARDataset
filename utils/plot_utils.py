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
