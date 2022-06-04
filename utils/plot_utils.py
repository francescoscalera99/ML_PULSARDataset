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
    for j in range(array.shape[0]):
        # for j in range(1):
        f = plt.gcf()
        for i in range(len(set(labels))):
            plt.hist(array[j, labels == i], bins=nbins, density=True, alpha=0.7)

        plt.title(titles[j])
        f.show()
        # f.savefig(fname=f'outputs/figure{j}')


def create_heatmap(dataset, cmap='Reds', title=None):
    # TODO
    """

    :param dataset:
    :param cmap:
    :param title:
    :return:
    """
    heatmap = np.corrcoef(dataset)
    plt.title(title)
    sns.heatmap(heatmap, cmap=cmap, annot=True)
    plt.show()