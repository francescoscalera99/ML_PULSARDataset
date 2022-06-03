import numpy as np
import matplotlib.pyplot as plt


def vcol(X):
    """
    Reshapes the given array into a column vector
    :param X: the input array
    :return: the column vector
    """
    try:
        X = np.array(X)
    except:
        raise RuntimeError(f"Error: {X} is not an iterable")
    return X.reshape((X.size, 1))


def vrow(X):
    """
    Reshapes the given array into a row vector
    :param X: the input array
    :return: the row vector
    """
    try:
        X = np.array(X)
    except:
        raise RuntimeError(f"Error: {X} is not an iterable")
    return X.reshape((1, X.size))


def load_dataset(path: str = './') -> tuple:
    """
    Loads the dataset from the specified path
    :param path: the specified path. If not specified, default value is current working folder
    :return: two tuples: (train_partition, train_labels), (test_partition, test_labels)
    """
    dataset_train = []
    labels_train = []
    with open(f'{path}data/Train.txt') as f:
        for line in f:
            fields = line.split(',')
            label = fields[-1]
            features = fields[:-1]
            dataset_train.append(features)
            labels_train.append(label)

    dataset_test = []
    labels_test = []
    with open('data/Test.txt') as f:
        for line in f:
            fields = line.split(',')
            label = fields[-1]
            features = fields[:-1]
            dataset_test.append(features)
            labels_test.append(label)

    return (np.array(dataset_train, dtype=float).T, np.array(labels_train, dtype=int)), \
           (np.array(dataset_test, dtype=float).T, np.array(labels_test, dtype=int))


def compute_accuracy_MVG(SPost, L):
    """
        Loads the dataset from the specified path
        :param SPost: matrix of post conditional class probabilities
        :param L: labels of training set
        :return: two tuples: (train_partition, train_labels), (test_partition, test_labels)
    """
    predictedLabels = np.argmax(SPost, axis=0)

    nCorrect_labels = (predictedLabels == L).sum()
    nSamples = predictedLabels.shape[0]

    acc = nCorrect_labels / nSamples
    return (acc, 1 - acc)


def plot_histogram(array, labels, titles, nbins: int = 10):
    """
    Plots the histogram of each feature of the given array
    :param array: the (training) dataset
    :param labels: the labels of parameter array
    :param titles: the array for the titles of the histograms (the names of each feature)
    :param nbins: the number of bins for the histograms
    """
    for j in range(array.shape[0]):
        f = plt.gcf()
        for i in range(len(set(labels))):
            plt.hist(array[j, labels == i], bins=nbins, density=True)

        plt.title(titles[j])
        f.show()
        f.savefig(fname=f'outputs/figure{j}')


def main():
    (dtr, ltr), (dte, lte) = load_dataset()
    # print(dtr[:, 0])
    titles = ['1. Mean of the integrated profile',
              '2. Standard deviation of the integrated profile',
              '3. Excess kurtosis of the integrated profile',
              '4. Excess kurtosis of the integrated profile',
              '5. Mean of the DM-SNR curve',
              '6. Standard deviation of the DM-SNR curve',
              '7. Excess kurtosis of the DM-SNR curve',
              '8. Skewness of the DM-SNR curve']
    plot_histogram(dtr, ltr, titles)


if __name__ == '__main__':
    main()
