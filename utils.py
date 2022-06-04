import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns


def vcol(x) -> np.ndarray:
    """
    Reshapes the given array into a column vector
    :param x: the input array
    :return: the column vector
    """
    try:
        x = np.array(x)
    except:
        raise RuntimeError(f"Error: {x} is not an iterable")
    return x.reshape((x.size, 1))


def vrow(x) -> np.ndarray:
    """
    Reshapes the given array into a row vector
    :param x: the input array
    :return: the row vector
    """
    try:
        x = np.array(x)
    except:
        raise RuntimeError(f"Error: {x} is not an iterable")
    return x.reshape((1, x.size))


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
    heatmap = np.corrcoef(dataset)
    plt.title(title)
    sns.heatmap(heatmap, cmap=cmap, annot=True)
    plt.show()


def empirical_dataset_mean(dataset: np.ndarray) -> np.ndarray:
    """
    Computes the empirical mean of the given dataset
    :param dataset: the input dataset
    :return: the vector of means
    """
    return vcol(np.average(dataset, axis=1))


def empirical_dataset_covariance(dataset: np.ndarray) -> np.ndarray:
    """
    Computes the empirical covariance of the given dataset
    :param dataset: the input dataset
    :return: the covariance matrix
    """
    dataset = dataset - empirical_dataset_mean(dataset)
    n = dataset.shape[1]
    return (dataset @ dataset.T)/n


def z_normalization(dataset: np.ndarray) -> np.ndarray:
    """
    Computes the Z-normalization
    :param dataset:
    :return:
    """
    mean = dataset.mean(axis=1)
    std = dataset.std(axis=1)
    return (dataset - vcol(mean)) / vcol(std)


def gaussianize(training_data: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    """
    Performs the mapping from original data distribution to normal distribution, using the training set
    :param training_data: the training partition
    :param dataset: the data to gaussianize
    :return: the gaussianized data
    """
    ranks = []
    for feature in range(dataset.shape[0]):
        counts = np.zeros(dataset.shape[1])
        for sample in range(dataset.shape[1]):
            count = np.int64(training_data[feature, :] < dataset[feature, sample]).sum()
            counts[sample] = (count + 1) / (dataset.shape[1] + 2)
        ranks.append(counts)

    ranks = np.vstack(ranks)

    data = []
    for feature in range(dataset.shape[0]):
        y = norm.ppf(ranks[feature])
        data.append(y)

    data = np.vstack(data)

    return data


def evaluate_classification_errors(testing_labels: np.ndarray, predicted_labels) -> tuple[int, int]:
    # the two arrays are row vectors
    if testing_labels.size != predicted_labels.size:
        raise RuntimeError("Testing labels and predicted labels should be the same")
    num_samples = testing_labels.size
    comparison = testing_labels == predicted_labels
    num_correct = np.sum(comparison)
    return num_samples - num_correct, num_samples


def k_fold(dataset: np.ndarray, labels: np.ndarray, classifier, k: int, seed: int = None) -> float:
    """
    Perform a k-fold cross-validation on the given dataset

    :param dataset: the input dataset
    :param labels: the input labels
    :param classifier: the classifier function
    :param k: the number of partitions
    :param seed: the seed for the random permutation (for debug purposes)
    :return: the error rate
    """
    if not k:
        raise RuntimeError('Value of k must be set')

    num_samples = dataset.shape[1]
    partition_size = num_samples // k

    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    partitions = np.empty(k, np.ndarray)
    partitions_labels = np.empty(k, np.ndarray)

    q = 1
    for p in range(k):
        if p == k - 1:
            partitions[p] = dataset[:, indices[p * partition_size:]]
            partitions_labels[p] = labels[indices[p * partition_size:]]
            break
        partitions[p] = dataset[:, indices[p * partition_size: q * partition_size]]
        partitions_labels[p] = labels[indices[p * partition_size: q * partition_size]]
        q += 1

    bool_indices = np.array([True] * k)
    n_errors = 0
    n_classifications = 0
    for i in range(k):
        bool_indices[i] = False
        training_data = np.hstack(partitions[bool_indices])
        training_labels = np.hstack(partitions_labels[bool_indices])
        testing_data = partitions[i]
        testing_labels = np.hstack(partitions_labels[i])

        # perform the classification
        predictions, _ = classifier(training_data, training_labels, testing_data, testing_labels)
        # evaluate number of errors
        err, samples = evaluate_classification_errors(testing_labels, predictions)
        n_errors += err
        n_classifications += samples
        bool_indices[i] = True

    return float(n_errors / n_classifications)



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
    # plot_histogram(dtr, ltr, titles)

    z_dtr = z_normalization(dtr)
    # plot_histogram(z_dtr, ltr, titles)

    gauss = gaussianize(z_dtr, z_dtr)
    # plot_histogram(gauss, ltr, titles, nbins=20)
    create_heatmap(gauss, title="Whole dataset")
    create_heatmap(gauss[:, ltr == 1], cmap="Blues", title="True class")
    create_heatmap(gauss[:, ltr == 0], cmap="Greens", title="False class")


if __name__ == '__main__':
    main()
