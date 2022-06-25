import numpy as np

from preprocessing.preprocessing import PCA, gaussianize
from classifiers.Classifier import ClassifierClass


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
    with open(f'{path}data/Test.txt') as f:
        for line in f:
            fields = line.split(',')
            label = fields[-1]
            features = fields[:-1]
            dataset_test.append(features)
            labels_test.append(label)

    return (np.array(dataset_train, dtype=float).T, np.array(labels_train, dtype=int)), \
           (np.array(dataset_test, dtype=float).T, np.array(labels_test, dtype=int))


def compute_accuracy(predictedLabels: np.ndarray, L: np.ndarray):
    """
    Loads the dataset from the specified path
    :param predictedLabels: the labels produced by the classifier
    :param L: labels of testing set
    :return: one tuples: (accuracy, error_rate)
    """

    nCorrect_labels = np.array(predictedLabels == L).sum()
    nSamples = predictedLabels.shape[0]

    acc = nCorrect_labels / nSamples
    return acc, 1 - acc


def evaluate_classification_errors(testing_labels: np.ndarray, predicted_labels) -> tuple[int, int]:
    # the two arrays are row vectors
    if testing_labels.size != predicted_labels.size:
        raise RuntimeError("Testing labels and predicted labels should be the same")
    num_samples = testing_labels.size
    comparison = testing_labels == predicted_labels
    num_correct = np.sum(comparison)
    return num_samples - num_correct, num_samples


def k_fold(dataset: np.ndarray,
           labels: np.ndarray,
           classifier: ClassifierClass.__class__,
           nFold: int,
           seed: int = None,
           **kwargs) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the k-fold cross-validation on the given dataset

    :param dataset: the input dataset
    :param labels: the input labels
    :param classifier: the classifier function
    :param nFold: the number of partitions
    :param seed: the seed for the random permutation (for debug purposes)
    :return: the error rate
    """
    if not nFold:
        raise RuntimeError('Value of k must be set')

    num_samples = dataset.shape[1]
    partition_size = num_samples // nFold

    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    partitions = np.empty(nFold, np.ndarray)
    partitions_labels = np.empty(nFold, np.ndarray)

    q = 1
    for p in range(nFold):
        if p == nFold - 1:
            partitions[p] = dataset[:, indices[p * partition_size:]]
            partitions_labels[p] = labels[indices[p * partition_size:]]
            break
        partitions[p] = dataset[:, indices[p * partition_size: q * partition_size]]
        partitions_labels[p] = labels[indices[p * partition_size: q * partition_size]]
        q += 1

    bool_indices = np.array([True] * nFold)
    llrs = []
    labels = []
    for i in range(nFold):
        print(f"Fold {i+1}/{nFold}")
        bool_indices[i] = False
        training_data = np.hstack(partitions[bool_indices])
        training_labels = np.hstack(partitions_labels[bool_indices])
        testing_data = partitions[i]
        testing_labels = np.hstack(partitions_labels[i])
        m = kwargs['m']
        raw = kwargs['raw']
        if not raw:
            dtr = gaussianize(training_data, training_data)
            dte = gaussianize(training_data, testing_data)
        else:
            dtr = training_data
            dte = testing_data

        if m is not None:
            dtrain = PCA(dtr, dtr, m)
            dtev = PCA(dte, dtr, m)
        else:
            dtrain = dtr
            dtev = dte
        c = classifier(dtrain, training_labels, **kwargs)
        c.train_model(**kwargs)
        c.classify(dtev, None)
        llrs.extend(c.get_llrs().tolist())
        labels.extend(testing_labels.tolist())
        bool_indices[i] = True

    return np.array(llrs), np.array(labels)


def splitData_SingleFold(dataset_train, labels_train, seed=0):
    nTrain = int(dataset_train.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(dataset_train.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = dataset_train[:, idxTrain]
    DTEV = dataset_train[:, idxTest]
    LTR = labels_train[idxTrain]
    LTEV = labels_train[idxTest]
    return (DTR, LTR), (DTEV, LTEV)


#FIXME
def calibrateScores(scores, evaluationLabels, lambd, prior=0.5):
    # f(s) = as+b can be interpreted as the llr for the two class hypothesis
    # class posterior probability: as+b+log(pi/(1-pi)) = as +b'
    # logReg = LR(scores, evaluationLabels, lbd=lambd, pi_T=prior)
    # logReg.train_model()
    # alpha = x[0]
    # betafirst = x[1]
    # calibratedScores = alpha * scores + betafirst - np.log(prior/(1 - prior))
    #
    # return calibratedScores
    return


def main():
    # (dtr, ltr), (dte, lte) = load_dataset()
    # print(dtr[:, 0])
    # titles = ['1. Mean of the integrated profile',
    #           '2. Standard deviation of the integrated profile',
    #           '3. Excess kurtosis of the integrated profile',
    #           '4. Excess kurtosis of the integrated profile',
    #           '5. Mean of the DM-SNR curve',
    #           '6. Standard deviation of the DM-SNR curve',
    #           '7. Excess kurtosis of the DM-SNR curve',
    #           '8. Skewness of the DM-SNR curve']
    # plot_histogram(dtr, ltr, titles)
    #
    # z_dtr = z_normalization(dtr)
    # plot_histogram(z_dtr, ltr, titles)
    #
    # gauss = gaussianize(z_dtr, z_dtr)
    # plot_histogram(gauss, ltr, titles, nbins=20)
    # create_heatmap(gauss, title="Whole dataset")
    # create_heatmap(gauss[:, ltr == 1], cmap="Blues", title="True class")
    # create_heatmap(gauss[:, ltr == 0], cmap="Greens", title="False class")
    pass


if __name__ == '__main__':
    main()
