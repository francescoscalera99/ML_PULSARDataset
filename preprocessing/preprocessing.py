import numpy as np
from scipy.stats import norm


def PCA(dataset, training_data, m):
    """
    Performs dimensionality reduction to #features=m on the given dataset
    :param dataset: the input dataset
    :param m: the number of features of the resulting dataset
    :return: the reduced dataset
    """

    # computing covariance matrix
    mu = training_data.mean(1)
    DC = training_data - mu.reshape(mu.size, 1)
    C = np.dot(DC, DC.T)
    C = C / float(training_data.shape[1])

    # SVD decomposition
    U, _, _ = np.linalg.svd(C)
    P = U[:, 0:m]

    # compute the projection of points of dataset over P
    DP = np.dot(P.T, dataset)

    return DP


def gaussianize(training_data: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    """
    Performs the mapping from original data distribution to normal distribution, using the training set
    :param training_data: the training partition
    :param dataset: the data to gaussianize
    :return: the gaussianized data
    """
    ranks = []
    for j in range(dataset.shape[0]):
        tempSum = 0
        for i in range(training_data.shape[1]):
            tempSum += (dataset[j, :] < training_data[j, i]).astype(int)
        tempSum += 1
        ranks.append(tempSum / (training_data.shape[1] + 2))
    y = norm.ppf(ranks)
    return y