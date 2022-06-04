import numpy as np


def PCA(dataset, m):
    """
    Performs dimensionality reduction to #features=m on the given dataset
    :param dataset: the input dataset
    :param m: the number of features of the resulting dataset
    :return: the reduced dataset
    """

    # computing covariance matrix
    mu = dataset.mean(1)
    DC = dataset - mu.reshape(mu.size, 1)
    C = np.dot(DC, DC.T)
    C = C / float(dataset.shape[1])

    # SVD decomposition
    U, _, _ = np.linalg.svd(C)
    P = U[:, 0:m]

    # compute the projection of points of dataset over P
    DP = np.dot(P.T, dataset)

    return DP
