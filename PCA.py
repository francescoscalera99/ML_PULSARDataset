import numpy as np


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
