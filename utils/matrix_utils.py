import numpy as np


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
    return (dataset @ dataset.T) / n


def covariance_matrix_mean(D):
    mu = vcol(D.mean(1))
    DC = D - mu
    C = np.dot(DC, DC.T)
    C = C / float(D.shape[1])
    return C, mu
