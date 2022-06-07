import abc
import numpy as np


class ClassifierClass(abc.ABC):
    class Model(abc.ABC):
        pass

    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray):
        self.training_data = training_data
        self.training_labels = training_labels

    @abc.abstractmethod
    def train_model(self) -> None:
        pass

    @abc.abstractmethod
    def classify(self, testing_data: np.ndarray, priors: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_llrs(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
