import numpy as np
from scipy import optimize as opt

from classifiers.Classifier import ClassifierClass
from utils.matrix_utils import vrow
# from utils.metrics_utils import compute_min_DCF
# from utils.utils import k_fold
from utils.misc_utils import splitData_SingleFold


class LR(ClassifierClass):
    """
    A Logistic Regression classifier
    """
    class Model(ClassifierClass.Model):
        """
        The parameters found in the training step
        """
        def __init__(self, w: np.ndarray, b: float):
            self.w = w
            self.b = b

    def __init__(self, training_data, training_labels, **kwargs):
        super().__init__(training_data, training_labels)
        self._lambda = kwargs['lbd']
        self._model = None
        self._pi_t = kwargs['pi_T']
        self._scores = None

    def objective_function(self, v):
        """
        The objective function of the model
        :param v: the array returned by the L-BFGS algorithm
        :return: the value of the objective function
        """
        w, b = v[0:-1], v[-1]
        z = 2 * self.training_labels - 1
        x = self.training_data
        regularization_term = (self._lambda / 2) * (w.T @ w)
        nf = len(self.training_labels == 0)
        nt = len(self.training_labels == 1)
        second_term_t = self._pi_t/nt * np.logaddexp(0, -z[self.training_labels == 1] * (w.T @ x[:, self.training_labels == 1] + b)).mean()
        second_term_f = (1-self._pi_t)/nf * np.logaddexp(0, -z[self.training_labels == 0] * (w.T @ x[:, self.training_labels == 0] + b)).mean()
        return regularization_term + second_term_t + second_term_f

    def train_model(self, **kwargs) -> None:
        x0 = np.zeros(self.training_data.shape[0] + 1)
        v, _, _ = opt.fmin_l_bfgs_b(self.objective_function, x0=x0, approx_grad=True)

        w, b = v[0:-1], v[-1]
        self._model = LR.Model(w, b)

    def classify(self, testing_data: np.ndarray, priors: np.ndarray) -> np.ndarray:
        self._scores = np.dot(self._model.w.T, testing_data) + self._model.b
        predicted_labels = (self._scores > 0).astype(int)
        return predicted_labels

    def get_scores(self):
        return self._scores


def calibrateScores(scores, evaluationLabels, lambd, prior):
    """
    The function used to calibrate the scores through prior-weighted LR
    :param scores: the "raw" scores produced by some classifier
    :param evaluationLabels: the corresponding labels
    :param lambd: the lambda value for the Logistic Regression
    :param prior: the prior used to train the prior-weighted LR
    :return: the calibrated scores and the corresponding labels
    """
    # f(s) = as+b can be interpreted as the llr for the two class hypothesis
    # class posterior probability: as+b+log(pi/(1-pi)) = as +b'

    (cal_set, cal_labels), (val_set, val_labels) = splitData_SingleFold(vrow(scores), evaluationLabels, seed=0)
    lr = LR(cal_set, cal_labels, lbd=lambd, pi_T=prior)
    lr.train_model()
    lr.classify(val_set, None)

    calibrated_scores = lr.get_scores()

    return calibrated_scores, val_labels
