import numpy as np
from scipy import optimize as opt

from classifiers.Classifier import ClassifierClass
from utils.matrix_utils import vrow
from utils.metrics_utils import compute_min_DCF
from utils.utils import k_fold


class LR(ClassifierClass):
    class Model(ClassifierClass.Model):
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

    def get_llrs(self):
        return self._scores


def tuning_lambda(training_data, training_labels):
    m_values = [False, None, 7, 5]
    lbd_values = np.logspace(-5, 5, 50)
    priors = [0.5, 0.1, 0.9]

    for m in m_values:
        for pi in priors:
            DCFs = []
            for (i,lbd) in enumerate(lbd_values):
                if m == False:
                    llrs, evaluationLabels = k_fold(training_data, training_labels, LR, 5, m=None, raw=True, seed=0, lbd=lbd, pi_T=0.5)
                else:
                    llrs, evaluationLabels = k_fold(training_data, training_labels, LR, 5, m=m, raw=False, seed=0, lbd=lbd, pi_T=0.5)
                min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
                print(f"{i} data:PCA{m}, lbd={lbd}, pi={pi} -> min DCF", min_dcf)
                DCFs.append(min_dcf)
            np.save(f"LR_prior_{str(pi).replace('.', '-')}_PCA{m}", np.array(DCFs))


def calibrateScores(scores, evaluationLabels, lambd, prior, pi_T=0.5):
    # f(s) = as+b can be interpreted as the llr for the two class hypothesis
    # class posterior probability: as+b+log(pi/(1-pi)) = as +b'
    calibratedScore, calibratedEvaluationLabels = k_fold(vrow(scores), evaluationLabels, LR, 5, m=None, raw=True, seed=0, lbd=lambd, pi_T=pi_T)
    calibratedScore = calibratedScore - np.log(prior / (1 - prior))
    return calibratedScore, calibratedEvaluationLabels
