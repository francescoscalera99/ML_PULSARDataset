import numpy as np
from scipy import optimize as opt

from classifiers.Classifier import ClassifierClass


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

    def train_model(self) -> None:
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


# def find_optLambda(training_data, training_labels):
#     titles_Kfold = ['Gaussianized feature (5-fold, no PCA)', 'Guassianized feature (5-fold, PCA = 7)', 'Gaussianized feature (5-fold, PCA = 6)']
#     titles_SingleFold = ['Gaussianized feature (Single fold, no PCA)', 'Guassianized feature (Single fold, PCA = 7)',
#                          'Gaussianized feature (Single fold, PCA = 6)']
#     datasets = []
#
#     training_dataPCA7 = PCA(training_data, 7)
#     training_dataPCA6 = PCA(training_data, 6)
#     datasets.append(training_data, training_dataPCA7, training_dataPCA6)
#
#     lbd = np.logspace(-5, 5, 50)
#     priors = [0.5, 0.1, 0.9]
#     colors = ['Red', 'Green', 'Blue']
#     labels = ['min DCF (π=0.5)', 'min DCF (π=0.1)', 'min DCF (π=0.9)']
#     j = 0
#     print("===========TUNING OF LAMBDA -> K-FOLD  (K = 5, tries with NO PCA, PCA = 7, PCA = 6) ====== ")
#     for dataset in datasets:
#         allKFolds, evaluationLabels = kFold(dataset, training_labels)
#         i = 0
#         plt.figure()
#         for prior in priors:
#             DCFs = []
#             for lb in lbd:
#                 llrs = []
#                 for singleKFold in allKFolds:
#                     dtr_gaussianized = gaussianize(singleKFold[1], singleKFold[1])
#                     dte_gaussianized = gaussianize(singleKFold[1], singleKFold[2])
#                     lr = LR(dtr_gaussianized, singleKFold[0], lb, 0.5)
#                     lr.train_model()
#                     lr.classify(dte_gaussianized, np.array([0.5, 0.5]))
#                     llr = lr.get_llrs()
#                     llr = llr.tolist()
#                     llrs.extend(llr)
#                 min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, prior, 1, 1)
#                 print("dataset:", labels[j], "lambda: ", lb, "prior: ", prior, ":", min_dcf)
#                 DCFs.append(min_dcf)
#             plt.plot(lbd, DCFs, color=colors[i], label=labels[i])
#             i += 1
#         plt.title(titles_Kfold[j])
#         j += 1
#         plt.legend()
#         plt.xscale('log')
#         plt.show()
#
#     print("===============FIND BEST LAMBDA - SINGLE FOLD(tries with NO PCA, PCA = 7, PCA = 6)==============")
#     j = 0
#     for dataset in datasets:
#         i = 0
#         plt.figure()
#         (dtr, ltr), (dte, lte) = splitData_SingleFold(dataset, training_labels, seed=0)
#         dtr_gaussianized = gaussianize(dtr, dtr)
#         dte_gaussianized = gaussianize(dtr, dte)
#         lbd = np.logspace(-5, +5, 50)
#         DCFs = []
#         for lb in lbd:
#             lr = LR(dtr_gaussianized, ltr, lb, 0.5)
#             lr.train_model()
#             lr.classify(dte_gaussianized, np.array([0.5, 0.5]))
#             llr = lr.get_llrs()
#             min_dcf = compute_min_DCF(llr, lte, 0.5, 1, 1)
#             DCFs.append(min_dcf)
#         plt.plot(lbd, DCFs, color="Blue", label=labels[i])
#         i += 1
#         plt.title(titles_SingleFold[j])
#         j += 1
#         plt.xscale('log')
#         plt.show()
