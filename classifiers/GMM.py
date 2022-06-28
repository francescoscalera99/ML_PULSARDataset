import itertools

import numpy as np
import scipy.special as special

from classifiers.Classifier import ClassifierClass
from utils.matrix_utils import vrow, vcol, empirical_dataset_mean, empirical_dataset_covariance
from utils.metrics_utils import compute_min_DCF
from utils.utils import k_fold


class GMM(ClassifierClass):
    class Model(ClassifierClass.Model):
        def __init__(self):
            self.gmms = []
            self.scores = []

        def add_gmm(self, gmm):
            self.gmms.append(gmm)

        def add_score(self, score):
            self.scores.append(score)

        def get_gmm(self, c):
            return self.gmms[c]

        def log_pdf(self, dataset, c):
            # logpdf_GMM
            """
            Evaluates the log-density of a GMM
            """
            num_components = self.gmms[c][0].size
            scores_list = []
            for g in range(num_components):
                scores_list.append(
                    vrow(GMM._logpdf_GAU_ND(dataset, self.gmms[c][1][g, :], self.gmms[c][2][g, :, :])))
            return np.vstack(scores_list)

    @staticmethod
    def _logpdf_GAU_ND(x, mu, C):
        """
        Evaluates the log-pdf for samples
        :param x: the array of samples
        :param mu: the average of :param x
        :param C: the covariance matrix of :param x
        :return: the log-density
        """
        if mu.shape[0] != C.shape[0]:
            raise RuntimeError("Mean and covariance should have one dimension in common.\n"
                               f"mu.shape = {mu.shape}\n"
                               f"C.shape = {C.shape}")

        # x has shape 4 x N
        n_samples = x.shape[1]

        y = [GMM._logpdf_GAU_ND_1(x[:, i], mu, C) for i in range(n_samples)]

        return vcol(np.array(y))

    @staticmethod
    def _logpdf_GAU_ND_1(x, mu, C):
        """
        Evaluates the log-pdf for a single sample
        :param x:
        :param mu:
        :param C:
        :return:
        """
        if mu.shape[0] != C.shape[0]:
            raise RuntimeError

        M = mu.shape[0]

        const = M * np.log(2 * np.pi)

        det_sigma = np.linalg.slogdet(C)[1]

        sigma_inv = np.linalg.inv(C)

        quadratic_term = vrow(vcol(x) - vcol(mu)) @ sigma_inv @ vcol(vcol(x) - vcol(mu))

        r = (const + det_sigma + quadratic_term)
        return -0.5 * r[0]

    def __init__(self, training_data, training_labels, **kwargs):
        super().__init__(training_data, training_labels)
        self._scores = None

        # Possible values of kwargs['type'] are from ['full-cov', 'diag', 'tied']
        if kwargs['type'] not in ['full-cov', 'diag', 'tied']:
            raise RuntimeError("Error: type can only be 'full-cov', 'diag' or 'tied'")
        self._type = kwargs['type']

        self._model = None

    def _em_estimation(self, dataset, gmm0, psi):
        num_components = gmm0[0].size
        num_samples = dataset.shape[1]

        def expectation(weights: np.ndarray, means: np.ndarray, covariances: np.ndarray):
            component_likelihoods = [vrow(GMM._logpdf_GAU_ND(dataset, means[i], covariances[i, :, :])) for i in range(num_components)]
            log_score_matrix = np.vstack(component_likelihoods)
            joint_log_densities = log_score_matrix + vcol(np.log(weights))
            marginal_log_densities = vrow(special.logsumexp(joint_log_densities, axis=0))
            log_posterior_densities = joint_log_densities - marginal_log_densities
            posterior_probs = np.exp(log_posterior_densities)
            return posterior_probs, marginal_log_densities.sum() / num_samples

        def maximization(gammas: np.ndarray):
            # *************** values _{g, t} ***************
            # zero: (G,)
            zero_order = gammas.sum(axis=1)
            # first: (G, D)
            first_order = gammas @ dataset.T
            # second: (G, D, D)
            second_order = dataset @ (gammas.reshape(*gammas.shape, 1) * dataset.T)

            # *************** values _{g, t+1} ***************
            # mus: (G, D)
            mus = (first_order.T / zero_order).T
            # sigmas: (G, D, D)
            sigmas = ((second_order.T / zero_order).T
                      - (mus.reshape(*mus.shape, 1) @ mus.reshape(mus.shape[0], 1, mus.shape[1])))

            if self._type == 'diag':
                # remove entries that for some reason are -0.0
                sigmas = np.abs(sigmas)
                sigmas = sigmas * np.eye(sigmas.shape[-1])
            elif self._type == 'tied':
                zo2 = vcol(zero_order)
                tied = (zo2.reshape(*zo2.shape, 1) * sigmas).sum(axis=0) / num_samples
                sigmas = np.stack([tied] * num_components)

            # EIGENVALUE CONSTRAINING - WITH BROADCASTING
            u, s, v = np.linalg.svd(sigmas)
            s[s < psi] = psi
            sigmas = u @ (s.reshape(*s.shape, 1) * v)
            _, s, _ = np.linalg.svd(sigmas)
            # ws: (G,)
            ws = vcol(zero_order / zero_order.sum())
            return ws, mus, sigmas

        avg_ll = None
        avg_ll_new = None

        gmm = gmm0
        n_iter = 0
        while avg_ll is None or (avg_ll_new - avg_ll) > 1e-6:
            n_iter += 1
            avg_ll = avg_ll_new
            posteriors, avg_ll_new = expectation(*gmm)
            gmm = maximization(posteriors)
        return gmm

    def _lbg(self, dataset, desired_n_components, alpha=0.1, psi=0.01):
        def split_gmm(gmm_to_split):
            ds = []
            for g in range(gmm_to_split[0].size):
                u, s, _ = np.linalg.svd(gmm_to_split[2][g, :, :])
                d = u[:, 0:1] * s[0] ** 0.5 * alpha
                ds.append(vrow(d))
            ds = np.vstack(ds)
            ws = gmm_to_split[0] / 2
            mu1 = gmm_to_split[1] - ds
            mu2 = gmm_to_split[1] + ds
            sigmas = gmm_to_split[2]

            new_gmm = (np.vstack((ws, ws)), np.vstack((mu1, mu2)), np.vstack((sigmas, sigmas)))

            return new_gmm

        mu0 = vrow(empirical_dataset_mean(dataset))
        c0 = empirical_dataset_covariance(dataset)
        c0 = c0.reshape(1, *c0.shape)
        num_components = 1
        gmm = (np.array([1.0]), mu0, c0)
        gmm = self._em_estimation(dataset, gmm, psi)
        while num_components < desired_n_components:
            gmm = split_gmm(gmm)
            gmm = self._em_estimation(dataset, gmm, psi)
            num_components = gmm[0].size

        return gmm

    # def _mix_pdf(self, x):
    #     d = np.zeros_like(x)
    #     num_components = self._gmm[0].size
    #     weights, means, covs = self._gmm
    #     weights = vcol(weights)
    #     for g in range(num_components):
    #         dd = weights[g] * norm.pdf(x, loc=float(means[g, :]), scale=float(covs[g, :, :]))
    #         d += vrow(dd)
    #     return d

    def train_model(self, **kwargs) -> None:
        alpha = kwargs['alpha']
        psi = kwargs['psi']
        desired_n_components = kwargs['G']
        self._model = GMM.Model()
        num_classes = len(set(self.training_labels))
        for c in range(num_classes):
            dataset = self.training_data[:, self.training_labels == c]
            gmm = self._lbg(dataset, desired_n_components, alpha=alpha, psi=psi)
            weights = vcol(gmm[0])
            self._model.add_gmm(gmm)
            score_matrix = self._model.log_pdf(dataset, c) + np.log(weights)
            self._model.add_score(special.logsumexp(score_matrix, axis=0))

    def classify(self, testing_data: np.ndarray, priors: np.ndarray) -> np.ndarray:
        num_samples = testing_data.shape[1]

        def log_likelihood(gmm):
            num_components = gmm[0].size
            ll = np.empty((num_components, num_samples))
            for g in range(num_components):
                ll[g, :] = vrow(np.log(gmm[0][g])) + vrow(GMM._logpdf_GAU_ND(testing_data, gmm[1][g, :], gmm[2][g, :, :]))
            return special.logsumexp(ll, axis=0)
        # BINARY: Compute llr to score testing samples

        ll1 = log_likelihood(self._model.get_gmm(1))
        ll0 = log_likelihood(self._model.get_gmm(0))

        self._scores = ll1 - ll0
        if priors is not None:
            thresh = np.log(priors[1] / priors[0])
            predictions = np.array(self._scores > thresh).astype(int)
            return predictions

    def get_llrs(self):
        return self._scores


def tuning_componentsGMM(training_data, training_labels, alpha=0.1, psi=0.01):
    variants = ['full-cov', 'diag', 'tied']
    raw = [True, False]
    m_values = [None, 7]
    components_values = [1, 2, 4, 8, 16, 32]

    # len(hyperparameters): 12
    # FOR EACH TUPLE IN hyperparameters WE PERFORM 4 INNER ITERATIONS
    hyperparameters = list(itertools.product(variants, raw, m_values))
    # SPLIT HYPERPARAMETERS IN 6 PARTITIONS OF 2 TUPLES -> 8 INNER ITERATIONS OVERALL

    # ELENA: hyperparameters[:2]
    # CICCIO: hyperparameters[2:4]
    # TODO: hyperparameters[4:6]
    # TODO: hyperparameters[6:8]
    # TODO: hyperparameters[8:10]
    # TODO: hyperparameters[10:]

    curr_hyp = hyperparameters[:1]

    i = 0
    for variant, r, m in curr_hyp:
        DCFs = []
        for g in components_values:
            print(f"Inner iteration {i + 1}/{len(curr_hyp) * len(components_values)}")
            llrs, evalutationLables = k_fold(training_data, training_labels, GMM, 5, seed=0, raw=r, m=m, type=variant, alpha=alpha, psi=psi, G=g)
            min_dcf = compute_min_DCF(llrs, evalutationLables, 0.5, 1, 1)
            DCFs.append(min_dcf)
        i += 1
        np.save(f"GMM_rawFeature-{r}_PCA{m}_{variant}", DCFs)
