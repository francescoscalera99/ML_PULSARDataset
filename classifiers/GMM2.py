import itertools

import numpy as np
import scipy.special as special

from classifiers.Classifier import ClassifierClass
from utils.matrix_utils import vrow, vcol, empirical_dataset_mean, empirical_dataset_covariance
from utils.utils import k_fold

# import warnings
# warnings.filterwarnings('error')


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
            num_components = len(self.gmms[c])
            scores_list = []
            for g in range(num_components):
                scores_list.append(
                    vrow(GMM._logpdf_GAU_ND(dataset, self.gmms[c][g][1], self.gmms[c][g][2])))
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

    @staticmethod
    def _constrain_covariances_eigs(sigma, psi):
        u, s, v = np.linalg.svd(sigma)
        s[s < psi] = psi
        ret = u @ (s.reshape(s.size, 1) * v)
        return ret

    def __init__(self, training_data, training_labels, **kwargs):
        super().__init__(training_data, training_labels)
        self._scores = None

        # Possible values of kwargs['type'] are from ['full-cov', 'diag', 'tied']
        if kwargs['type'] not in ['full-cov', 'diag', 'tied']:
            raise RuntimeError("Error: type can only be 'full-cov', 'diag' or 'tied'")
        self._type = kwargs['type']

        self._model = None

    def _em_estimation(self, dataset, gmm0, psi):
        num_components = len(gmm0)
        num_samples = dataset.shape[1]

        def expectation(gmm_arg):
            component_likelihoods = [vrow(GMM._logpdf_GAU_ND(dataset, gmm_arg[g][1], gmm_arg[g][2])) for g in range(num_components)]
            log_score_matrix = np.vstack(component_likelihoods)
            weights = [gmm_arg[g][0] for g in range(num_components)]
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
            new_gmm = []
            for g in range(num_components):
                mu = vcol(first_order[g, :] / zero_order[g])

                sigma = second_order[g, :, :] / zero_order[g] - mu @ mu.T
                # HERE GO DIAG, TIED ECC

                if self._type == 'diag':
                    sigma = np.abs(sigma * np.eye(sigma.shape[-1]))

                # EIGENVALUE CONSTRAINING
                sigma = GMM._constrain_covariances_eigs(sigma, psi)
                w = zero_order[g] / zero_order.sum()
                new_gmm.append((w, mu, sigma))

            if self._type == 'tied':
                s = np.zeros_like(new_gmm[0][2])
                for g in range(num_components):
                    s += zero_order[g] * new_gmm[g][2]

                s /= num_samples
                new_gmm = [(new_gmm[g][0], new_gmm[g][1], s) for g in range(num_components)]
            return new_gmm

        avg_ll_new = avg_ll = None

        gmm = gmm0
        n_iter = 0
        while True:
            n_iter += 1
            avg_ll = avg_ll_new
            posteriors, avg_ll_new = expectation(gmm)
            if avg_ll is not None and (avg_ll_new - avg_ll) > 1e-6:
                break
            gmm = maximization(posteriors)
        return gmm

    def _lbg(self, dataset, desired_n_components, alpha=0.1, psi=0.01):
        def split_gmm(gmm_to_split):
            # gmm in the form [(w0, µ0, ∑0), (w1, µ1, ∑1), ...]
            new_gmm = []
            for g in range(len(gmm_to_split)):
                u, s, _ = np.linalg.svd(gmm_to_split[g][2])
                d = u[:, 0:1] * s[0] ** 0.5 * alpha
                new_gmm.append((gmm_to_split[g][0]/2, gmm_to_split[g][1]-d, gmm_to_split[g][2]))
                new_gmm.append((gmm_to_split[g][0]/2, gmm_to_split[g][1]+d, gmm_to_split[g][2]))

            return new_gmm

        mu0 = vcol(empirical_dataset_mean(dataset))
        c0 = empirical_dataset_covariance(dataset)
        c0 = GMM._constrain_covariances_eigs(c0, psi)
        num_components = 1
        gmm = [(1, mu0, c0)]
        while num_components < desired_n_components:
            gmm = split_gmm(gmm)
            gmm = self._em_estimation(dataset, gmm, psi)
            num_components = len(gmm)

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
            weights = vcol([gmm[g][0] for g in range(len(gmm))])
            self._model.add_gmm(gmm)
            score_matrix = self._model.log_pdf(dataset, c) + np.log(weights)
            self._model.add_score(special.logsumexp(score_matrix, axis=0))

    def classify(self, testing_data: np.ndarray, priors: np.ndarray) -> np.ndarray:
        num_samples = testing_data.shape[1]
        num_classes = len(set(self.training_labels))

        def log_likelihood(gmm):
            num_components = len(gmm)
            ll = np.empty((num_components, num_samples))
            for g in range(num_components):
                ll[g, :] = vrow(np.log(gmm[g][0])) + vrow(GMM._logpdf_GAU_ND(testing_data, gmm[g][1], gmm[g][2]))
            return special.logsumexp(ll, axis=0)

        # BINARY: Compute llr to score testing samples
        if num_classes == 2:
            ll1 = log_likelihood(self._model.get_gmm(1))
            ll0 = log_likelihood(self._model.get_gmm(0))

            self._scores = ll1 - ll0
            if priors is not None:
                thresh = np.log(priors[1] / priors[0])
                predictions = np.array(self._scores > thresh).astype(int)
                return predictions
        else:
            lls = []
            for c in range(num_classes):
                lls.append(log_likelihood(self._model.get_gmm(c)))

            lls = np.vstack(lls)

            log_priors = vcol(np.log(priors))
            log_joint = log_priors + lls

            marginal_log_densities = vrow(special.logsumexp(log_joint, axis=0))

            log_posterior_densities = log_joint - marginal_log_densities
            posterior_probs = np.exp(log_posterior_densities)
            self._scores = posterior_probs
            if priors is not None:
                predictions = np.argmax(posterior_probs, axis=0)
                return predictions

    def get_llrs(self):
        return self._scores


def tuning_componentsGMM(training_data, training_labels, alpha=0.1, psi=0.01):
    variants = ['full-cov', 'diag', 'tied']
    raw = [True, False]
    m_values = [None, 7]
    components_values = [2**i for i in range(9)]
    pis = [0.1, 0.9]

    hyperparameters = list(itertools.product(variants, raw, m_values, pis))

    # CICCIO: 12:24
    curr_hyp = hyperparameters[0:12]

    i = 0
    for variant, r, m, p in curr_hyp:
        DCFs = []
        for g in components_values:
            print(f"Inner iteration {i+1}/{len(curr_hyp)*len(components_values)}")
            llrs, evalutationLabels = k_fold(training_data, training_labels, GMM, 5, seed=0, raw=r, m=m, type=variant,
                                             alpha=alpha, psi=psi, G=g)
            min_dcf = compute_min_DCF(llrs, evalutationLabels, p, 1, 1)
            DCFs.append(min_dcf)
            i += 1
        np.save(f"simulations/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}_pi{str(p).replace('.', '-')}", DCFs)
