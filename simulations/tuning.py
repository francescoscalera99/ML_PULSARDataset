import itertools
import numpy as np

from classifiers.GMM2 import GMM
from classifiers.LR import LR
from classifiers.SVM import SVM
from preprocessing.preprocessing import gaussianize, PCA
from utils.metrics_utils import compute_min_DCF
from utils.utils import k_fold


def tuning_lambda_evaluation(training_data, training_labels, testing_data, testing_labels):
    m_values = [False, None, 7, 5]
    lbd_values = np.logspace(-8, +5, 70)
    lbd_values = np.array([0, *lbd_values])
    priors = [0.5, 0.1, 0.9]

    for m in m_values:
        for pi in priors:
            DCFs = []
            for (i, lbd) in enumerate(lbd_values):
                if m == False:
                    dtr = training_data
                    dte = testing_data
                else:
                    dtr_gauss = gaussianize(training_data, training_data)
                    dte_gauss = gaussianize(training_data, testing_data)
                    if m is not None:
                        dtr = PCA(dtr_gauss, dtr_gauss, m)
                        dte = PCA(dte_gauss, dtr_gauss, m)
                    else:
                        dtr = dtr_gauss
                        dte = dte_gauss

                logReg = LR(dtr, training_labels, lbd=lbd, pi_T=0.5)
                logReg.train_model()
                logReg.classify(dte, None)
                llrs = logReg.get_llrs()

                min_dcf = compute_min_DCF(np.array(llrs), testing_labels, pi, 1, 1)
                print(f"Done iteration {i+1} data:PCA{m}, lbd={lbd}, pi={pi} -> min DCF", min_dcf)
                DCFs.append(min_dcf)
            np.save(f"simulations/evaluation/LR/LR_EVAL_prior_{str(pi).replace('.', '-')}_PCA{m}", np.array(DCFs))


def tuning_parameters_LinearSVMUnbalanced_evaluation(training_data, ltr, testing_data, evaluationLabels):
    C_values = np.logspace(-2, 2, 20)
    K_values = [1.0]
    priors = [0.5, 0.1, 0.9]
    ms = [False, None, 7, 5]

    hyperparameters = itertools.product(ms, K_values, priors)
    for m, K, p in hyperparameters:
        DCFs = []
        for i, C in enumerate(C_values):
            if m == False:
                dtr = training_data
                dte = testing_data
            else:
                dtr_gauss = gaussianize(training_data, training_data)
                dte_gauss = gaussianize(training_data, testing_data)
                if m is not None:
                    dtr = PCA(dtr_gauss, dtr_gauss, m)
                    dte = PCA(dte_gauss, dtr_gauss, m)
                else:
                    dtr = dtr_gauss
                    dte = dte_gauss

            svm = SVM(dtr, ltr, k=K, c=C, kernel_params=(1, 0), kernel_type='poly')
            svm.train_model(balanced=False, pi_T=None)
            svm.classify(dte, None)
            llrs = svm.get_llrs()
            min_dcf = compute_min_DCF(llrs, evaluationLabels, p, 1, 1)
            print(f"Dataset PCA{m} iteration {i + 1} ", "min_DCF for K = ", K, "with prior = ", p, "->",
                  min_dcf)
            DCFs.append(min_dcf)
        np.save(f"simulations/evaluation/linearSVM/unbalanced/K{str(K).replace('.', '-')}_p{str(p).replace('.', '-')}_PCA{m}",
                np.array(DCFs))


def tuning_parameters_LinearSVMBalanced_evaluation(training_data, ltr, testing_data, evaluationLabels):
    K_values = [1.0]
    priors = [0.5, 0.1, 0.9]
    pi_T_values = [0.5, 0.1, 0.9]
    ms = [False, None, 7, 5]
    C_values = np.logspace(-2, 2, 20)
    h = itertools.product(ms, pi_T_values, K_values, priors)

    for i, (m, pi_T, K, p) in enumerate(h):
        DCFs = []
        for i, C in enumerate(C_values):
            if m == False:
                dtr = training_data
                dte = testing_data
            else:
                dtr_gauss = gaussianize(training_data, training_data)
                dte_gauss = gaussianize(training_data, testing_data)
                if m is not None:
                    dtr = PCA(dtr_gauss, dtr_gauss, m)
                    dte = PCA(dte_gauss, dtr_gauss, m)
                else:
                    dtr = dtr_gauss
                    dte = dte_gauss

            svm = SVM(dtr, ltr, k=K, c=C, kernel_params=(1, 0), kernel_type='poly')
            svm.train_model(balanced=True, pi_T=pi_T)
            svm.classify(dte, None)

            llrs = svm.get_llrs()
            min_dcf = compute_min_DCF(llrs, evaluationLabels, p, 1, 1)
            print(f"Dataset PCA{m} iteration {i + 1} ", "min_DCF for K = ", K, "with prior = ", p, "pi_T = ",
                  pi_T, "->",
                  min_dcf)
            DCFs.append(min_dcf)
        np.save(
            f"simulations/evaluation/linearSVM/balanced/K{str(K).replace('.', '-')}_p{str(p).replace('.', '-')}_pT{str(pi_T).replace('.', '-')}_PCA{m}",
            np.array(DCFs))


def tuning_parameters_PolySVM_evaluation(training_data, ltr, testing_data, evaluationLabels):
    # m_values = [False, None, 7, 5]
    # m_values = [7, 5]
    m_values = [False, None]
    C_values = np.logspace(-3, 3, 20)
    K_values = [1.0]
    c_values = [0, 1, 10, 15]

    for m in m_values:
        hyperparameters = itertools.product(c_values, K_values)
        for c, K in hyperparameters:
            DCFs = []
            for i, C in enumerate(C_values):
                if m == False:
                    dtr = training_data
                    dte = testing_data
                else:
                    dtr_gauss = gaussianize(training_data, training_data)
                    dte_gauss = gaussianize(training_data, testing_data)
                    if m is not None:
                        dtr = PCA(dtr_gauss, dtr_gauss, m)
                        dte = PCA(dte_gauss, dtr_gauss, m)
                    else:
                        dtr = dtr_gauss
                        dte = dte_gauss

                svm = SVM(dtr, ltr, k=K, c=C, kernel_params=(2, c), kernel_type='poly')
                svm.train_model(balanced=True, pi_T=0.5)
                svm.classify(dte, None)

                llrs = svm.get_llrs()
                min_dcf = compute_min_DCF(llrs, evaluationLabels, 0.5, 1, 1)
                print(i + 1, f"PCA{m} min_DCF for C ={C} with c ={c} and K ={K}-> {min_dcf}")
                DCFs.append(min_dcf)
            np.save(f"simulations/evaluation/polySVM/K{str(K).replace('.', '-')}_c{str(c).replace('.', '-')}_PCA{str(m)}", np.array(DCFs))


def tuning_parameters_RBFSVM_evaluation(training_data, ltr, testing_data, evaluationLabels):
    m_values = [False, None, 7, 5]
    C_values = np.logspace(-3, 3, 20)
    K_values = [0.0]
    gamma_values = [1e-2, 1e-3, 1e-4]

    for m in m_values:
        hyperparameters = itertools.product(gamma_values, K_values)
        for gamma, K in hyperparameters:
            DCFs = []
            for (i, C) in enumerate(C_values):
                if m == False:
                    dtr = training_data
                    dte = testing_data
                else:
                    dtr_gauss = gaussianize(training_data, training_data)
                    dte_gauss = gaussianize(training_data, testing_data)
                    if m is not None:
                        dtr = PCA(dtr_gauss, dtr_gauss, m)
                        dte = PCA(dte_gauss, dtr_gauss, m)
                    else:
                        dtr = dtr_gauss
                        dte = dte_gauss

                svm = SVM(dtr, ltr, k=K, c=C, kernel_params=gamma, kernel_type='RBF')
                svm.train_model(balanced=True, pi_T=0.5)
                svm.classify(dte, None)

                llrs = svm.get_llrs()
                min_dcf = compute_min_DCF(llrs, evaluationLabels, 0.5, 1, 1)
                print(i + 1, "min_DCF for C = ", C, "with gamma = ", gamma, "and K =", K, "->", min_dcf)
                DCFs.append(min_dcf)
            np.save(f"simulations/evaluation/RBF/RBF_K{str(K).replace('.', '-')}_gamma{str(gamma).replace('.', '-')}_PCA{str(m)}", np.array(DCFs))


def tuning_componentsGMM_evaluation(training_data, ltr, testing_data, evaluationLabels, alpha=0.1, psi=0.01):
    variants = ['full-cov', 'diag', 'tied']
    raw = [True, False]
    m_values = [None, 7]
    components_values = [2**i for i in range(9)]
    pis = [0.1, 0.5, 0.9]

    hyperparameters = list(itertools.product(variants, raw, m_values, pis))

    i = 0
    for variant, r, m, p in hyperparameters:
        DCFs = []
        for g in components_values:
            print(f"Inner iteration {i+1}/{len(hyperparameters)*len(components_values)}")
            if raw:
                if m is not None:
                    dtr = PCA(training_data, training_data, m)
                    dte = PCA(testing_data, training_data, m)
                else:
                    dtr = training_data
                    dte = testing_data
            else:
                dtr_gauss = gaussianize(training_data, training_data)
                dte_gauss = gaussianize(training_data, testing_data)
                if m is not None:
                    dtr = PCA(dtr_gauss, dtr_gauss, m)
                    dte = PCA(dte_gauss, dtr_gauss, m)
                else:
                    dtr = dtr_gauss
                    dte = dte_gauss

            gmm = GMM(dtr, ltr, type=variant)
            gmm.train_model(alpha=alpha, psi=psi, G=g)
            gmm.classify(dte, None)

            llrs = gmm.get_llrs()
            min_dcf = compute_min_DCF(llrs, evaluationLabels, p, 1, 1)
            DCFs.append(min_dcf)
            i += 1
        np.save(f"simulations/evaluation/GMM/GMM_rawFeature-{r}_PCA{m}_{variant}_pi{str(p).replace('.', '-')}", DCFs)

