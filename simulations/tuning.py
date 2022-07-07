import numpy as np

from classifiers.LR import LR
from preprocessing.preprocessing import gaussianize, PCA
from utils.metrics_utils import compute_min_DCF
from utils.utils import k_fold


def tuning_lambda(training_data, training_labels):
    m_values = [False, None, 7, 5]
    lbd_values = np.logspace(-8, 5, 70)
    lbd_values = np.array([0, *lbd_values])
    priors = [0.5, 0.1, 0.9]

    for m in m_values:
        for pi in priors:
            DCFs = []
            for (i, lbd) in enumerate(lbd_values):
                if m == False:
                    llrs, evaluationLabels = k_fold(training_data, training_labels, LR, 5, m=None, raw=True, seed=0, lbd=lbd, pi_T=0.5)
                else:
                    llrs, evaluationLabels = k_fold(training_data, training_labels, LR, 5, m=m, raw=False, seed=0, lbd=lbd, pi_T=0.5)
                min_dcf = compute_min_DCF(np.array(llrs), evaluationLabels, pi, 1, 1)
                print(f"Done iteration {i+1} data:PCA{m}, lbd={lbd}, pi={pi} -> min DCF", min_dcf)
                DCFs.append(min_dcf)
            np.save(f"simulations/LR/LR_prior_{str(pi).replace('.', '-')}_PCA{m}", np.array(DCFs))


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
            np.save(f"simulations/evaluation/LR_EVAL_prior_{str(pi).replace('.', '-')}_PCA{m}", np.array(DCFs))