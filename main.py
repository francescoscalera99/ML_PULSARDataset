import numpy as np

from utils.matrix_utils import vrow, vcol
from utils.plot_utils import plot_histogram
from utils.utils import load_dataset, \
    z_normalization, \
    gaussianize, \
    compute_accuracy, splitData_SingleFold
from classifiers.MVG import MVG
from sklearn import preprocessing


def main():
    (training_data, training_labels), _ = load_dataset()



    gaussianized_training_data = gaussianize(training_data, training_data)
    #gaussianized_dte = gaussianize(z_normalized_dte, z_normalized_dte)
    z_normalized_training_data = z_normalization(gaussianized_training_data)

    (gaussianized_dtr, ltr), (gaussianized_dte, lte) = splitData_SingleFold(z_normalized_training_data, training_labels, seed=0)

    titles = ['1. Mean of the integrated profile',
              '2. Standard deviation of the integrated profile',
              '3. Excess kurtosis of the integrated profile',
              '4. Excess kurtosis of the integrated profile',
              '5. Mean of the DM-SNR curve',
              '6. Standard deviation of the DM-SNR curve',
              '7. Excess kurtosis of the DM-SNR curve',
              '8. Skewness of the DM-SNR curve']

    # gaussianized_dtr = gaussianize(dtr, dtr)



    mvg = MVG(gaussianized_dtr, ltr, variant='tied')
    mvg.train_model()
    predictions = mvg.classify(gaussianized_dte, np.array([0.5, 0.5]))
    cm = build_confusion_matrix(lte, predictions)
    print(cm)

    llrs = mvg.get_llrs()
    # min_dcf = min_dcf_function((0.5, 1, 1), llrs, lte)
    min_dcf = compute_min_DCF(llrs, lte, 0.5, 1, 1)
    print(min_dcf)


def build_confusion_matrix(testing_labels: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    if testing_labels.size != predicted_labels.size:
        raise RuntimeError("Testing labels array and predicted labels array should have the same size.\n"
                           f"Got instead 'len(testing_labels)'={len(testing_labels)},"
                           f" 'len(predicted_labels)'={len(predicted_labels)}")

    num_classes = np.unique(testing_labels).size
    num_samples = testing_labels.size
    predicted_labels = list(predicted_labels)
    testing_labels = list(testing_labels)

    cf = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(num_samples):
        cf[predicted_labels[i], testing_labels[i]] += 1

    return cf


def min_dcf_function(working_point: tuple, llrs: np.ndarray, testing_labels: np.ndarray, plot=False, out=False):
    thresholds = np.array([-np.inf, *sorted(list(llrs)), +np.inf])
    min_dcf = +np.inf
    tprs = []
    fprs = []
    for t in thresholds:
        predicted_labels = predict_bayes_optimal_decisions(llrs, t)

        confusion_matrix = build_confusion_matrix(testing_labels, predicted_labels)

        _, normalized_bayes_risk = compute_bayes_risk(working_point, confusion_matrix, normalized=True)

        min_dcf = min(min_dcf, normalized_bayes_risk)

        tpr, fpr = tpr_fpr(confusion_matrix)
        tprs.append(tpr)
        fprs.append(fpr)

    return min_dcf


def predict_bayes_optimal_decisions(llrs: np.ndarray, threshold: float):
    # True class is on row 0 and false on row 1

    return np.array(llrs > threshold, dtype=int)


def compute_bayes_risk(working_point: tuple, confusion_matrix: np.ndarray, normalized=False):
    if len(working_point) != 3:
        raise RuntimeError("Error: working point must be a tuple of the type\n", "(π_1, C_fn, C_fp)")

    fn = confusion_matrix[0, 1]
    fp = confusion_matrix[1, 0]
    tn = confusion_matrix[0, 0]
    tp = confusion_matrix[1, 1]

    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    pi_1, c_fn, c_fp = working_point
    pi_0 = 1 - pi_1

    b_risk = (pi_1 * c_fn * fnr) + (pi_0 * c_fp * fpr)

    if normalized:
        b_dummy = min(pi_1 * c_fn, pi_0 * c_fp)
        return b_risk, b_risk / b_dummy
    else:
        return b_risk


def tpr_fpr(confusion_matrix: np.ndarray) -> tuple:
    fn = confusion_matrix[0, 1]
    fp = confusion_matrix[1, 0]
    tn = confusion_matrix[0, 0]
    tp = confusion_matrix[1, 1]

    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    tpr = 1 - fnr

    return tpr, fpr

def compute_OBD_given_treshold(llr, labels, treshold):
  nClasses = np.unique(labels).size
  OBD = np.zeros([nClasses, nClasses])
  for i in range(llr.size):
    if (llr[i]>treshold):
      OBD[1, labels[i]]+=1
    else:
      OBD[0, labels[i]]+=1
  return OBD


if __name__ == '__main__':
    main()
