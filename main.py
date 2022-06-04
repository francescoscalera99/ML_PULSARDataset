import numpy as np

from utils.matrix_utils import vrow, vcol
from utils.utils import load_dataset, \
                  z_normalization, \
                  gaussianize, \
                  compute_accuracy
from classifiers.MVG import MVG


def main():
    (dtr, ltr), (dte, lte) = load_dataset()
    titles = ['1. Mean of the integrated profile',
              '2. Standard deviation of the integrated profile',
              '3. Excess kurtosis of the integrated profile',
              '4. Excess kurtosis of the integrated profile',
              '5. Mean of the DM-SNR curve',
              '6. Standard deviation of the DM-SNR curve',
              '7. Excess kurtosis of the DM-SNR curve',
              '8. Skewness of the DM-SNR curve']

    z_normalized_dtr = z_normalization(dtr)

    gaussianized_dtr = gaussianize(z_normalized_dtr, z_normalized_dtr)
    # plot_histogram(gauss, ltr, titles, nbins=20)
    # create_heatmap(gaussianized_dtr, title="Whole dataset")
    # create_heatmap(gaussianized_dtr[:, ltr == 1], cmap="Blues", title="True class")
    # create_heatmap(gaussianized_dtr[:, ltr == 0], cmap="Greens", title="False class")

    mvg = MVG(gaussianized_dtr, ltr, variant='tied')
    mvg.train_model()
    predictions = mvg.classify(dte, np.array([0.9, 0.1]))
    acc, err = compute_accuracy(predictions, lte)
    print(round(err*100, 2), '%')
    cm = build_confusion_matrix(vcol(lte), vcol(predictions))
    print(cm)


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


if __name__ == '__main__':
    main()
