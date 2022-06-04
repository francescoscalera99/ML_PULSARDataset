import numpy as np

from utils import load_dataset, \
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

    mvg = MVG(gaussianized_dtr, ltr)
    mvg.train_model()
    predictions = mvg.classify(dte, np.array([0.5, 0.5]))
    acc, err = compute_accuracy(predictions, lte)
    print(round(err*100, 2), '%')


if __name__ == '__main__':
    main()
