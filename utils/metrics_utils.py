import numpy as np


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


def compute_OBD_given_treshold(llr, labels, treshold):
  nClasses = np.unique(labels).size
  OBD = np.zeros([nClasses, nClasses])
  for i in range(llr.size):
    if (llr[i]>treshold):
      OBD[1, labels[i]]+=1
    else:
      OBD[0, labels[i]]+=1
  return OBD


def compute_normalizeDCF(optimal_bayes_decisions, prior_class_probability, Cfn, Cfp):
  FNR = optimal_bayes_decisions[0,1]/(optimal_bayes_decisions[0,1]+optimal_bayes_decisions[1,1])
  FPR = optimal_bayes_decisions[1,0]/(optimal_bayes_decisions[0,0]+optimal_bayes_decisions[1,0])

  DCF = (prior_class_probability*Cfn*FNR) + ((1-prior_class_probability)*Cfp*FPR)
  return DCF/min(prior_class_probability*Cfn, (1-prior_class_probability)*Cfp)


def compute_min_DCF(llr, labels, prior_class_probability, Cfn, Cfp):
    minDCF = np.inf
    tresholds = np.hstack(([-np.inf], llr, [np.inf]))
    tresholds.sort()
    for treshold in tresholds:
        OBD = compute_OBD_given_treshold(llr, labels, treshold)
        currentDCF = compute_normalizeDCF(OBD, prior_class_probability, Cfn, Cfp)
        if (currentDCF < minDCF):
            minDCF = currentDCF

    return minDCF

def main():
    pass


if __name__ == '__main__':
    main()
