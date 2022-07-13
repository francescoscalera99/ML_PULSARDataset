# MLPR Project: Pulsar detection

### Team members
* Francesco Scalera (s292432)
* Riccardo Sepe (s287760)


### Project structure
The core of our project is the `classifiers` folder: it contains the classes that model the classifiers we used.
The base abstract class `ClassifierClass` is contained in `Classifier.py`: its subclasses can be constructed by providing the training data and labels and optionally some `**kwargs` that will vary based on which is the subclass. It exposes three abstract methods: `train_model()`, `classify()` and `get_scores()`. The latter is used to return the classifiers scores generated inside `classify()`, so it is not directly necessary for classification, but it's useful for our purpose of evaluating the performance of our model itself in the Optimal Bayes Decision (DCF computation) framework.
The `preprocessing` folder contains code useful for the pre-processing steps we considered and the `utils` folder contains some utility functions grouped by purpose: `matrix_utils`, `plot_utils` and `misc_utils`.
The `data`, `results` and `simulations` folders contain respectively the data, the results in terms of `DCFs` for each classifier and for each possible configuration and various collections of `.npy` files containing the values for the plots.
In file `main.py` is loaded the data and are called the tuning functions and simulations functions.


### Project requirements
The project requirements are gathered in the `requirements.txt` file. They are:
 * `distinctipy`: used to generate distinguishable colors for the plots
 * `matplotlib`: used to produce all the plots. As an additional requirement there must be a Latex compiler on the machine running the code to produce Latex labels
 * `numpy`: for numerical computations
 * `prettytable`: to produce human-readable tables with all the results
 * `scipy`: for numerical computations

### Credits
_R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656_

_R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1._
