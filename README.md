# Tabular Ad-Click Prediction Neural Network

This is the project I made for the module 2 of the course "Machine Learning, Artificial Neural Networks and Deep Learning", it contains a neural network for binary classification on tabular data: given a user's profile, the model predicts whether the user clicked on an advertisement (`Clicked on Ad`).
The notebook is structured as required by the exam paper.   

## Design choices
 
With fewer than 1000 samples, the main concern is overfitting. `Country` and `Search Queries` are compressed into 10-dimensional embeddings by two autoencoders, concatenated with the numerical features, and fed to a two-layer MLP: a three-layer version memorized the training data, and dropout and L2 regularization proved essential. `Ad Topic Line` and `City` are dropped because their values are almost unique per sample. Hyperparameters are selected with `RandomizedSearchCV` (4 folds), with the scaler inside the pipeline and the split done before any fitting to avoid leakage; the autoencoders are trained once on the full training set to keep computation affordable.

## Repository contents

```
.
├── TabularAdClickPredictionNN.ipynb   # notebook with the whole project
├── requirements.txt                   # dependencies
└── data/
    └── input_data.zip                 # dataset (contains input_data.pkl)
```

## Dataset

The dataset was directly supplied by the professor and it contains:
- 953 samples, 11 columns, no missing values.
- Target: `Clicked on Ad` (0/1).

## Installation and usage

```bash
git clone <repository-URL>
cd <folder-name>
pip install -r requirements.txt
jupyter notebook TabularAdClickPredictionNN.ipynb
```

The first cell of the notebook also installs `scikit-learn`, `scikeras` and `tensorflow`. Run the cells in order: the test set is used only in the final evaluation section.

Main libraries: `numpy`, `pandas`, `tensorflow` / `keras`, `scikit-learn`, `scikeras`, `scipy`, `matplotlib`.

For reproducibility, the seeds (`random`, `numpy`, `tensorflow`) are set to 123.

## Notes and limitations

- The autoencoders are trained once on the full training set rather than on each cross-validation fold: doing so for every hyperparameter combination would have required too much computation and running time.
- The concatenation of the inputs is done outside the model, because `RandomizedSearchCV` had problems handling the input dimensions during `.fit` with multiple inputs.
- The dataset is small (fewer than 1000 samples): for this reason the cross-validation uses 4 folds and the model is kept at two layers with dropout and L2 regularization.
- The exact L2 regularization value chosen by the search is not printed in the notebook output.
- The **CHANGE** markdown cells flag slight modifications to the solution I proposed during the written exam. As required by the professor, the implementation had to remain consistent with that solution.