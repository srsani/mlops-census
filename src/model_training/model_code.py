import logging
from sklearn.model_selection import (GridSearchCV,
                                     )
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (fbeta_score,
                             precision_score,
                             recall_score)
import sys
sys.path.append("..")


# logging.basicConfig(
#     filename='./logs/results.log',
#     level=logging.INFO,
#     filemode='a+',
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt="%Y-%m-%d %H:%M:%S")


def train_model(X_train, y_train):
    """GridSearchCV and trains a RandomForestClassifier model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(n_jobs=-1)
    n_estimators = list(range(100, 200, 100))
    min_samples_split = list(range(10, 50, 30))
    min_samples_leaf = list(range(1, 10, 50))
    criterion = ['entropy', ]
    param_grid = dict(n_estimators=n_estimators, min_samples_split=min_samples_split,
                      criterion=criterion, min_samples_leaf=min_samples_leaf,
                      )

    logging.info(param_grid)

    grid = GridSearchCV(model,
                        param_grid,
                        cv=5,
                        scoring='accuracy',
                        verbose=3)
    grid.fit(X_train, y_train)
    logging.info(f'best cv score: {grid.best_score_}\n')
    logging.info(f'best model: {grid.best_estimator_}\n')

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass
