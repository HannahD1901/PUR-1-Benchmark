import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score, 
    accuracy_score, 
    f1_score
)

def evaluate_auc(model, X, y):
    """
    Evaluates the model on the given data, returning the AUC score.

    Parameters
    ----------
    model: sklearn-like model
        an instance of a trained model (LogisticRegression, DecisionTree, RandomForest, XGBoost)
    X: DataFrame
        feature DataFrame
    y: array-like
        target array
    """
    y_proba = model.predict_proba(X)[:, 1]  # Get probabilities for positive class
    auc = roc_auc_score(y, y_proba)

    return auc

def evaluate_accuracy(model, X, y):
    """
    Evaluates the model on the given data, returning the accuracy score.

    Parameters
    ----------
    model: sklearn-like model
        an instance of a trained model (LogisticRegression, DecisionTree, RandomForest, XGBoost)
    X: DataFrame
        feature DataFrame
    y: array-like
        target array
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    return acc

def evaluate_f1(model, X, y):
    """
    Evaluates the model on the given data, returning the F1 score.

    Parameters
    ----------
    model: sklearn-like model
        an instance of a trained model (LogisticRegression, DecisionTree, RandomForest, XGBoost)
    X: DataFrame
        feature DataFrame
    y: array-like
        target array
    """
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred)

    return f1