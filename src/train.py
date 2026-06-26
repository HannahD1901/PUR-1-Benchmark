import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from src.evaluate import evaluate_auc, evaluate_accuracy, evaluate_f1


def train_model(model_class, X, y, param_grid=None, cv=5):
    """
    Train a classical ML model with optional hyperparameter tuning.

    Parameters
    ----------
    model_class : class
        Wrapper class containing `.model`
    X_train : DataFrame
    y_train : array-like
    param_grid : dict or None
    cv : int

    Returns
    -------
    best_model : sklearn estimator
    best_params : dict
    """

    # Fresh model instance
    clf = model_class(random_state=42).model

    if param_grid is not None:
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

    else:
        best_model = clf.fit(X, y)
        best_params = None

    return best_model, best_params

def nested_cross_validation(X, y, event_ids, model_class, param_grid, outer_folds=5, inner_folds=3):
    """
    Perform nested cross-validation.

    Parameters
    ----------
    X : DataFrame
    y : array-like
    event_ids : array-like
    model_class : class
    param_grid : dict
    outer_folds : int
    inner_folds : int

    Returns
    -------
    outer_auc_scores : list
    outer_acc_scores : list
    outer_f1_scores : list
    all_y_true : list
    all_y_pred : list
    """

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    # Lists to store results from outer cross-validation
    outer_auc_scores = []
    outer_acc_scores = []
    outer_f1_scores = []

    # For final confusion matrix
    all_event_ids = []
    all_y_true = []
    all_y_pred = []

    # Best model and hyperparameters for each fold
    best_model_list = []
    best_params_list = []

    # Outer CV
    for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X, y)):
        print(f"Outer fold {outer_fold+1}/{outer_folds}")

        # Split data into training and test sets for the outer fold
        X_train_outer, X_test_outer = X.iloc[train_outer_idx], X.iloc[test_outer_idx]
        y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

        # Inner CV
        best_model, best_params = train_model(
            model_class=model_class,
            X=X_train_outer,
            y=y_train_outer,
            param_grid=param_grid,
            cv=inner_folds
        )

        # Evaluation
        auc = evaluate_auc(best_model, X_test_outer, y_test_outer)
        acc = evaluate_accuracy(best_model, X_test_outer, y_test_outer)
        f1 = evaluate_f1(best_model, X_test_outer, y_test_outer)

        # Store metrics
        outer_auc_scores.append(auc)
        outer_acc_scores.append(acc)
        outer_f1_scores.append(f1)

        # Save event IDs for confusion matrix
        event_ids = np.asarray(event_ids)
        all_event_ids.extend(event_ids[test_outer_idx])

        # Store predictions for confusion matrix
        y_pred = best_model.predict(X_test_outer)

        all_y_true.extend(y_test_outer)
        all_y_pred.extend(y_pred)

        # Store best model and hyperparameters
        best_model_list.append(best_model)
        best_params_list.append(best_params)

    return outer_auc_scores, outer_acc_scores, outer_f1_scores, all_event_ids, all_y_true, all_y_pred, best_model_list, best_params_list


def final_cross_validation(X, y, model_class, param_grid_final, cv=5):

    # Fresh model instance
    clf = model_class(random_state=42).model

    # GridSearchCV to find the best hyperparameters from the final param grid
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid_final,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params