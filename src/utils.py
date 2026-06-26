import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Set seed for reproducibility
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Inspired by ChatGPT (https://chatgpt.com/c/698b154f-3ef8-8333-9482-ed82a8761c64)
def extract_features(all_dfs, feature_cols):
    """
    all_dfs: List of DataFrames with time rows and feature columns
    feature_cols: List of feature column names to extract features from
    returns: DataFrame of extracted features, 1 row per df in all_dfs
    """

    # Initialise feature and target lists
    X_list = []

    # if target exists, create list for it
    target_exists = False
    if 'target' in all_dfs[0].columns: # Check first df
        target_exists = True
        y_list = []

    # Iterating through shut down events
    for df in all_dfs:
        features = {} # dictionary

        for col in feature_cols:
            x = df[col].values

            features[f"{col}_mean"] = x.mean()
            features[f"{col}_std"] = x.std()
            features[f"{col}_min"] = x.min()
            features[f"{col}_max"] = x.max()

            # trend (linear slope)
            t = np.arange(len(x))
            slope = np.polyfit(t, x, 1)[0]
            features[f"{col}_slope"] = slope

            features[f"{col}_last"] = x[-1]

        X_list.append(pd.DataFrame([features])) # append features as a DataFrame to the list
        
        if target_exists:
            y_list.append(df['target'][0])

    X = pd.concat(X_list, ignore_index=True) # combine all feature DataFrames into one

    if target_exists:
        return X, np.array(y_list)
    else:
        return X


def flatten_features(all_dfs, feature_cols):
    """
    all_dfs: List of DataFrames with time rows and feature columns
    feature_cols: List of feature column names to include in the flattened output
    returns: DataFrame of flattened features, 1 row per df in all_dfs
    """
    X_list = []

    # if target exists, create list for it
    target_exists = False
    if 'target' in all_dfs[0].columns: # Check first df
        target_exists = True
        y_list = []
    
    for df in all_dfs:
        X_list.append(df[feature_cols])

        if target_exists:
            y_list.append(df['target'][0])
    
    X = np.array(X_list)

    n_events, n_timesteps, n_features = X.shape
    X_flat = X.reshape(n_events, n_timesteps*n_features)

    # Renaming feature names to include timestamp
    feature_names = []

    for t in range(n_timesteps):
        for col in feature_cols:
            feature_names.append(f"{col}_t{t}")

    X_flat = pd.DataFrame(X_flat, columns=feature_names)

    if target_exists:
        return X_flat, np.array(y_list)
    else:
        return X_flat

def flatten_features_resampled_timesteps(all_dfs, feature_cols, step=80):
    """
    all_dfs: List of DataFrames with time rows and feature columns
    feature_cols: List of feature column names to include in the flattened output
    step: The step size for resampling (default is 80)
    returns: DataFrame of flattened features, 1 row per df in all_dfs
    """
    X_list = []
    y_list = []
    
    for df in all_dfs:
        # Resample by taking every `step`-th row #e.g. every 80th
        df_resampled = df.iloc[::step, :]

        # Store features and target
        X_list.append(df_resampled[feature_cols].values)
        y_list.append(df_resampled['target'].iloc[0])
    
    # Convert to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)

    n_events, n_timesteps, n_features = X.shape

    # Flatten
    X_flat = X.reshape(n_events, n_timesteps*n_features)

    # Renaming feature names with correct timestep spacing
    feature_names = []
    for t in range(n_timesteps):
        actual_timestep = t * step  # Calculate the actual timestep based on the resampling step
        for col in feature_cols:
            feature_names.append(f"{col}_t{actual_timestep}")

    # Convert to DataFrame
    X_flat = pd.DataFrame(X_flat, columns=feature_names)

    return X_flat, y


# def get_feature_importance(model):
#     """
#     model: an instance of a trained model (LogisticRegression, DecisionTree, RandomForest, XGBoost)
#     returns: DataFrame with columns 'Feature' and 'Importance'
#     """
#     if isinstance(model, LogisticRegression):
#         return pd.DataFrame({'Feature': model.model.feature_names_in_, 'Importance': np.abs(model.model.coef_[0])})
#     elif isinstance(model, (DecisionTree, RandomForest)):
#         return pd.DataFrame({'Feature': model.model.feature_names_in_, 'Importance': model.model.feature_importances_})
#     elif isinstance(model.model, XGBClassifier):
#         booster = model.model.get_booster()
#         score = booster.get_score(importance_type='gain') # the importance scores of the features (does not include all features, only those that were used in the trees)

#         # Get original feature names
#         all_features = model.model.feature_names_in_

#         # Create a dictionary with all features initialised to 0 importance
#         full_importance = {feature: 0.0 for feature in all_features}  

#         # Update the dictionary with the actual importance scores
#         for feature, importance in score.items():
#             full_importance[feature] = importance

#         return pd.DataFrame({'Feature': list(full_importance.keys()),
#                 'Importance': list(full_importance.values())}) 
#     else:
#         raise ValueError("Feature importance not implemented for this model type.")


# # Move to visualisation.py?
# def plot_feature_importance(model, max_num_features):
#     """
#     model: an instance of a trained model (LogisticRegression, DecisionTree, RandomForest, XGBoost)
#     max_num_features: the maximum number of features to display in the plot
#     Plots a horizontal bar chart of feature importance for the given model, showing only the top `max_num_features` features.
#     """
#     get_feature_importance(model).sort_values('Importance', ascending=False).head(max_num_features).plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6), title=f"Feature Importance for {model.__class__.__name__}, {max_num_features} most important features")
#     plt.gca().invert_yaxis()


# # Move?
# def plot_feature_importance_multiple_runs(model_class, X, y, num_runs=5, max_num_features=20):
#     """
#     model_class: the class of the model to train (LogisticRegression, DecisionTree, RandomForest, XGBoost)
#     X: feature DataFrame
#     y: target array
#     num_runs: the number of times to run the training process
#     max_num_features: the maximum number of features to display in the plot
#     Trains the specified model multiple times with different random splits of the data, collects feature importance scores, and plots the mean feature importance with error bars representing the standard deviation across runs."""
#     feature_importances = []
#     for i in range(num_runs):
#         # Split the data with a different random state
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=i, stratify=y)
        
#         # Train the model
#         model = model_class(random_state=i) # create instance of the model class
#         model.train(X_train, y_train) # train the model
        
#         # Get feature importances
#         temp = get_feature_importance(model)
#         feature_importances.append(temp['Importance'])

#     feature_importances = pd.DataFrame(feature_importances)
#     feature_importances.columns = temp['Feature']
#     feature_importances.index = range(1, num_runs+1)
    
#     # Summary
#     summary = pd.DataFrame({
#         "Mean Importance": feature_importances.mean(axis=0),
#         "Std Importance": feature_importances.std(axis=0)
#     })

#     summary = summary.sort_values("Mean Importance", ascending=False).head(max_num_features)

#     plt.errorbar(summary['Mean Importance'], summary.index, xerr=summary['Std Importance'], fmt='o')
#     plt.gca().invert_yaxis()
#     plt.xlabel('Mean Feature Importance with Std Dev')
#     plt.title(f'Feature Importance across {num_runs} runs')
#     plt.grid()
#     plt.show()
#     return summary


# def train_vs_test_performance(model, X_train, y_train, X_test, y_test):
#     """
#     model: an instance of a trained model (LogisticRegression, DecisionTree, RandomForest, XGBoost)
#     X_train: training features
#     y_train: training targets
#     X_test: test features
#     y_test: test targets
#     returns: dictionary with train and test performance metrics (AUC and accuracy)
#     Evaluates the model on both training and test data, returning a dictionary with AUC and accuracy for each.
#     """

#     auc_train, acc_train = evaluate_model(model, X_train, y_train)
#     auc_test, acc_test = evaluate_model(model, X_test, y_test)

#     return {
#         "train": {"auc": auc_train, "accuracy": acc_train},
#         "test": {"auc": auc_test, "accuracy": acc_test}
#     }


def get_final_param_grid(best_params_list):
    """
    best_params_list: a list of dictionaries containing the best hyperparameters from each fold of nested cross-validation
    returns: a dictionary where each key is a hyperparameter and the value is a list of unique values for that hyperparameter across all folds
    Creates a final parameter grid for hyperparameter tuning based on the best parameters obtained from nested cross-validation.
    """
    param_grid_final = {}
    for best_params in best_params_list:
        for key, value in best_params.items():
            if key not in param_grid_final:
                param_grid_final[key] = []
            if value not in param_grid_final[key]:
                param_grid_final[key].append(value)
    return param_grid_final


def save_model(model, filename):
    """
    model: trained model instance
    filename: base filename without extension

    Saves the model inside:
        models/models_%Y-%m-%d/

    Example:
        models/models_2026-05-23/2026-05-23_14-30-12_rf_model.pkl
    """

    # Date folder name
    date_today = datetime.now().strftime("%Y-%m-%d")

    # Create models/models_YYYY-MM-DD directory
    models_dir = Path("models") / f"models_{date_today}"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = models_dir / f"{timestamp}_{filename}.pkl"

    # Save model
    with open(filepath, "wb") as f: # open a text file
        pickle.dump(model, f) # serialise model


def load_model(filepath):
    """
    Loads a previously saved model.

    Parameters
    ----------
    filepath : str or Path
        Full path to the .pkl model file

    Returns
    -------
    model : object
        Deserialised trained model
    """

    filepath = Path(filepath)

    with open(filepath, "rb") as f:  # open binary file for reading
        model = pickle.load(f)       # deserialise model

    return model


## New funcs (27.05.26)
def top_outliers(combined_df, feature_cols):

    # To store scores
    outlier_scores = []

    # Check if target exists
    has_target = "target" in combined_df.columns

    # Loop through features
    for col in feature_cols:

        # Mean trajectory
        mean_series = (
            combined_df
            .groupby("timestamp")[col]
            .mean()
        )

        # Compare each ID's trajectory to the mean trajectory
        for ID in combined_df['ID'].unique():

            subset = (
                combined_df[combined_df['ID'] == ID]
                .sort_values("timestamp")
            )

            # Ensure aligned lengths
            if len(subset) != len(mean_series):
                continue

            values = subset[col].values

            # Euclidean distance to mean trajectory
            distance = np.linalg.norm(
                values - mean_series.values
            )

            result = {
                "ID": ID,
                "feature": col,
                "distance": distance
            }

            if has_target:
                result['target'] = subset['target'].iloc[0]

            outlier_scores.append(result)

    # Dataframe of outlier scores
    outlier_df = pd.DataFrame(outlier_scores)

    # Highest deviation IDs
    outlier_df = outlier_df.sort_values(
        "distance",
        ascending=False
    )

    return outlier_df

def get_feature_importance(model):
    """
    model: an instance of a trained model (LogisticRegression, DecisionTree, RandomForest, XGBoost)
    returns: DataFrame with columns 'Feature' and 'Importance'
    """

    # unwrap model if needed
    estimator = model.model if hasattr(model, "model") else model

    if isinstance(estimator, LogisticRegression):
        return pd.DataFrame({'Feature': estimator.feature_names_in_, 'Importance': np.abs(estimator.coef_[0])})
    elif isinstance(estimator, (DecisionTreeClassifier, RandomForestClassifier)):
        return pd.DataFrame({'Feature': estimator.feature_names_in_, 'Importance': estimator.feature_importances_})
    elif isinstance(estimator, XGBClassifier):
        booster = estimator.get_booster()
        score = booster.get_score(importance_type='gain') # the importance scores of the features (does not include all features, only those that were used in the trees)

        # Get original feature names
        all_features = estimator.feature_names_in_

        # Create a dictionary with all features initialised to 0 importance
        full_importance = {feature: 0.0 for feature in all_features}  

        # Update the dictionary with the actual importance scores
        for feature, importance in score.items():
            full_importance[feature] = importance

        return pd.DataFrame({'Feature': list(full_importance.keys()),
                'Importance': list(full_importance.values())}) 
    else:
        raise ValueError("Feature importance not implemented for this model type.")


def get_final_importance_summary(model, max_num_features=20):

    importance_df = get_feature_importance(model)

    # Normalise
    max_importance = importance_df["Importance"].max()
    if max_importance > 0:
        importance_df["Importance"] /= max_importance

    importance_df = (
        importance_df
        .sort_values("Importance", ascending=False)
        .head(max_num_features)
    )

    return importance_df

def get_cv_importance_summary(
    model_list,
    max_num_features=20
):

    all_importances = []

    for model in model_list:

        temp = get_feature_importance(model)

        temp = temp.set_index("Feature")

        importance = temp["Importance"]

        # normalise within fold
        importance = importance / importance.max()

        all_importances.append(importance)

    importance_df = pd.concat(all_importances, axis=1)

    summary = pd.DataFrame({
        "Mean Importance": importance_df.mean(axis=1),
        "Std Importance": importance_df.std(axis=1)
    })

    summary = (
        summary
        .sort_values("Mean Importance", ascending=False)
        .head(max_num_features)
    )

    return summary


# For confusion matrix, identifying which IDs were correctly or incorrectly classified by each model, and comparing across models
def create_comparison_df(results):
    # Use first model as reference
    comparison_df = pd.DataFrame({
        "event_id": results[0]["event_ids"],
        "y_true": results[0]["y_true"]
    })

    # Add predictions from each model/preprocessing combination
    for result in results:
        
        column_name = (
            f"{result['model']}"
            f" ({result['preprocessing']})"
        )

        comparison_df[column_name] = result["y_pred"]

    return comparison_df


def get_hard_events(comparison_df, threshold=3):
    """
    Get events that were misclassified by at least 'threshold' models.
    """

    model_cols = [col for col in comparison_df.columns
                if col not in ["event_id", "y_true"]]

    rows = []

    for _, row in comparison_df.iterrows():
        wrong_models = [
            model for model in model_cols
            if row[model] != row["y_true"]
        ]

        if len(wrong_models) > threshold:
            rows.append({
                "event_id": row["event_id"],
                "y_true": row["y_true"],
                "n_misclassified": len(wrong_models),
                "wrong_models": wrong_models
            })

    hard_events_detailed = pd.DataFrame(rows)

    return hard_events_detailed