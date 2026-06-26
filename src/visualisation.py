import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.utils import get_final_importance_summary, get_cv_importance_summary


def histogram_plotting(df: pd.DataFrame, feature_cols, bins=10):
    """
    Plots histograms for each feature column, separated by target class and time bins.

    Parameters
    ----------
    df (pd.DataFrame): the input DataFrame containing the data
    feature_cols (list): list of feature column names to plot
    bins (int): number of bins to use for the histograms
    """
    # Timestamp to hue map
    df["time_bin"] = (df["timestamp"] - 1) // (df['timestamp'].max()/bins) + 1 # max timestamp = 800 in this case 

    # Split data
    df_gang_lower = df[df['target']==0]
    df_scram = df[df['target']==1]

    for col in feature_cols:
        # Drop NaNs and ensure numeric
        data = pd.to_numeric(df[col], errors='coerce').dropna()

        # Skip empty columns
        if data.empty:
            print(f"Skipping {col}: no valid data")
            continue

        # Compute shared bin edges
        bin_edges = np.histogram_bin_edges(data, bins=bins)

        # Create subplots (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Left plot: gang lower
        sns.histplot(
            data=df_gang_lower,
            x=col,
            hue='time_bin',
            multiple='stack',
            ax=axes[0], 
            stat='density', 
            bins=bin_edges
        )
        axes[0].set_title(f'{col} - Gang Lower')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Density')

        # Right plot: SCRAM
        sns.histplot(
            data=df_scram,
            x=col,
            hue='time_bin',
            multiple='stack',
            ax=axes[1], 
            stat='density',
            bins=bin_edges
        )
        axes[1].set_title(f'{col} - SCRAM')
        axes[1].set_xlabel(col)

        # Adjust layout
        plt.tight_layout()
        plt.show()
        
# Plotting function 
def plot(df: pd.DataFrame, feature_cols, ID, time_start=None, time_end=None):
    """
    df (pd.DataFrame): time series data containing 'timestamp' and 'ID' columns
    feature_cols (list): list of feature column names to plot
    ID (int): shutdown ID
    time_start (int): timestamp to start plotting
    time_end (int): timestart to end plotting 

    Plots all features of the time series data from time_start to time_end if given, else plots for the entire time interval of df. 
    """
    subset = df[(df['ID']==ID)]

    # if no time start and end is provided
    if time_start is None:
        time_start = subset['timestamp'].min()
    
    if time_end is None:
        time_end = subset['timestamp'].max()    

    subset = subset[subset['timestamp'].between(time_start, time_end)]

    # ChatGPT for subplotting
    n_features = len(feature_cols)

    ncols = 5
    nrows = (n_features + ncols - 1) // ncols  # ceiling division

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, 4 * nrows),
        sharex=True
    )

    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        axes[i].plot(subset['timestamp'], subset[col])
        axes[i].set_title(col)
        axes[i].set_ylabel(col)

    # Remove unused subplots (if features < grid size)
    for ax in axes[n_features:]:
        ax.remove()

    for ax in axes[-ncols:]:
        ax.set_xlabel("Time (s)")

    plt.suptitle(f"Shutdown ID {ID}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_correlation_matrix(X, y, max_features=None):
    """
    Plots a heatmap of the correlation matrix for the top features based on correlation with the target variable.
    
    Parameters
    ----------
    X: DataFrame
        The feature matrix
    y: array-like
        The target variable
    max_features: int, optional
        The maximum number of features to include in the correlation matrix
    """
    if max_features is None:
        max_features = X.shape[1] # all features

    # Create a DataFrame for correlation analysis
    df_features = pd.DataFrame(X, columns=X.columns)
    df_features['target'] = y
    
    # Compute the correlation matrix
    correlation_matrix = df_features.corr()

    # Keep only the top features based on correlation with target
    corr_target = correlation_matrix['target'].abs().sort_values(ascending=False)
    top_features = corr_target.head(max_features).index
    correlation_matrix = correlation_matrix.loc[top_features, top_features]

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix of Top {max_features} Features')
    plt.show()


def plot_feature_correlation_with_target(X, y, max_features=None):
    """
    Plots a bar chart of the correlation of each feature with the target variable, showing only the top features based on absolute correlation.
    
    Parameters
    ----------
    X: DataFrame
        The feature matrix
    y: array-like
        The target variable
    max_features: int, optional
        The maximum number of features to include in the correlation plot
    """
    if max_features is None:
        max_features = X.shape[1] # all features

    # Create a DataFrame for correlation analysis
    df_features = pd.DataFrame(X, columns=X.columns)
    df_features['target'] = y

    # Compute correlation with target
    corr_target = df_features.corr()['target'].drop('target')

    # Sort by absolute correlation but keep sign
    corr_target = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)

    # Select top features
    corr_top = corr_target[:max_features]

    # Plot
    plt.figure(figsize=(6, 6))
    sns.barplot(x=corr_top.values, y=corr_top.index)
    for i, v in enumerate(corr_top.values):
        plt.text(v, i, f"{v:.2f}", ha='left', va='center')
    plt.axvline(0, color='black', linewidth=1)
    # plt.title(f'Feature Correlation with Target, Top {max_features} Features', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=10)
    plt.ylabel('Features', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

    return corr_top

# SHOULD THIS BE INCLUDED STILL?

# def plot_train_vs_test(metrics_dict):
#     """
#     Plots the training and test performance metrics.

#     Parameters
#     ----------
#     metrics_dict: dict
#         A dictionary containing 'train' and 'test' keys, each with a sub-dictionary of performance metrics (e.g., AUC and accuracy).
#     """
#     metrics = ["AUC", "Accuracy"]
#     train_values = [
#         metrics_dict["train"]["auc"],
#         metrics_dict["train"]["accuracy"]
#     ]
#     test_values = [
#         metrics_dict["test"]["auc"],
#         metrics_dict["test"]["accuracy"]
#     ]

#     x = np.arange(len(metrics))
#     width = 0.35

#     plt.figure()
#     plt.bar(x - width/2, train_values, width, label="Train")
#     plt.bar(x + width/2, test_values, width, label="Test")

#     plt.xticks(x, metrics)
#     plt.ylabel("Score")
#     plt.title("Train vs Test Performance")
#     plt.legend()

#     plt.show()

# Epoch plot
def plot_training_curves(train_losses, val_losses, lrs=None):
    """
    Plots training and validation loss curves over epochs, with an optional secondary axis for learning rate.

    Parameters
    ----------
    train_losses: list or array-like
        Training losses over epochs
    val_losses: list or array-like
        Validation losses over epochs
    lrs: list or array-like, optional
        Learning rates over epochs (for secondary axis)
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure()

    # Loss curves
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Optional LR on second axis
    if lrs is not None:
        ax2 = plt.gca().twinx()
        ax2.plot(epochs, lrs, linestyle='--', label="Learning Rate")
        ax2.set_ylabel("Learning Rate")

    plt.show()


def plot_5fold_training_curves(train_losses_folds, val_losses_folds):
    """
    Plot training and validation loss curves
    for 5-fold cross-validation in a 2-row layout.

    Parameters
    ----------
    train_losses_folds : list of lists
        Training losses for each fold.
        Shape: [n_folds][n_epochs]

    val_losses_folds : list of lists
        Validation losses for each fold.
        Shape: [n_folds][n_epochs]
    """

    n_folds = len(train_losses_folds)

    # 2 rows × 3 columns
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 8),
        sharey=True
    )

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    for fold_idx in range(n_folds):

        ax = axes[fold_idx]

        train_losses = train_losses_folds[fold_idx]
        val_losses = val_losses_folds[fold_idx]

        epochs = np.arange(1, len(train_losses) + 1)

        # --------------------------
        # LOSS CURVES
        # --------------------------
        ax.plot(
            epochs,
            train_losses,
            label="Train Loss",
            linewidth=2
        )

        ax.plot(
            epochs,
            val_losses,
            label="Validation Loss",
            linewidth=2
        )

        # --------------------------
        # LABELS / STYLING
        # --------------------------
        ax.set_title(f"Fold {fold_idx + 1}")

        ax.set_xlabel("Epoch")

        if fold_idx % 3 == 0:
            ax.set_ylabel("Loss")

        ax.grid(alpha=0.3)

    # Remove unused subplot (6th panel)
    if n_folds < len(axes):
        fig.delaxes(axes[-1])

    # --------------------------
    # SHARED LEGEND
    # --------------------------
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02)
    )

    fig.suptitle(
        "Training and Validation Loss Across 5 Folds",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


def print_final_metrics(auc_scores, acc_scores, f1_scores, y_true, y_pred):
    """
    Prints the final performance metrics for a model in a formatted way.

    Parameters
    ----------
    auc_scores: list or array-like
        AUC scores from cross-validation
    acc_scores: list or array-like
        Accuracy scores from cross-validation
    f1_scores: list or array-like
        F1 scores from cross-validation
    y_true: array-like
        True labels for the test set
    y_pred: array-like
        Predicted labels for the test set
    """
    print(
        f"Final Performance Metrics\n"
        f"{'-'*40}\n"
        f"Mean AUC:       {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}\n"
        f"Mean Accuracy:  {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}\n"
        f"Mean F1 Score:  {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}\n"
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Recall, precision, F1 score
    from sklearn.metrics import classification_report
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred))



#### New plots (27.05.2026)

def plot_raw_timeseries(
    df,
    feature_cols,
    target_col="target",
    id_col="ID",
    time_col="timestamp",
    class_value=0,
    title="Class - Raw Time Series",
    ncols=4
):
    """
    Plots raw time series for each feature per ID, grouped by class.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    target_col : str
        Column indicating class label
    id_col : str
        ID column
    time_col : str
        Timestamp column
    class_value : int
        Which class to plot (0 or 1)
    title : str
        Figure title
    ncols : int
        Number of columns in subplot grid
    """

    if class_value is not None:
        subset = df[df[target_col] == class_value]
    else: 
        subset = df.copy()

    n_features = len(feature_cols)
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharex=False
    )

    axes = np.array(axes).flatten()

    for i, col in enumerate(feature_cols):

        ax = axes[i]

        for ID in subset[id_col].unique():

            ts = (
                subset[subset[id_col] == ID]
                .sort_values(time_col)
            )

            ax.plot(
                ts[time_col],
                ts[col],
                alpha=0.4,
                linewidth=1
            )

        ax.set_xlabel('Timestamp')
        ax.set_ylabel(col)
        ax.set_title(col, fontsize=10)
        ax.tick_params(axis='x', rotation=45)

    # Hide unused axes (important for non-perfect grids)
    for j in range(len(feature_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    plt.show()


# def top_outlier_plots(combined_df, outlier_df, feature_cols, top_n=5, chosen_ID=None):
#     """
#     Plot feature trajectories with:
#     - all events (light gray)
#     - top-N outliers
#     - optionally selected IDs (one or many)
#     """
#     # Top outlier IDs
#     top_outliers = (
#         outlier_df
#         .groupby("ID")["distance"]
#         .mean()
#         .sort_values(ascending=False)
#         .head(top_n)
#         .index
#     )

#     # -----------------------------------
#     # Plot
#     # -----------------------------------
#     for col in feature_cols:

#         plt.figure(figsize=(10, 5))

#         # Plot all IDs lightly
#         for ID in combined_df['ID'].unique():

#             subset = (
#                 combined_df[combined_df['ID'] == ID]
#                 .sort_values("timestamp")
#             )

#             plt.plot(
#                 subset['timestamp'],
#                 subset[col],
#                 alpha=0.15,
#                 color='gray'
#             )

#         # Highlight top outliers
#         for ID in top_outliers:

#             subset = (
#                 combined_df[combined_df['ID'] == ID]
#                 .sort_values("timestamp")
#             )

#             plt.plot(
#                 subset['timestamp'],
#                 subset[col],
#                 linewidth=3,
#                 label=f"ID: {ID}"
#             )

#         # OR: Highlight chosen ID
#         if chosen_ID is not None:

#             subset = (
#                 combined_df[combined_df['ID'] == chosen_ID]
#                 .sort_values("timestamp")
#             )

#             if len(subset) > 0:
#                     plt.plot(
#                         subset["timestamp"],
#                         subset[col],
#                         linewidth=3,
#                         label=f"ID: {chosen_ID}"
#                     )

#         # Median trajectory
#         mean_series = (
#             combined_df
#             .groupby("timestamp")[col]
#             .median()
#         )

#         plt.plot(
#             mean_series.index,
#             mean_series.values,
#             color='black',
#             linewidth=4,
#             linestyle='--',
#             label='Median'
#         )

#         plt.title(f"Outlier Detection - {col}")

#         plt.xlabel("Timestamp")
#         plt.ylabel(col)

#         plt.legend()

#         plt.tight_layout()
#         plt.show()

# # No outliers, if one already knows the IDs to plot
# def plot_selected_ids(
#     combined_df,
#     feature_cols,
#     chosen_IDs,
#     show_median=True,
# ):
#     """
#     Plot selected event IDs against all trajectories.

#     Parameters
#     ----------
#     combined_df : pd.DataFrame
#         Must contain columns:
#         ['ID', 'timestamp'] + feature_cols

#     feature_cols : list[str]
#         Features to plot.

#     chosen_IDs : int, str, or list
#         ID or list of IDs to highlight.

#     show_median : bool, default=True
#         Whether to overlay the median trajectory.
#     """

#     # Allow a single ID
#     if not isinstance(chosen_IDs, (list, tuple, set)):
#         chosen_IDs = [chosen_IDs]

#     for col in feature_cols:

#         plt.figure(figsize=(10, 5))

#         # Background trajectories
#         for ID in combined_df["ID"].unique():

#             subset = (
#                 combined_df[combined_df["ID"] == ID]
#                 .sort_values("timestamp")
#             )

#             plt.plot(
#                 subset["timestamp"],
#                 subset[col],
#                 color="gray",
#                 alpha=0.15,
#             )

#         # Highlight chosen IDs
#         for ID in chosen_IDs:

#             subset = (
#                 combined_df[combined_df["ID"] == ID]
#                 .sort_values("timestamp")
#             )

#             if len(subset) == 0:
#                 continue

#             plt.plot(
#                 subset["timestamp"],
#                 subset[col],
#                 linewidth=3,
#                 label=f"ID {ID}",
#             )

#         # Median trajectory
#         if show_median:
#             median_series = (
#                 combined_df
#                 .groupby("timestamp")[col]
#                 .median()
#             )

#             plt.plot(
#                 median_series.index,
#                 median_series.values,
#                 color="black",
#                 linewidth=4,
#                 linestyle="--",
#                 label="Median",
#             )

#         plt.title(f"{col}")
#         plt.xlabel("Timestamp")
#         plt.ylabel(col)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()


def top_outlier_plots(combined_df, outlier_df, feature_cols, top_n=5, chosen_ID=None):
    """
    Plot feature trajectories with:
    - all events (light gray)
    - top-N outliers
    - optionally selected IDs (one or many)
    """

    # Ensure chosen_ID is list
    if chosen_ID is None:
        chosen_IDs = []
    elif isinstance(chosen_ID, (list, tuple, set, np.ndarray)):
        chosen_IDs = list(chosen_ID)
    else:
        chosen_IDs = [chosen_ID]


    # Top outliers
    top_outliers = (
        outlier_df
        .groupby("ID")["distance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

    # Plot per feature
    for col in feature_cols:

        plt.figure(figsize=(10, 5))

        # All IDs in light grey
        for ID in combined_df["ID"].unique():

            subset = (
                combined_df[combined_df["ID"] == ID]
                .sort_values("timestamp")
            )

            plt.plot(
                subset["timestamp"],
                subset[col],
                color="gray",
                alpha=0.15,
            )

        # Top outliers
        for ID in top_outliers:

            subset = (
                combined_df[combined_df["ID"] == ID]
                .sort_values("timestamp")
            )

            if len(subset) > 0:
                plt.plot(
                    subset["timestamp"],
                    subset[col],
                    linewidth=3,
                    label=f"ID: {ID}",
                )

        # User-selected IDs
        for ID in chosen_IDs:

            subset = (
                combined_df[combined_df["ID"] == ID]
                .sort_values("timestamp")
            )

            if len(subset) > 0:
                plt.plot(
                    subset["timestamp"],
                    subset[col],
                    linewidth=3,
                    label=f"ID: {ID}",
                )

        # Median trajectory
        mean_series = (
            combined_df
            .groupby("timestamp")[col]
            .median()
        )

        plt.plot(
            mean_series.index,
            mean_series.values,
            color="black",
            linewidth=4,
            linestyle="--",
            label="Median",
        )

        plt.title(f"Outlier Detection - {col}")
        plt.xlabel("Timestamp")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.show()


def top_outlier_plots_grid(
    combined_df,
    outlier_df,
    feature_cols,
    top_n=5,
    chosen_ID=None,
):
    """
    Grid version:
    - One figure
    - Subplots per feature (default 4x2)
    - Same overlays as before
    """

    # ----------------------------
    # Normalize chosen IDs
    # ----------------------------
    if chosen_ID is None:
        chosen_IDs = []
    elif isinstance(chosen_ID, (list, tuple, set, np.ndarray)):
        chosen_IDs = list(chosen_ID)
    else:
        chosen_IDs = [chosen_ID]

    # ----------------------------
    # Top outliers
    # ----------------------------
    if outlier_df is not None and len(outlier_df) > 0 and top_n > 0:
        top_outliers = (
            outlier_df
            .groupby("ID")["distance"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )
    else:
        top_outliers = []

    # ----------------------------
    # Figure setup (4x2 default)
    # ----------------------------
    n_features = len(feature_cols)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 4 * n_rows),
        sharex=True
    )

    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    # ----------------------------
    # Plot each feature
    # ----------------------------
    for i, col in enumerate(feature_cols):

        ax = axes[i]

        # -----------------------------------
        # Background trajectories
        # -----------------------------------
        for ID in combined_df["ID"].unique():

            subset = (
                combined_df[combined_df["ID"] == ID]
                .sort_values("timestamp")
            )

            ax.plot(
                subset["timestamp"],
                subset[col],
                color="gray",
                alpha=0.15,
            )

        # -----------------------------------
        # Top outliers
        # -----------------------------------
        for ID in top_outliers:

            subset = (
                combined_df[combined_df["ID"] == ID]
                .sort_values("timestamp")
            )

            if len(subset) > 0:
                ax.plot(
                    subset["timestamp"],
                    subset[col],
                    linewidth=2.5,
                    label=f"ID: {ID}",
                )

        # -----------------------------------
        # Chosen IDs
        # -----------------------------------
        for ID in chosen_IDs:

            subset = (
                combined_df[combined_df["ID"] == ID]
                .sort_values("timestamp")
            )

            if len(subset) > 0:
                ax.plot(
                    subset["timestamp"],
                    subset[col],
                    linewidth=2.5,
                    label=f"ID: {ID}",
                )

        # -----------------------------------
        # Median trajectory
        # -----------------------------------
        median_series = (
            combined_df
            .groupby("timestamp")[col]
            .median()
        )

        ax.plot(
            median_series.index,
            median_series.values,
            color="black",
            linewidth=3,
            linestyle="--",
            label="Median",
        )

        # -----------------------------------
        # Mean trajectory
        # -----------------------------------
        mean_series = (
            combined_df
            .groupby("timestamp")[col]
            .mean()
        )

        ax.plot(
            mean_series.index,
            mean_series.values,
            color="gray",
            linewidth=3,
            linestyle="--",
            label="Mean",
        )

        ax.set_title(col)
        ax.set_ylabel(col)
        ax.set_xlabel("Timestamp")
        ax.grid(True, alpha=0.2)

    # ----------------------------
    # Hide empty subplots
    # ----------------------------
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # ----------------------------
    # Shared labels + legend
    # ----------------------------
    axes[min(i, len(axes)-1)].set_xlabel("Timestamp")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    # plt.suptitle("Outlier Events", y=1.00, fontsize=20)
    plt.suptitle("Frequently Misclassified Events", y=1.00, fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_selected_events_grid(
    combined_df,
    feature_cols,
    IDs,
):
    """
    Plot selected event trajectories against the full dataset.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Must contain:
            - ID
            - timestamp
            - feature columns

    feature_cols : list
        Features to plot.

    IDs : list
        Event IDs to highlight.
    """

    # Ensure list
    if not isinstance(IDs, (list, tuple, set, np.ndarray)):
        IDs = [IDs]

    # ----------------------------
    # Figure setup
    # ----------------------------
    n_features = len(feature_cols)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 4 * n_rows),
        sharex=True
    )

    axes = np.array(axes).reshape(-1)

    # ----------------------------
    # Plot each feature
    # ----------------------------
    for i, col in enumerate(feature_cols):

        ax = axes[i]

        # -----------------------------------
        # Background trajectories
        # -----------------------------------
        for event_id in combined_df["ID"].unique():

            subset = (
                combined_df[combined_df["ID"] == event_id]
                .sort_values("timestamp")
            )

            ax.plot(
                subset["timestamp"],
                subset[col],
                color="gray",
                alpha=0.15,
                linewidth=2,
            )

        # -----------------------------------
        # Median trajectory
        # -----------------------------------
        median_series = (
            combined_df
            .groupby("timestamp")[col]
            .median()
        )

        ax.plot(
            median_series.index,
            median_series.values,
            color="black",
            linewidth=3,
            linestyle="--",
            label="Median",
        )

        # -----------------------------------
        # Selected IDs
        # -----------------------------------
        for event_id in IDs:

            subset = (
                combined_df[combined_df["ID"] == event_id]
                .sort_values("timestamp")
            )

            if len(subset) == 0:
                continue

            ax.plot(
                subset["timestamp"],
                subset[col],
                linewidth=2.5,
                label=f"ID {event_id}",
            )

        # set log y-axis
        # ax.set_yscale("log")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.2)

    # ----------------------------
    # Remove unused axes
    # ----------------------------
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # ----------------------------
    # Shared labels
    # ----------------------------
    axes[min(i, len(axes) - 1)].set_xlabel("Timestamp")

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.suptitle("Selected Event Trajectories", fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_mean_std_timeseries(
    df,
    feature_cols,
    target_col="target",
    time_col="timestamp",
    label_0="Gang Lower",
    label_1="SCRAM"
):
    """
    Plots mean ± std time series per feature, splitting internally into:
    0 -> label_0
    1 -> label_1
    """

    df_0 = df[df[target_col] == 0]
    df_1 = df[df[target_col] == 1]

    mean_0 = df_0.groupby(time_col)[feature_cols].mean()
    std_0  = df_0.groupby(time_col)[feature_cols].std()

    mean_1 = df_1.groupby(time_col)[feature_cols].mean()
    std_1  = df_1.groupby(time_col)[feature_cols].std()

    for col in feature_cols:

        plt.figure(figsize=(10, 5))

        # -------------------------
        # Gang Lower (0)
        # -------------------------
        plt.plot(
            mean_0.index,
            mean_0[col],
            linewidth=2,
            label=label_0
        )

        plt.fill_between(
            mean_0.index,
            mean_0[col] - std_0[col],
            mean_0[col] + std_0[col],
            alpha=0.2
        )

        # -------------------------
        # SCRAM (1)
        # -------------------------
        plt.plot(
            mean_1.index,
            mean_1[col],
            linewidth=2,
            label=label_1
        )

        plt.fill_between(
            mean_1.index,
            mean_1[col] - std_1[col],
            mean_1[col] + std_1[col],
            alpha=0.2
        )

        plt.title(f"Mean ± Std Time Series - {col}")
        plt.xlabel("Timestamp")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.show()


def present_results_compact(results):
    """
    Present compact overview of model performance.

    Parameters
    ----------
    results : list of dict
        List containing result dictionaries.
    """

    rows = []

    for res in results:

        rows.append({
            "Model": res["model"],
            "Preprocessing": res["preprocessing"],

            "ACC": f"{np.mean(res['acc_scores']):.3f} ± {np.std(res['acc_scores']):.3f}",
            "F1":  f"{np.mean(res['f1_scores']):.3f} ± {np.std(res['f1_scores']):.3f}",
            "AUC": f"{np.mean(res['auc_scores']):.3f} ± {np.std(res['auc_scores']):.3f}",
        })

    df = pd.DataFrame(rows)

    return df


def plot_all_metrics(results):
    """
    Plot grouped bar chart with:
    - AUC mean ± std
    - ACC mean ± std
    - F1 mean ± std

    for every model/preprocessing combination.
    """

    labels = []

    auc_means, auc_stds = [], []
    acc_means, acc_stds = [], []
    f1_means, f1_stds = [], []

    # ------------------------------
    # EXTRACT RESULTS
    # ------------------------------

    for res in results:

        labels.append(
            f"{res['model']}\n({res['preprocessing']})"
        )

        auc_means.append(np.mean(res["auc_scores"]))
        auc_stds.append(np.std(res["auc_scores"]))

        acc_means.append(np.mean(res["acc_scores"]))
        acc_stds.append(np.std(res["acc_scores"]))

        f1_means.append(np.mean(res["f1_scores"]))
        f1_stds.append(np.std(res["f1_scores"]))

    # ------------------------------
    # BAR POSITIONS
    # ------------------------------

    x = np.arange(len(labels))
    width = 0.25

    # ------------------------------
    # PLOT
    # ------------------------------

    plt.figure(figsize=(14, 7))

    plt.bar(
        x - width,
        auc_means,
        width,
        yerr=auc_stds,
        capsize=5,
        label="AUC"
    )

    plt.bar(
        x,
        acc_means,
        width,
        yerr=acc_stds,
        capsize=5,
        label="ACC"
    )

    plt.bar(
        x + width,
        f1_means,
        width,
        yerr=f1_stds,
        capsize=5,
        label="F1"
    )

    # ------------------------------
    # FORMATTING
    # ------------------------------

    plt.xticks(
        x,
        labels,
        rotation=20,
        ha="right"
    )

    plt.ylabel("Score")
    plt.ylim(0.5, 1.12)

    plt.grid()

    plt.title("Model Performance Comparison (Mean ± Std)", fontsize=18)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results):

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, result in enumerate(results):
        
        y_true = result["y_true"]
        y_pred = result["y_pred"]
        model_name = result["model"]

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[i], colorbar=False)

        # Set subplot title
        axes[i].set_title(
        f"{model_name}\n({result['preprocessing']})",
        fontsize=11
        )

    # Remove empty space and improve layout
    plt.tight_layout()

    # Optional overall title
    plt.suptitle("Confusion Matrices for All Models", fontsize=18, y=1.02)

    plt.show()


# def plot_misclassification_matrix(comparison_df):
#     """
#     Plot a misclassification matrix.

#     Parameters
#     ----------
#     comparison_df : pd.DataFrame

#         Columns:
#             event_id
#             y_true
#             model_1
#             model_2
#             ...

#     Red = incorrect prediction
#     White = correct prediction

#     Ground truth shown in first column.
#     """

#     model_cols = [
#         c for c in comparison_df.columns
#         if c not in ["event_id", "y_true"]
#     ]

#     # Build matrix
#     n_events = len(comparison_df)
#     n_models = len(model_cols)

#     # First column is always correct (ground truth)
#     plot_matrix = np.zeros((n_events, n_models + 1))

#     # Mark misclassifications
#     for i, model in enumerate(model_cols):

#         incorrect = (
#             comparison_df[model]
#             !=
#             comparison_df["y_true"]
#         )

#         plot_matrix[:, i + 1] = incorrect.astype(int)

#     # Plot
#     fig, ax = plt.subplots(
#         figsize=(max(10, n_models*1.5),
#                  max(6, n_events*0.3))
#     )

#     cmap = plt.cm.Reds

#     ax.imshow(
#         plot_matrix,
#         aspect="auto",
#         cmap=cmap,
#         vmin=0,
#         vmax=1
#     )

#     # Labels
#     ax.set_xticks(range(n_models + 1))
#     ax.set_xticklabels(
#         ["Ground Truth"] + model_cols,
#         rotation=45,
#         ha="right"
#     )

#     ax.set_yticks(range(n_events))
#     ax.set_yticklabels(
#         comparison_df["event_id"].astype(str)
#     )

#     ax.set_xlabel("Model")
#     ax.set_ylabel("Event ID")
#     ax.set_title("Misclassification Matrix", fontsize=18)

#     # Write values inside cells
#     ax.set_xticks(np.arange(-0.5, n_models + 1, 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, n_events, 1), minor=True)
#     ax.grid(which="minor", color="grey", linewidth=0.5)

#     plt.tight_layout()
#     plt.show()

def plot_misclassification_matrix(comparison_df):
    """
    Plot a misclassification matrix.

    Parameters
    ----------
    comparison_df : pd.DataFrame

        Columns:
            event_id
            y_true
            model_1
            model_2
            ...

    Red   = incorrect prediction
    White = correct prediction

    Rows are sorted by event_id numerically.
    """

    # Sort event IDs numerically
    comparison_df = (
        comparison_df
        .copy()
        .sort_values(
            by="event_id",
            key=lambda x: x.astype(int)
        )
        .reset_index(drop=True)
    )

    model_cols = [
        c for c in comparison_df.columns
        if c not in ["event_id", "y_true"]
    ]

    n_events = len(comparison_df)
    n_models = len(model_cols)

    # Matrix: 0 = correct, 1 = incorrect
    plot_matrix = np.zeros((n_events, n_models))

    for i, model in enumerate(model_cols):

        incorrect = (
            comparison_df[model]
            !=
            comparison_df["y_true"]
        )

        plot_matrix[:, i] = incorrect.astype(int)

    # Plot
    fig, ax = plt.subplots(
        figsize=(
            max(8, n_models * 1.5),
            max(6, n_events * 0.3)
        )
    )

    ax.imshow(
        plot_matrix,
        aspect="auto",
        cmap=plt.cm.Reds,
        vmin=0,
        vmax=1
    )

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(
        model_cols,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(range(n_events))
    ax.set_yticklabels(
        comparison_df["event_id"].astype(str)
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Event ID")
    ax.set_title("Misclassification Matrix", fontsize=18)

    # Grid lines
    ax.set_xticks(
        np.arange(-0.5, n_models, 1),
        minor=True
    )
    ax.set_yticks(
        np.arange(-0.5, n_events, 1),
        minor=True
    )
    ax.grid(
        which="minor",
        color="grey",
        linewidth=0.5
    )

    plt.tight_layout()
    plt.show()


# def plot_feature_importance_grid(
#     results,
#     preprocessing_name,
#     max_num_features=15
# ):
#     """
#     Creates a 2x4 grid:

#     Row 1:
#         Final model feature importances

#     Row 2:
#         CV mean +/- std feature importances

#     Columns:
#         Different ML models
#     """

#     fig, axes = plt.subplots(
#         4,
#         2,
#         figsize=(26, 20)
#     )

#     fig.suptitle(
#         f"{preprocessing_name} Feature Importance",
#         fontsize=20
#     )

#     # Filter preprocessing type
#     filtered_results = [
#         r for r in results
#         if r["preprocessing"] == preprocessing_name
#     ]

#     # Sort for consistent ordering
#     filtered_results = sorted(
#         filtered_results,
#         key=lambda x: x["model"]
#     )

#     for col, result in enumerate(filtered_results):

#         model_name = result["model"]

#         # Column 1 — Final model importance
#         final_summary = get_final_importance_summary(
#             result["final_model"],
#             max_num_features=max_num_features
#         )

#         ax = axes[col, 0]
#         ax.set_ylabel(model_name, fontsize=16)
#         ax.tick_params(axis='y', labelsize=16)

#         ax.barh(
#             final_summary["Feature"],
#             final_summary["Importance"]
#         )

#         ax.invert_yaxis()

#         ax.set_xlim(0, 1.05)

#         ax.set_title(
#             f"{model_name} - Final Model"
#         )

#         ax.set_xlabel(
#             "Normalised Importance"
#         )

#         # Column 2 — CV importance
#         cv_summary = get_cv_importance_summary(
#             result["best_model_list"],
#             max_num_features=max_num_features
#         )

#         ax = axes[col, 1]
#         ax.tick_params(axis='y', labelsize=16)

#         ax.barh(
#             cv_summary.index,
#             cv_summary["Mean Importance"],
#             xerr=cv_summary["Std Importance"]
#         )

#         ax.invert_yaxis()

#         ax.set_xlim(0, 1.05)


#         ax.set_title(
#             f"{model_name} - CV Mean ± Std"
#         )

#         ax.set_xlabel(
#             "Normalised Importance"
#         )

#     plt.tight_layout()

#     plt.show()

def plot_feature_importance_grid(
    results,
    preprocessing_name,
    max_num_features=15
):

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(26, 20)
    )

    fig.suptitle(
        f"{preprocessing_name} Feature Importance",
        fontsize=28,
        fontweight="bold"
    )

    filtered_results = [
        r for r in results
        if r["preprocessing"] == preprocessing_name
    ]

    for col, result in enumerate(filtered_results):

        model_name = result["model"]

        # ==================================================
        # FINAL MODEL
        # ==================================================

        final_summary = get_final_importance_summary(
            result["final_model"],
            max_num_features=max_num_features
        )

        # Remove zero-importance features for cleaner plots
        final_summary = final_summary[
            final_summary["Importance"] > 0
        ]

        ax = axes[col, 0]

        ax.set_ylabel(
            model_name,
            fontsize=16,
            fontweight="bold"
        )

        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=12)

        ax.barh(
            final_summary["Feature"],
            final_summary["Importance"]
        )

        ax.invert_yaxis()

        ax.set_xlim(0, 1.05)

        ax.set_title(
            f"{model_name} - Final Model",
            fontsize=18,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Relative Importance",
            fontsize=14
        )

        # ==================================================
        # CV MODEL
        # ==================================================

        cv_summary = get_cv_importance_summary(
            result["best_model_list"],
            max_num_features=max_num_features
        )

        # Remove zero-importance features for cleaner plots
        cv_summary = cv_summary[
            cv_summary["Mean Importance"] > 0
        ]

        ax = axes[col, 1]

        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=12)

        ax.barh(
            cv_summary.index,
            cv_summary["Mean Importance"],
            xerr=cv_summary["Std Importance"]
        )

        ax.invert_yaxis()

        ax.set_xlim(0, 1.05)

        ax.set_title(
            f"{model_name} - CV Relative Importance (Mean ± Std)",
            fontsize=18,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Relative Importance",
            fontsize=14
        )

    plt.subplots_adjust(
        hspace=0.35,
        wspace=0.25,
        top=0.90
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    # return feature importances
    return cv_summary