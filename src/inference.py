from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.utils import load_model
from src.utils import extract_features, flatten_features

def load_all_models(models_dir):

    model_files = sorted(Path(models_dir).glob("*.pkl"))

    loaded_models = {}

    for filepath in model_files:

        filename = filepath.stem
        parts = filename.split("_")

        model_name = parts[2]
        preprocessing = "_".join(parts[3:])
        model_key = f"{model_name}_{preprocessing}"

        loaded_models[model_key] = {
            "model": load_model(filepath),
            "preprocessing": preprocessing
        }

        print(f"Loaded: {model_key}")

    return loaded_models


def prepare_inputs(blind_dfs, feature_cols, lstm_feature_cols):

    X_data = {

        "Extracted_features":
            extract_features(blind_dfs, feature_cols),

        "Flattened_features":
            flatten_features(blind_dfs, feature_cols),

        # special case, since LSTM was trained on three features only
        "raw_data":
            np.array([
                df[lstm_feature_cols].values
                for df in blind_dfs
            ])
    }

    return X_data


def run_inference(models, X_data):

    predictions = {}

    for model_key, model_info in models.items():

        model = model_info["model"]
        preprocessing = model_info["preprocessing"]

        X = X_data[preprocessing]
        print(f"\nRunning inference for {model_key} with preprocessing: {preprocessing}") # debugging
        print(f"Input shape for {model_key}: {X.shape}") # debugging

        # -----------------------------
        # SCIKIT MODELS
        # -----------------------------

        if hasattr(model, "predict"):

            y_pred = model.predict(X)

        # -----------------------------
        # PYTORCH MODELS
        # -----------------------------

        else:

            model.eval()

            X_tensor = torch.tensor(
                np.array(X),
                dtype=torch.float32
            )

            with torch.no_grad():
                outputs = model(X_tensor).squeeze()
        
                y_pred = (outputs >= 0.5).int().numpy()

        predictions[model_key] = y_pred

        print(f"\n{model_key}")
        print(y_pred)

    return predictions


def plot_vote_matrix_heatmap(predictions, sort_by_disagreement=True):
    """
    Plot a heatmap showing binary predictions from multiple models.
    """

    model_names = list(predictions.keys())

    pred_matrix = np.array([
        predictions[m] for m in model_names
    ])  # (n_models, n_samples)

    # Sort by disagreement (optional)
    if sort_by_disagreement:
        disagreement = pred_matrix.std(axis=0)
        sorted_idx = np.argsort(-disagreement)
        pred_matrix = pred_matrix[:, sorted_idx]

    plt.figure(figsize=(12, 6))

    im = plt.imshow(
        pred_matrix,
        aspect='auto',
        interpolation='nearest',
        cmap='viridis',
        vmin=0,
        vmax=1
    )

    # Draw vertical lines between samples
    for x in np.arange(-0.5, pred_matrix.shape[1], 1):
        plt.axvline(x=x, color='white', linewidth=0.3, alpha=0.5)

    plt.yticks(
        ticks=np.arange(len(model_names)),
        labels=model_names
    )

    plt.xlabel("Samples")
    plt.ylabel("Models")
    plt.title("Model Vote Matrix Heatmap")

    # --------------------------------------------------
    # SIMPLE LEGEND (instead of colorbar)
    # --------------------------------------------------
    legend_elements = [
        Patch(facecolor=im.cmap(im.norm(0)), label="Gang Lower"),
        Patch(facecolor=im.cmap(im.norm(1)), label="SCRAM"),
    ]

    plt.legend(
        handles=legend_elements,
        loc="upper right",
        title="Prediction"
    )

    plt.tight_layout()
    plt.show()


# Find outliers based on disagreement between models
def get_disagreement_IDs(predictions, blind_dfs_clean):
    """
    Identifies event IDs where there is disagreement between the models' predictions.
    """
    model_names = list(predictions.keys())

    pred_matrix = np.array([
        predictions[m] for m in model_names
    ])

    # if cases of both class 0 and class 1 predictions, then there is disagreement
    disagreement_mask = pred_matrix.max(axis=0) != pred_matrix.min(axis=0) 

    event_ids = []
    for blind_df in blind_dfs_clean:
        event_ids.append(blind_df['ID'][0])

    outlier_ids = np.array(event_ids)[disagreement_mask]
    return outlier_ids


# Majority voting
def majority_vote(predictions):

    pred_matrix = np.array(
        list(predictions.values())
    )

    final_pred = (
        pred_matrix.mean(axis=0) >= 0.5
    ).astype(int)

    return final_pred