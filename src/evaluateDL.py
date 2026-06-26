from sklearn.metrics import roc_auc_score, f1_score
import torch
import numpy as np

def evaluate_auc_DL(model, data_loader):
    """
    Evaluates the model on the given data, returning the AUC score.

    Parameters
    ----------
    model: an instance of a trained PyTorch model (e.g., LSTMClassifier)
    data_loader: DataLoader providing batches of data to evaluate on
    """
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)

            y_true.extend(y_batch.numpy())
            y_pred.extend(outputs.numpy())

    auc = roc_auc_score(y_true, y_pred)

    return auc

def evaluate_f1_DL(model, data_loader, threshold=0.5):
    """
    Evaluates the model on the given data, returning the F1 score.

    Parameters
    ----------
    model: an instance of a trained PyTorch model (e.g., LSTMClassifier)
    data_loader: DataLoader providing batches of data to evaluate on
    threshold: float, optional
        The threshold for converting predicted probabilities to binary predictions
    """
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            probs = outputs

            y_true.extend(y_batch.numpy())
            y_pred.extend(probs.numpy())

    y_pred = (np.array(y_pred) > threshold).astype(int)

    return f1_score(y_true, y_pred)


def evaluate_accuracy_DL(model, data_loader, threshold=0.5):
    """
    Evaluates the model on the given data, returning the accuracy score.

    Parameters
    ----------
    model: an instance of a trained PyTorch model (e.g., LSTMClassifier)
    data_loader: DataLoader providing batches of data to evaluate on
    threshold: float, optional
        The threshold for converting predicted probabilities to binary predictions
    """
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            probs = outputs

            y_true.extend(y_batch.numpy())
            y_pred.extend(probs.numpy())

    y_pred = (np.array(y_pred) > threshold).astype(int)

    return np.mean(np.array(y_true) == y_pred)


def model_predict_proba(model, data_loader, device="cpu"):
    model.eval()
    probs = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            # If model outputs logits (most common)
            batch_probs = outputs

            probs.append(batch_probs.cpu().numpy())

    return np.concatenate(probs).flatten()