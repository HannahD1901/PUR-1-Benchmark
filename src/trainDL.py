import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from src.preprocessing import ReactorDataset
from torch.utils.data import DataLoader
from src.utils import set_seed
from src.evaluateDL import evaluate_accuracy_DL, evaluate_auc_DL, evaluate_f1_DL, model_predict_proba
import copy


# Training
def train_DL_model(model, train_loader, val_loader, epochs=150, lr=1e-3, patience=150):
    """
    Train the LSTM model.

    Args:
        model: an instance of LSTMClassifier
        train_loader: DataLoader for training data, meaning it should be an instance of torch.utils.data.DataLoader that provides batches of training data
        val_loader: DataLoader for validation data, meaning it should be an instance of torch.utils.data.DataLoader that provides batches of validation data
        epochs: number of training epochs
        lr: learning rate
        patience: number of epochs to wait for improvement in validation loss before early stopping
    Returns:
        train_losses: list of training losses for each epoch
        val_losses: list of validation losses for each epoch
    """

    criterion = nn.BCELoss() # Binary Cross Entropy Loss (commonly used in classification)
    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam optimizer is a popular choice

    train_losses = [] # To store training losses for each epoch
    val_losses = [] # To store validation losses for each epoch

    # For early stopping and rollback
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs): 
        # print(f"Training epoch {epoch+1}/{epochs}")

        model.train() # Set model to training mode (enables dropout, batch norm, etc.)
        train_loss = 0

        for X_batch, y_batch in train_loader: # Loop through batches of training data
            optimizer.zero_grad() # Zero the gradients before backpropagation

            outputs = model(X_batch) # Get model predictions for the batch (runs .forward())
            loss = criterion(outputs, y_batch) # Calculate loss between predictions and true labels

            loss.backward() # Backpropagation to compute gradients
            optimizer.step() # Update model parameters based on gradients

            train_loss += loss.item() * X_batch.size(0) # Accumulate training loss for the epoch

        train_loss /= len(train_loader.dataset) # Training loss: average loss over all samples in the epoch

        # Validation
        model.eval() # Set model to evaluation mode (disables dropout, batch norm, etc.)
        val_loss = 0

        with torch.no_grad(): # Disable gradient calculation for validation (saves memory and computations)
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch) # Get model predictions for the validation batch
                loss = criterion(outputs, y_batch) # Calculate loss for the validation batch
                val_loss += loss.item() * X_batch.size(0) # Accumulate validation loss for the epoch
        val_loss /= len(val_loader.dataset) # Validation loss: average loss over all batches in the validation set for the epoch

        train_losses.append(train_loss) 
        val_losses.append(val_loss)

        # Early stopping and rollback
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict()) # For rollback to work
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        # print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")

    # Load best weights before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return train_losses, val_losses


def nested_cross_validation_DL(model_class, X, y, event_ids, param_grid, outer_folds=5, inner_folds=3, epochs=150, patience=150):
    """
    Perform nested cross-validation for the LSTM model.

    Parameters
    ----------
    model_class: class
        The class of the model to be trained (e.g., LSTMClassifier)
    X: array-like
        Feature matrix 
    y: array-like
        Target vector
    param_grid: dict
        Grid of hyperparameters to tune ('hidden_size', 'lr')
    outer_folds: int
        Number of folds for the outer loop (model evaluation)
    inner_folds: int
        Number of folds for the inner loop (hyperparameter tuning)
    epochs: int
        Number of training epochs for each model
    patience: int
        Number of epochs to wait for improvement in validation loss before early stopping
    """
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    # Lists to store results from outer cross-validation
    outer_auc_scores = []
    outer_acc_scores = []
    outer_f1_scores = []

    # For final confusion matrix
    event_ids = np.array(event_ids) # Ensure event_ids is a numpy array for indexing
    all_event_ids = []
    all_y_true = []
    all_y_pred = []

    # To store training curves for each outer fold
    all_train_losses = []
    all_val_losses = []

    # To store best models and hyperparameters
    best_model_list = []
    best_params_list = []
    
    # Outer CV
    for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X, y)):
        print(f"Outer fold {outer_fold+1}/{outer_folds}")

        # Split data into training and test sets for the outer fold
        X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
        y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

        # Create DataLoaders for the outer fold
        train_dataset_outer = ReactorDataset(X_train_outer, y_train_outer)
        test_dataset_outer = ReactorDataset(X_test_outer, y_test_outer)

        train_loader_outer = DataLoader(train_dataset_outer, batch_size=8, shuffle=True)
        test_loader_outer = DataLoader(test_dataset_outer, batch_size=8)

        # Inner CV
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42) 

        best_auc = -np.inf # Initialize best AUC to negative infinity for comparison
        best_params = None # To store the best hyperparameters from inner CV

        for hidden_size in param_grid["hidden_size"]:
            for learning_rate in param_grid["lr"]:

                inner_auc_scores = []

                for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
                    print(f"  Inner fold {inner_fold+1}/{inner_folds} (hidden_size={hidden_size}, lr={learning_rate })")
    
                    # Split data into training and validation sets for the inner fold
                    X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
                    y_train_inner, y_val_inner = y_train_outer[train_inner_idx], y_train_outer[val_inner_idx]

                    # Create DataLoaders for the inner fold
                    train_dataset_inner = ReactorDataset(X_train_inner, y_train_inner)
                    val_dataset_inner = ReactorDataset(X_val_inner, y_val_inner)

                    train_loader_inner = DataLoader(train_dataset_inner, batch_size=8, shuffle=True)
                    val_loader_inner = DataLoader(val_dataset_inner, batch_size=8)

                    # Reinitialise model
                    set_seed(42 + outer_fold + inner_fold)
                    
                    model = model_class(input_size=X.shape[2], hidden_size=hidden_size)

                    # Train
                    train_DL_model(model, train_loader_inner, val_loader_inner, epochs=epochs, lr=learning_rate, patience=patience) 

                    # Evaluate
                    val_auc = evaluate_auc_DL(model, val_loader_inner)

                    if np.isnan(val_auc): # Handle NaN AUC (e.g., if only one class is present in the validation fold)
                        print("    Warning: AUC is NaN for this fold. This can happen if the validation fold contains only one class. Skipping this fold.")
                        val_auc = 0

                    inner_auc_scores.append(val_auc)

                # Mean inner cv performance
                mean_inner_auc = np.mean(inner_auc_scores)
                print(f"  Mean inner AUC for hidden_size={hidden_size}, lr={learning_rate}: {mean_inner_auc:.4f}")

                # Track best hyperparameters
                if mean_inner_auc > best_auc:
                    best_auc = mean_inner_auc
                    best_params = {"hidden_size": hidden_size, "lr": learning_rate}

        # Retrain using best hyperparameters 
        print(f"Best hyperparameters for outer fold {outer_fold+1}: {best_params} with mean inner AUC: {best_auc:.4f}")

        # Reinitialise
        set_seed(100 + outer_fold)

        outer_model = model_class(input_size=X.shape[2], hidden_size=best_params["hidden_size"])

        # Train on the entire outer training set with the best hyperparameters
        train_losses, val_losses = train_DL_model(outer_model, train_loader_outer, test_loader_outer, epochs=epochs, lr=best_params["lr"], patience=patience)

        # Evaluation
        # train_auc = evaluate_auc(outer_model, train_loader_outer) ## hmm
        test_auc = evaluate_auc_DL(outer_model, test_loader_outer)
        test_acc = evaluate_accuracy_DL(outer_model, test_loader_outer)
        test_f1 = evaluate_f1_DL(outer_model, test_loader_outer)

        # Warnings
        if np.isnan(test_auc):
            print("  Warning: Test AUC is NaN for this fold. This can happen if the test fold contains only one class. Setting AUC to 0.")
            test_auc = 0
        
        if np.isnan(test_acc):
            print("  Warning: Test accuracy is NaN for this fold. This can happen if the test fold contains only one class. Setting accuracy to 0.")
            test_acc = 0

        if np.isnan(test_f1):
            print("  Warning: Test F1 score is NaN for this fold. This can happen if the test fold contains only one class. Setting F1 score to 0.")
            test_f1 = 0

        # Append results 
        outer_auc_scores.append(test_auc)
        outer_acc_scores.append(test_acc)
        outer_f1_scores.append(test_f1)

        # Confusion matrix
        print(f"Event_ids.shape: {len(event_ids)}, test_outer_idx.shape: {len(test_outer_idx)}") # Debugging shapes
        print(f"Event_ids.type: {type(event_ids)}, test_outer_idx.type: {type(test_outer_idx)}") # Debugging types
        all_event_ids.extend(event_ids[test_outer_idx]) # Store event IDs for the test fold

        y_probs = model_predict_proba(outer_model, test_loader_outer)
        y_pred = (y_probs > 0.5).astype(int)

        all_y_true.extend(y_test_outer)
        all_y_pred.extend(y_pred) # Get binary predictions for confusion matrix

        # Store training curves
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Store best models and best parameters
        best_model_list.append(outer_model)
        best_params_list.append(best_params)

    return outer_auc_scores, outer_acc_scores, outer_f1_scores, all_event_ids, all_y_true, all_y_pred, all_train_losses, all_val_losses, best_model_list, best_params_list


def retrain_with_5_fold_cv_DL(X, y, model_class, param_grid, epochs=150, patience=20):
    """
    Retrain the model on the entire dataset using 5-fold cross-validation to find the best hyperparameters.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    model_class : class
        The model class to instantiate (must have .train, .predict, .predict_proba)
    param_grid : dict
        Hyperparameter grid for grid search
    epochs: int
        Number of training epochs for each model
    patience: int
        Number of epochs to wait for improvement in validation loss before early stopping

    Returns
    -------
    best_model : object
        The best model found during cross-validation
    best_params : dict
        The best hyperparameters found during cross-validation
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_auc = -np.inf # Initialize best AUC to negative infinity for comparison
    best_params = None # To store the best hyperparameters from CV

    print("Starting 5-fold cross-validation for hyperparameter tuning...")

    for hidden_size in param_grid["hidden_size"]:
        for learning_rate in param_grid["lr"]:

            auc_scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                print(f"Fold {fold+1}/5 (hidden_size={hidden_size}, lr={learning_rate})")

                # Split data into training and validation sets for the fold
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Create DataLoaders for the fold
                train_dataset = ReactorDataset(X_train, y_train)
                val_dataset = ReactorDataset(X_val, y_val)

                train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=8)

                # Reinitialise model
                set_seed(42 + fold)
                
                model = model_class(input_size=X.shape[2], hidden_size=hidden_size)

                # Train
                train_DL_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate, patience=patience) 

                # Evaluate
                val_auc = evaluate_auc_DL(model, val_loader)

                if np.isnan(val_auc): # Handle NaN AUC (e.g., if only one class is present in the validation fold)
                    print("  Warning: AUC is NaN for this fold. This can happen if the validation fold contains only one class. Skipping this fold.")
                    val_auc = 0

                auc_scores.append(val_auc)

            # Calculate mean AUC for the current hyperparameter combination
            mean_auc = np.mean(auc_scores)

            # Update best model and parameters if current combination is better
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = {"hidden_size": hidden_size, "lr": learning_rate}

    # Retrain the best model on the entire dataset
    set_seed(42)
    best_model = model_class(input_size=X.shape[2], hidden_size=best_params["hidden_size"])
    train_DL_model(best_model, train_loader, val_loader, epochs=epochs, lr=best_params["lr"], patience=patience)

    return best_model, best_params