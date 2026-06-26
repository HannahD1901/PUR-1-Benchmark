import torch
import torch.nn as nn

# Logistic regression class
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
class LogisticRegression:
    def __init__(self, **params):
        self.model = SklearnLogisticRegression(**params)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]  # Probability of the positive class


# Decision tree class
from sklearn.tree import DecisionTreeClassifier
class DecisionTree:
    def __init__(self, **params):
        self.model = DecisionTreeClassifier(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]  # Probability of the positive class


# Random forest class 
from sklearn.ensemble import RandomForestClassifier
class RandomForest:
    def __init__(self, **params):
        default_params = {
            "n_estimators": 200,
            "max_depth": 5
        }

        # Override defaults if parameters are provided
        default_params.update(params)

        self.model = RandomForestClassifier(**default_params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]  # Probability of the positive class


# XGBoost class
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBoost(BaseEstimator, ClassifierMixin): # Need BaseEstimator and ClassifierMixins for proper wrapping
    # _estimator_type = "classifier" # Hardcode this to ensure it is recognised as a classifier by scikit-learn
    
    def __init__(self, **params):
        default_params = {
            "max_depth": 3,
            "n_estimators": 200,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 10,
            "reg_alpha": 1,
            "eval_metric": "logloss",
            "random_state": 42
        }

        # Override defaults if parameters are provided
        default_params.update(params)
        self.params = default_params

        self.model = XGBClassifier(**default_params)
    
    def fit(self, X_train, y_train): # For cv
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        return self

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]  # Probability of the positive class

    def get_params(self, deep=True): # For cv
        return self.params

    def set_params(self, **params): # For cv
        self.params.update(params)
        self.model = XGBClassifier(**self.params)
        return self
    

# LSTM class
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=4, num_layers=1):
        super().__init__() # Call the parent class constructor to initialize the nn.Module

        self.lstm = nn.LSTM(
            input_size=input_size,   # 10 features (or fewer if feature selection is applied)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1) # Output layer for binary classification (takes in hidden state and outputs a single value for binary classification)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation to convert output to probability between 0 and 1


    # forward() runs the data through LSTM + Fully Connected layer.
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)

        # Mean pooling af all timesteps 
        pooled = torch.mean(lstm_out, dim=1) # (batch, hidden_size)
        out = self.fc(pooled)
        out = self.sigmoid(out)
        return out.squeeze(-1) # Squeeze to get shape (batch,) for binary classification 