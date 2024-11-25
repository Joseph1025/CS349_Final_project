"""
Given a price dataset of snaekers, we will use regression to
predict the price of a sneaker given the number of sales.
Since there are multiple features, we will use multiple linear regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from spacy.cli.train import train


class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def scikit_regression(train_data, test_data, valid_data):
    """
    Use scikit-learn to perform regression on the data.
    """
    # One-hot encoding for categorical features
    train_data = pd.get_dummies(train_data, columns=['Buyer Region', 'Sneaker Name'], drop_first=True)
    test_data = pd.get_dummies(test_data, columns=['Buyer Region', 'Sneaker Name'], drop_first=True)
    valid_data = pd.get_dummies(valid_data, columns=['Buyer Region', 'Sneaker Name'], drop_first=True)

    # Features selected:
    # Retail Price, Shoe Size, Days Since Release, Buyer Region, Sneaker Name
    Feature_train = train_data[['Retail Price', 'Shoe Size', 'Days Since Release'] + list(
        train_data.columns[train_data.columns.str.startswith('Buyer Region_') | train_data.columns.str.startswith('Sneaker Name_')])]
    target_train = train_data['Sale Price']
    Feature_test = test_data[['Retail Price', 'Shoe Size', 'Days Since Release'] + list(
        test_data.columns[test_data.columns.str.startswith('Buyer Region_') | test_data.columns.str.startswith('Sneaker Name_')])]
    target_test = test_data['Sale Price']
    Feature_valid = valid_data[['Retail Price', 'Shoe Size', 'Days Since Release'] + list(
        valid_data.columns[valid_data.columns.str.startswith('Buyer Region_') | valid_data.columns.str.startswith('Sneaker Name_')])]
    target_valid = valid_data['Sale Price']


    # Train the model
    model = Ridge(alpha=1.0)
    model.fit(Feature_train, target_train)

    # Make predictions
    train_pred = model.predict(Feature_train)
    test_pred = model.predict(Feature_test)
    valid_pred = model.predict(Feature_valid)

    # Calculate the RMSE
    train_rmse = np.sqrt(mean_squared_error(target_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(target_test, test_pred))
    val_rmse = np.sqrt(mean_squared_error(target_valid, valid_pred))

    # Calculate the MAE
    train_mae = np.mean(np.abs(target_train - train_pred))
    test_mae = np.mean(np.abs(target_test - test_pred))
    val_mae = np.mean(np.abs(target_valid - valid_pred))

    print(f"Train RMSE: {train_rmse:.2f}, Validation RMSE: {val_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
    print(f"Train MAE: {train_mae:.2f}, Validation MAE: {val_mae:.2f}, Test MAE: {test_mae:.2f}")

    # Plot precision (MAE) for training, validation, and test sets
    labels = ['Train', 'Validation', 'Test']
    mae_values = [train_mae, val_mae, test_mae]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, mae_values, alpha=0.7)
    plt.title('Mean Absolute Error (MAE) Across Train, Validation, and Test Sets')
    plt.xlabel('Dataset')
    plt.ylabel('MAE')
    plt.show()



