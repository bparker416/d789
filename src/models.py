# ML model implementations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow import layers, models
from keras.callbacks import EarlyStopping

class RegressionModels:

    @staticmethod
    def linear_regression(X_train, y_train, X_test):
        """
        Classical regression models

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_test (np.array): Test features

        Returns:
            tuple: (model, predictions)
        """
        print("\nTraining Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Linear Regression training completed")

        return model, predictions
    
    @staticmethod
    def ridge_regression(X_train, y_train, X_test, alpha=1.0):
        """Train and predict using Ridge Regression.

        Args:
            X_trains (np.array): Training features
            y_train (np.array): Training target
            X_test (np.array): Test features
            alpha (float): Regularization strength

        Returns:
            tuple: (model, predictions)
        """
        print(f"\nTraining Ridge Regression (alpha={alpha})...")
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Ridge Regression training completed")

        return model, predictions
    
    @staticmethod
    def random_forest_regressor(X_train, y_train, X_test, n_estimators=100, max_depth=None):
        """
        Train and predict using Random Forest Regressor

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_test (np.array): Test features
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees

        Returns:
            tuple: (model, predictions)
        """
        print(f"\nTraining Random Forest Regressor (n_estimators={n_estimators})...")
        model = RandomForestRegressor(
            n_estimators = n_estimators,
            max_depth = max_depth,
            random_state = 42,
            n_jobs = -1
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Random Forest Refressor training completed")

        return model,  predictions