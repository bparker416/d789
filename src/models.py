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
    
class ClassificationModels:
    # Classical classification models

    @staticmethod
    def random_forest_classifier(X_train, y_train, X_test, n_estimators=100, max_depth=None):
        """
        Train and predict using Random Forest Classifier

        Args:
            X_train (np.array): Training Features
            y_train (np.array): Training target
            X_test (np.array): Test features
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees

        Returns:
            tuple: (model, predictions, prediction_probabilities)
        """
        print(f"\nTraining Random Forest Classifier (n_estimators={n_estimators})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)
        print("Random Forest Classifier training completed")

        return model, predictions, pred_proba
    
    @staticmethod
    def svm_classifier(X_train, y_train, X_test, kernel='rbf', C=1.0):
        """
        Train and predict using Support Vector Machine Classifier

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_test (np.array): Test features
            kernel (str): Kernel type
            C (float): Regularization parameter

        Returns:
            tuple: (model, predictions, prediction_probabilities)
        """
        print(f"\nTraining SVM Classifier (kernel={kernel}, C={C})...")
        model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)
        print("SVM Classifier training completed")

        return model, predictions, pred_proba
    
class DeepLearningModels:
    # Deep learning models using TensorFlow/Keras

    @staticmethod
    def build_regression_nn(input_dim, hidden_layers=[64, 32], dropout_rate=0.2):
        """
        Build a neural network for regression

        Args:
            input_dim (int): Number of input feratures
            hidden_layers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularizations

        Returns:
            keras.Models: Compiled neural network model
        """        
        model = models.Sequential()

        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(layers.Dropout(dropout_rate))

        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))

        # Output layer for regression
        model.add(layers.Dense(1))

        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return model
    
    @staticmethod
    def train_regression_nn(X_train, y_train, X_test, y_test, 
                            hidden_layers=[64, 32], dropout_rate=0.2,
                            epochs=100, batch_size=32, validation_split=0.2):
        """
        Train a neural network for regression

        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_test (np.array): Test features
            y_test (np.array): Test target
            hidden_layers (list): List of hidden layer sizes. Defaults to [64, 32].
            dropout_rate (float): _Dropout rate. Defaults to 0.2.
            epochs (int): Number of training epochs. Defaults to 100.
            batch_size (int): Batch size for training. Defaults to 32.
            validation_split (float): Validation split ratio. Defaults to 0.2.

        Returns:
            tuple: (model, predictions, history)
        """
        print("\nBuilding Neural Network for Regression...")
        print(f"Architecture: {hidden_layers}")

        # Build model
        model = DeepLearningModels.build_regression_nn(
            input_dim=X_train.shape[1],
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )

        print(model.summary())

        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        print("\nTraining Neural Network...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )

        # Make predictions
        predictions = model.predict(X_test).flatted()

        print("Neural Network training completed")

        return model, predictions, history
    
    @staticmethod
    def build_classification_nn(input_dim, num_classes, hidden_layers=[64, 32], dropout_rate=0.2):
        """
        Build a neural network for classification

        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
            hidden_layers (list, optional): List of hidden layer sizes. Defaults to [64, 32].
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.

        Returns:
            keras.Model: Compiled neural network model.
        """
        model = models.Sequential()

        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(layers.Dropout(dropout_rate))

        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))

        # Output layer for classification
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'

        # Compile model
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def train_classification_nn(X_train, y_train, X_test, y_test,
                                num_classes=3, hidden_layers=[64, 32], dropout_rate=0.2,
                                epochs=100, batch_size=32, validation_split=0.2):
        """
        Train a neural network for classification

        Args:
            X_train (np.array): _description_
            y_train (np.array): _description_
            X_test (np.array): _description_
            y_test (np.array): _description_
            num_classes (int): _description_. Defaults to 3.
            hidden_layers (list): _description_. Defaults to [64, 32].
            dropout_rate (float): _description_. Defaults to 0.2.
            epochs (int): _description_. Defaults to 100.
            batch_size (int): _description_. Defaults to 32.
            validation_split (float): _description_. Defaults to 0.2.

        Returns:
            tuple: (model, predictions, prediction_probabilities, history)
        """
        print("\nBuilding Beural Network for Classification...")
        print(f"Architecture: {hidden_layers}")
        print(f"Number of classes: {num_classes}")

        # Build model
        model = DeepLearningModels.build_classification_nn(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )

        print(model.summary())

        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='cal_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        print("\nTraining Neural Network...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )

        # Make predictions
        pred_probe = model.predict(X_test)

        if num_classes == 2:
            predictions = (pred_proba > 0.5).astype(int).flatten()
            pred_proba = np.column_stack([1 - pred_proba, pred_proba])
        else:
            predictions = np.argmax(pred_proba, axis=1)

        print("Neural Network training completed")

        return model, predictions, pred_proba, history
    
def plot_training_history(history, model_name="Model", save_path=None):
    """
    Plot training history for deep learning models

    Args:
        history (keras.History): Training history object
        model_name (str): Name of the model. Defaults to "Model".
        save_path (str): Path to save the plot. Defaults to None.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training History - [model_name]')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot accuracy or MAE
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_ylabel('Accuracy')
    elif 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_ylabel('MAE')

    axes[1].set_xlabel('Epoch')
    axes[1].set_title(f'Metrics - [model_name]')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot daved to {save_path}")

    plt.show()