# src/model.py - for model construction
import random
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import hp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hyperopt import STATUS_OK

def build_and_train_model(hp, x_train, y_train, x_val, y_val, verbose = 0, max_epochs = 300, max_patience = 30, seed_choice = 42):
    """
    This function builds, compiles, and trains a neural network with given hyperparameters and data.

    Args:
        hp: A dictionary containing hyperparameter values
        x_train, y_train: Training data.
        x_val, y_val: Validation data.

    Returns:
        The trained Keras model and the training history.
    """

    # Set random seed for reproducibility
    os.environ['PYTHONHASHSEED']=str(seed_choice)
    random.seed(seed_choice)
    np.random.seed(seed_choice)      # Numpy
    tf.random.set_seed(seed_choice)  # TensorFlow

    # Define the model with hyperparameters
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=(x_train.shape[1],)))

    # Hidden layers
    for _ in range(hp['num_layers']):
        model.add(keras.layers.Dense(units=hp['units'], activation=hp['activation']))
        if hp.get('use_batch_norm', False):
            model.add(keras.layers.BatchNormalization())

    # Output layer with linear activation (since targets are scaled)
    model.add(keras.layers.Dense(units=y_train.shape[1], activation='linear'))

    # Choose optimizer, then compile module
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Define Early Stopping callback (monitor validation loss)
    early_stopping = EarlyStopping(monitor='val_loss', patience=max_patience, restore_best_weights=True)

    # Train the model with hyperparameters
    history = model.fit(x_train, y_train,
                        epochs=max_epochs,       # Fixed number of epochs
                        batch_size=hp['batch_size'],
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping],
                        shuffle=False,
                        verbose=verbose)

    return model, history

def define_hyperparameter_space():
  # Define the original choices of hyperparameter values
  num_layers_choices = [1, 2]
  units_choices = [16, 32, 64]
  activation_choices = ['relu', 'tanh']
  learning_rate_choices = [0.0001, 0.001, 0.01]
  batch_size_choices = [16, 32, 64]
  use_batch_norm_choices = [True, False]

  # Define the hyperparameter space
  space = {
      'num_layers': hp.choice('num_layers', num_layers_choices),            # Number of layers (depth)
      'units': hp.choice('units', units_choices),                           # Number of units (width, same for all layers)
      'activation': hp.choice('activation', activation_choices),            # Activation functions
      'learning_rate': hp.choice('learning_rate', learning_rate_choices),   # Learning rate
      'batch_size': hp.choice('batch_size', batch_size_choices),            # Mini-batch size
      'use_batch_norm': hp.choice('use_batch_norm', use_batch_norm_choices),# Batch normalization (regularizer)
      }
  return num_layers_choices, units_choices, activation_choices, learning_rate_choices, batch_size_choices, use_batch_norm_choices, space

def objective(hyperparameters, x_combined, y_combined):
    """
    Objective function for hyperopt to minimize, using 5-Fold Cross-Validation.

    Args:
        hyperparameters: A dictionary containing the hyperparameters to evaluate.
        x_combined: Combined features for cross-validation.
        y_combined: Combined targets for cross-validation.

    Returns:
        A dictionary for hyperopt, including the mean validation loss, status,
        fold-wise validation losses, and the evaluated hyperparameters.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_validation_losses = []

    print(f"Evaluating hyperparameters: {hyperparameters}")

    for fold, (train_index, val_index) in enumerate(kf.split(x_combined)):
        # Split data for the current fold
        x_train_fold, x_val_fold = x_combined.iloc[train_index], x_combined.iloc[val_index]
        y_train_fold, y_val_fold = y_combined.iloc[train_index], y_combined.iloc[val_index]

        # Standardize features (x) - fit only on training data for the fold
        scaler_x_fold = StandardScaler()
        x_train_scaled_fold = scaler_x_fold.fit_transform(x_train_fold)
        x_val_scaled_fold = scaler_x_fold.transform(x_val_fold)

        # Log-transform temperature targets (y) for the fold
        y_train_log_fold, y_val_log_fold = y_train_fold.copy(), y_val_fold.copy()
        log_transform_cols = [1, 3, 4] # Indices for tecore, teompsep, tetpeak
        for col_index in log_transform_cols:
            y_train_log_fold.iloc[:, col_index] = np.log(y_train_log_fold.iloc[:, col_index])
            y_val_log_fold.iloc[:, col_index] = np.log(y_val_log_fold.iloc[:, col_index])

        # Standardize targets (y) - fit only on log-transformed training data for the fold
        scaler_y_fold = StandardScaler()
        y_train_scaled_fold = scaler_y_fold.fit_transform(y_train_log_fold)
        y_val_scaled_fold = scaler_y_fold.transform(y_val_log_fold)

        # Build and train the model for the current fold
        model_fold, history_fold = build_and_train_model(
            hyperparameters, x_train_scaled_fold, y_train_scaled_fold, x_val_scaled_fold, y_val_scaled_fold, verbose=0)

        # Store the validation loss from the current fold
        current_fold_val_loss = history_fold.history['val_loss'][-1]
        fold_validation_losses.append(current_fold_val_loss)
        print(f"  Fold {fold + 1} completed. Validation Loss: {current_fold_val_loss:.6f}")

    mean_loss = np.mean(fold_validation_losses)
    print(f"Mean validation loss for current hyperparameters: {mean_loss:.6f}\n")

    # Return a dictionary for hyperopt, including the hyperparameters themselves
    return {'loss': mean_loss, 'status': STATUS_OK, 'fold_losses': fold_validation_losses, 'hyperparameters': hyperparameters}
  
def fit_linear_model(x_train, y_train, x_test, y_test):
  # Fit the linear regression model to the training data
  linear_model = LinearRegression()
  linear_model.fit(np.log(x_train), np.log(y_train))

  # Make predictions on the test data
  y_predicted_test_linear = np.exp(linear_model.predict(np.log(x_test)))

  # Calculate performance metrics (MSE and R-squared)
  mse_test_linear = mean_squared_error(y_test, y_predicted_test_linear)
  r2_test_linear = r2_score(y_test, y_predicted_test_linear)

  # Print the calculated metrics
  print(f"Linear Regression on Test Data - Mean Squared Error (MSE): {mse_test_linear:.4f}")
  print(f"Linear Regression on Test Data - R-squared (R2) Score: {r2_test_linear:.4f}")

  # Print the coefficients and the intercept of the fitted linear regression model
  print("Linear regression model coefficients:")
  print(linear_model.coef_)
  print("\nLinear regression model intercept:")
  print(linear_model.intercept_)

  return y_predicted_test_linear, mse_test_linear, r2_test_linear
