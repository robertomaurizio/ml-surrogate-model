# src/eval.py - evaluation of models

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def visualize_tpe_results(trials):
  # Calculate mean and standard deviation of fold losses for each trial
  mean_losses = [np.mean(res['fold_losses']) for res in trials.results]
  std_losses = [np.std(res['fold_losses']) for res in trials.results]

  # Calculate the best (lowest) mean loss and its corresponding std dev at each iteration
  best_losses_at_iteration = []
  best_std_dev_at_iteration = []
  current_best_loss = float('inf')
  current_best_std_dev = 0

  for i in range(len(mean_losses)):
      if mean_losses[i] < current_best_loss:
          current_best_loss = mean_losses[i]
          current_best_std_dev = std_losses[i]
      best_losses_at_iteration.append(current_best_loss)
      best_std_dev_at_iteration.append(current_best_std_dev)

  # Plot the best validation loss at each iteration with its standard deviation
  plt.figure(figsize=(4, 3))
  plt.plot(range(len(best_losses_at_iteration)), best_losses_at_iteration)
  plt.title('Best Mean 5-Fold Cross-Validation Loss')
  plt.xlabel('Trial Number')
  plt.ylabel('Validation Loss')
  plt.grid(True)
  plt.yscale('log')
  plt.tight_layout()
  plt.show()

  # Get the validation losses and sort them in descending order
  sorted_losses = sorted(trials.losses(), reverse=True)

  # Plot
  plt.figure(figsize=(4, 3))
  plt.plot(sorted_losses)
  plt.title('TPE Trial Validation Losses (Sorted from Worst to Best)')
  plt.xlabel('Trial number (sorted)')
  plt.ylabel('Validation Loss')
  plt.grid(True)
  plt.yscale('log')
  plt.show()

  # Get all trials and sort them by loss
  sorted_trials = sorted(trials.results, key=lambda x: x['loss'])

  print("Hyperparameters for the 20 lowest losses:")
  for i, trial in enumerate(sorted_trials[:20]):
      # Access hyperparameters directly from the 'hyperparameters' key added to the trial result
      hp_values = trial.get('hyperparameters', None)
      loss = trial['loss']
      print(f"\nTrial {i+1} (Loss: {loss:.6f}):")
      print(f"  num_layers: {hp_values['num_layers']}")
      print(f"  units: {hp_values['units']}")
      print(f"  activation: {hp_values['activation']}")
      print(f"  learning_rate: {hp_values['learning_rate']}")
      print(f"  batch_size: {hp_values['batch_size']}")
      print(f"  use_batch_norm: {hp_values['use_batch_norm']}")

def evaluate_single_model(final_model, x_test_scaled, y_test, scaler_y):
    # Calculate predictions on test data
    y_predicted_scaled = final_model.predict(x_test_scaled)

    # Inverse transform the scaled predictions back to the log-transformed scale
    y_predicted = scaler_y.inverse_transform(y_predicted_scaled)

    # Apply exponential transformation to temperature-related columns (indices 1, 3, 4)
    # This reverses the log transformation done earlier
    log_transform_cols = [1, 3, 4]
    y_predicted[:, log_transform_cols] = np.exp(y_predicted[:, log_transform_cols])

    # Calculate Mean Square Error (MSE) on the original scale
    mse_test_nn = mean_squared_error(y_test, y_predicted)

    # Calculate R-squared for all output quantities on the original scale
    r2_test_nn = r2_score(y_test, y_predicted)

    print(f"Neural Network on Test Data - Mean Squared Error (MSE) on Original Scale: {mse_test_nn:.4f}")
    print(f"Neural Network on Test Data - R-squared (R2) Score: {r2_test_nn:.4f}")

    return (mse_test_nn, r2_test_nn, y_predicted)

def evaluate_multiple_seeds(final_model_list, x_test_scaled, y_test, scaler_y):
  # Initialize lists for MSE and R2 scores from 100 independent network trainings
  mse_list, r2_list = [], []

  # Evaluate each model and store the MSE and R2 scores
  for i in range(len(final_model_list)):
    a, b, c = evaluate_single_model(final_model_list[i], x_test_scaled, y_test, scaler_y)
    mse_list.append(a)
    r2_list.append(b)

  # Calculate mean and std of MSE and R² across 100 independent network trainings
  mean_mse = np.mean(mse_list)
  std_mse = np.std(mse_list)
  mean_r2 = np.mean(r2_list)
  std_r2 = np.std(r2_list)

  # Plot distribution of MSE and R² across 100 independent network trainings
  fig, axs = plt.subplots(1, 2, figsize=(8, 4))

  n_mse, bins_mse, patches_mse = axs[0].hist(mse_list, bins=20, alpha=0.5, color='blue', label='MSE Distribution')
  axs[0].set_xlabel('MSE')
  axs[0].set_title('Distribution of Neural Network MSE')
  axs[0].grid(True)
  axs[0].axvline(mean_mse, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_mse:.2f}')
  axs[0].fill_betweenx([0, np.max(n_mse)], mean_mse - std_mse, mean_mse + std_mse, color='blue', alpha=0.1, label=f'\u00B1 Std Dev: {std_mse:.2f}')
  axs[0].axvline(1182.56, color='red', linestyle='dashed', linewidth=2, label='Linear Model: 1182.56')

  textstr_mse = f'NN: {mean_mse:.2f}$\pm${std_mse:.2f} \nLinear Model: 1182.56'
  props = dict(boxstyle='round', facecolor='wheat', alpha=1)
  axs[0].text(0.50, 0.95, textstr_mse, transform=axs[0].transAxes, fontsize=10, verticalalignment='top', bbox=props)

  n_r2, bins_r2, patches_r2 = axs[1].hist(r2_list, bins=20, alpha=0.5, color='blue', label='R2 Distribution')
  axs[1].set_xlabel('$R^2$')
  axs[1].set_title('Distribution of Neural Network $R^2$')
  axs[1].grid(True)
  axs[1].axvline(mean_r2, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_r2:.3f}')
  axs[1].fill_betweenx([0, np.max(n_r2)], mean_r2 - std_r2, mean_r2 + std_r2, color='blue', alpha=0.1, label=f'\u00B1 Std Dev: {std_r2:.3f}')
  axs[1].axvline(0.93, color='red', linestyle='dashed', linewidth=2, label='Linear Model: 0.93')

  textstr_r2 = f'NN: {mean_r2:.3f}$\pm${std_r2:.3f} \nLinear Model: 0.93'
  props = dict(boxstyle='round', facecolor='wheat', alpha=1)
  axs[1].text(0.05, 0.95, textstr_r2, transform=axs[1].transAxes, fontsize=10, verticalalignment='top', bbox=props)

  plt.tight_layout()
  plt.show()
  return mse_list, r2_list
  
def plot_single_network_predictions(mse_list, r2_list, final_model_list, final_history_list, x_test_scaled, y_test, scaler_y):
  # Choose a representative network from the ensemble
  # with performance metrics close to the ensemble mean
  ichoice = 3
  print(mse_list[ichoice])
  print(r2_list[ichoice])
  final_model = final_model_list[ichoice]
  final_history = final_history_list[ichoice]
  final_model.summary()

  if not isinstance(final_history, dict):
    final_history = final_history.history

  # Plot training/validation loss vs epoch
  plt.figure(figsize=(4, 3))
  plt.plot(final_history['loss'], label='Training Loss')
  plt.plot(final_history['val_loss'], label='Validation Loss')
  plt.title(f'Neural Network with MSE = {mse_list[ichoice]:.2f}, $R^2$ = {r2_list[ichoice]:.3f}')
  plt.xlabel('Epoch', fontsize=12)
  plt.ylabel('Loss', fontsize=12)
  plt.legend()
  plt.yscale('log')
  plt.grid(True)
  plt.show()

  # Call the function with the final_model
  mse_test_nn, r2_test_nn, y_predicted = evaluate_single_model(final_model, x_test_scaled, y_test, scaler_y)

  # Plot predictions vs actual values
  fig, axs = plt.subplots(2,3, figsize=(12, 8))
  fig.suptitle(f'Neural Network with MSE = {mse_list[ichoice]:.2f}, $R^2$ = {r2_list[ichoice]:.3f}',fontsize=18)
  axs = axs.flatten()
  target_names = ['$n_{e,edge}$', '$T_{e,edge}$', '$n_{e,sep}$', '$T_{e,sep}$', '$T_{e,t}$']
  r2_score_individual = np.zeros(y_test.shape[1])
  mse_individual = np.zeros(y_test.shape[1])
  relative_rmse_individual = np.zeros(y_test.shape[1])

  for i in range(y_test.shape[1]):
      # Calculate R-squared and MSE for the individual output
      r2_score_individual[i] = r2_score(y_test.iloc[:, i], y_predicted[:, i])
      mse_individual[i] = mean_squared_error(y_test.iloc[:, i], y_predicted[:, i])
      relative_rmse_individual[i] = np.sqrt(mse_individual[i]) / np.mean(y_test.iloc[:, i])

      axs[i].scatter(y_test.iloc[:, i], y_predicted[:, i], s=60, color='blue', alpha=0.7, label = target_names[i])
      min_val = min(y_test.iloc[:, i].min(), y_predicted[:, i].min())
      max_val = max(y_test.iloc[:, i].max(), y_predicted[:, i].max())
      axs[i].plot([min_val, max_val], [min_val, max_val], 'k--')
      axs[i].set_xlabel("Actual", fontsize=15)
      axs[i].set_ylabel("Predicted", fontsize=15)
      axs[i].grid(True)

      # Box with title and with R^2 and MSE
      textstr = f'{target_names[i]}\n$R^2 = {r2_score_individual[i]:.2f}$\nMSE = {mse_individual[i]:.3f}'
      props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
      axs[i].text(0.05, 0.95, textstr, transform=axs[i].transAxes, fontsize=16,
              verticalalignment='top', bbox=props)
      
      # Box with relative RMSE
      textstr = f'Relative RMSE: \n ~ {100*relative_rmse_individual[i]:.1f}%'
      props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
      axs[i].text(0.4, 0.2, textstr, transform=axs[i].transAxes, fontsize=16,
              verticalalignment='top', bbox=props)

  # Box with R^2 and MSE - total
  textstr = f'Total:\n$R^2 = {r2_test_nn:.3f}$\nMSE = {mse_test_nn:.2f}'
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  axs[5].text(0.25, 0.6, textstr,  fontsize=16,
          verticalalignment='top', bbox=props)
  axs[5].set_xticks([])
  axs[5].set_yticks([])

  plt.tight_layout()
  plt.show()

def plot_linear_model_predictions(y_test, y_predicted_test_linear, r2_test_linear, mse_test_linear):
  # Plot predictions vs actual values
  # Plot
  fig, axs = plt.subplots(2,3, figsize=(12, 8))
  fig.suptitle(f'Linear regression model',fontsize=18)

  axs = axs.flatten()
  target_names = ['$n_{e,edge}$', '$T_{e,edge}$', '$n_{e,sep}$', '$T_{e,sep}$', '$T_{e,t}$']
  r2_score_individual_linear = np.zeros(y_test.shape[1])
  mse_individual_linear = np.zeros(y_test.shape[1])
  relative_rmse_individual_linear = np.zeros(y_test.shape[1])

  for i in range(y_test.shape[1]):
      # Calculate R-squared and MSE for the individual output
      r2_score_individual_linear[i] = r2_score(y_test.iloc[:, i], y_predicted_test_linear[:, i])
      mse_individual_linear[i] = mean_squared_error(y_test.iloc[:, i], y_predicted_test_linear[:, i])
      relative_rmse_individual_linear[i] = np.sqrt(mse_individual_linear[i]) / np.mean(y_test.iloc[:, i])

      axs[i].scatter(y_test.iloc[:, i], y_predicted_test_linear[:, i], s=60, color='blue', alpha=0.7, label = target_names[i])
      # Adjust the ideal fit line based on the min and max values of the current subplot's data
      min_val = min(y_test.iloc[:, i].min(), y_predicted_test_linear[:, i].min())
      max_val = max(y_test.iloc[:, i].max(), y_predicted_test_linear[:, i].max())
      axs[i].plot([min_val, max_val], [min_val, max_val], 'k--')
      axs[i].set_xlabel("Actual", fontsize=15)
      axs[i].set_ylabel("Predicted", fontsize=15)
      axs[i].grid(True)

      # Box with title and with R^2 and MSE
      textstr = f'{target_names[i]}\n$R^2 = {r2_score_individual_linear[i]:.2f}$\nMSE = {mse_individual_linear[i]:.3f}'
      props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
      axs[i].text(0.05, 0.95, textstr, transform=axs[i].transAxes, fontsize=16,
              verticalalignment='top', bbox=props)
      
      # Box with relative RMSE
      textstr = f'Relative RMSE: \n ~ {100*relative_rmse_individual_linear[i]:.1f}%'
      props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
      axs[i].text(0.4, 0.2, textstr, transform=axs[i].transAxes, fontsize=16,
              verticalalignment='top', bbox=props)

  # Box with R^2 and MSE - total
  textstr = f'Total:\n$R^2 = {r2_test_linear:.3f}$\nMSE = {mse_test_linear:.2f}'
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  axs[5].text(0.25, 0.6, textstr,  fontsize=16,
          verticalalignment='top', bbox=props)
  axs[5].set_xticks([])
  axs[5].set_yticks([])

  plt.tight_layout()
  plt.show()
