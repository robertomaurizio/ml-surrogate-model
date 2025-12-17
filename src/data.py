# src/data.py -- for data creation/loading/preprocessing

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import pandas as pd
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from datetime import datetime

def generate_sobol_samples():
  # Generate Sobol sequence of varied input parameters

  # Define bounds
  x1_min, x1_max = 2, 20          # Power injected into the tokamak
  x2_min, x2_max = 10**20, 10**22 # Neutral gas flux into the tokamak
  x3_min, x3_max = 0.1, 2         # Anomalous particle diffusivity
  x4_min, x4_max = 0.3, 6         # Anomalous thermal diffusivity

  # Generate Sobol samples
  sobol = qmc.Sobol(d=4, scramble=True)
  n_samples = 1024
  samples = sobol.random(n_samples)

  # Scale samples to target rectangle
  x1 = x1_min + (x1_max - x1_min) * samples[:, 0]
  x2 = x2_min + (x2_max - x2_min) * samples[:, 1]
  x3 = x3_min + (x3_max - x3_min) * samples[:, 2]
  x4 = x4_min + (x4_max - x4_min) * samples[:, 3]

  # Plot
  plt.figure(figsize=(4, 4))
  plt.scatter(x1, x2/1e22, s=10, color='blue', alpha=0.7)
  plt.xlim(x1_min - 1, x1_max + 1)
  plt.ylim(float(x2_min/1e22) - 0.1, float(x2_max/1e22) + 0.1)
  plt.title(f"{n_samples} Sobol samples")
  plt.xlabel("Input Power (MW)")
  plt.ylabel("Neutral gas (10^22 D/s)")
  plt.grid(True)
  plt.show()
  return x1, x2, x3, x4

def load_sobol_solps_samples():
  
  # Mount disk with data
  drive.mount('/content/drive')

  # Inport input parameters generated using SOBOL sequence
  sb = pd.read_csv('/content/drive/My Drive/Colab Notebooks/sobol_1.txt', header=None, names=['power','flux','D','Chi'])

  # Set index to start at 1
  sb.index = sb.index + 1
  sb.head()

  # Import data using Pandas DataFrame
  sm = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data_1.txt')
  print(sm.shape)

  # Remove leading spacing in the column names
  sm.columns = sm.columns.str.strip()

  # Set idx as index, then sort
  sm = sm.set_index('idx')
  sm = sm.sort_index()
  sm.head(5)

  # Concatenate with Sobol dataframe
  df_all = pd.concat([sb, sm], axis = 1)
  df_all.head()

  # Select converged runs
  df = df_all[(df_all.convMetric < 0.05) & (df_all.runTime > 0.95) & (df_all.tetpeak < df_all.teompsep)]
  df_excluded = df_all.loc[df_all.index.difference(df.index)]

  # Print the minimum and maximum values for each column in the df DataFrame
  print("Minimum and Maximum values for each column in df:")
  for col in df.columns:
      min_val = df[col].min()
      max_val = df[col].max()
      print(f"{col}: [Min: {min_val:.4f}, Max: {max_val:.4f}]")

  return df, df_excluded

def plot_sobol_solps_samples(df, df_excluded):
  # Plot
  plt.figure(figsize=(4, 4))
  plt.scatter(df.power, df.flux/1e2, s=10, color='blue', alpha=0.3, label = 'Converged runs')
  plt.scatter(df_excluded.power, df_excluded.flux/1e2, s=12, color='red', alpha=1, label = 'Diverged runs')
  plt.xlabel("Input Power (MW)")
  plt.ylabel("Gas flux ($10^{22}$ atoms/s)")
  plt.grid(True)
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.show()

  # Some preliminary plots - global sanity checks
  fig, axs = plt.subplots(1, 2, figsize=(8, 4))
  axs = axs.flatten()

  axs[0].scatter(df.flux, df.necore, label = 'Core', alpha = 0.7)
  axs[0].scatter(df.flux, df.neompsep, label = 'Separatrix', alpha = 0.7)
  axs[0].set_xlabel("Gas flux ($10^{22}$ D/s)")
  axs[0].set_ylabel("($10^{19}m^{-3}$)")
  axs[0].set_title('Plasma density')
  axs[0].legend()
  axs[0].grid(True)

  axs[1].scatter(df.power, df.tecore, label = 'Core', alpha = 0.7)
  axs[1].scatter(df.power, df.teompsep, label = 'Separatrix', alpha = 0.7)
  axs[1].scatter(df.power, df.tetpeak, label = 'Target', alpha = 0.7)
  axs[1].set_xlabel("Input power (MW)")
  axs[1].set_ylabel("(eV)")
  axs[1].set_title('Plasma temperature')
  axs[1].set_yscale('log')
  axs[1].legend()
  axs[1].grid(True)

  plt.tight_layout()
  plt.show()

def preprocess_data(df):
  # Preprocess data: standardize features, scale targets
  x = df[['power', 'flux', 'D', 'Chi']]                           # x shape: [n_data, n_features=4]
  y = df[['necore', 'tecore', 'neompsep', 'teompsep', 'tetpeak']] # y shape: [n_data, n_targets=5]

  # Split off 15% for test set
  x_temp, x_test, y_temp, y_test = train_test_split(
      x, y, test_size=0.15, random_state=42)

  # Split remaining 85% into 70% train and 15% val
  # 15% of total = ~17.6% of remaining
  x_train, x_val, y_train, y_val = train_test_split(
      x_temp, y_temp, test_size=0.1765, random_state=42)

  # Print the range for each feature in x
  print("Range for each feature in x:")
  for col in x.columns:
      min_val = x[col].min()
      max_val = x[col].max()
      print(f"{col}: [{min_val:.2f}, {max_val:.2f}]")
  
  from sklearn.preprocessing import StandardScaler

  # Standardize the features
  # NB: normalization parameters are computed only from the training set
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_val_scaled = scaler.transform(x_val)
  x_test_scaled = scaler.transform(x_test)

  print("Features standardized.")
  print(f"Shape of x_train_scaled: {x_train_scaled.shape}")
  print(f"Shape of x_val_scaled: {x_val_scaled.shape}")
  print(f"Shape of x_test_scaled: {x_test_scaled.shape}")

  # Log-transform the temperature targets (columns = [1,3,4])
  y_train_log, y_val_log, y_test_log = y_train.copy(), y_val.copy(), y_test.copy()
  log_transform_cols = [1, 3, 4]
  for col_index in log_transform_cols:
      y_train_log.iloc[:, col_index] = np.log(y_train.iloc[:, col_index])
      y_val_log.iloc[:, col_index] = np.log(y_val.iloc[:, col_index])
      y_test_log.iloc[:, col_index] = np.log(y_test.iloc[:, col_index])

  # Scale the output targets using Min-Max normalization
  scaler_y = MinMaxScaler()
  y_train_scaled = scaler_y.fit_transform(y_train_log)
  y_val_scaled = scaler_y.transform(y_val_log)
  y_test_scaled = scaler_y.transform(y_test_log)

  print("Targets scaled")
  print(f"Shape of y_train_scaled: {y_train_scaled.shape}")
  print(f"Shape of y_val_scaled: {y_val_scaled.shape}")
  print(f"Shape of y_test_scaled: {y_test_scaled.shape}")

  return x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, x_temp, y_temp, scaler_y, x_train, y_train, x_test, y_test 

def save_TPE_results(best_hyperparameters, trials):
  # Save best_hyperparameters and trials to file
  # Get today's date
  today_date = datetime.now().strftime("%Y-%m-%d")

  save_directory = '/content/drive/My Drive/Colab Notebooks'
  TPE_save_path = os.path.join(save_directory, f'TPE_{today_date}.pkl')

  data = {'best_hyperparameters': best_hyperparameters,
          'trials': trials}

  # Save best_hyperparameters to a file
  with open(TPE_save_path, 'wb') as f:
      pickle.dump(data, f)

def load_TPE_results():
  # - DO NOT RUN IF JUST DID TPE OPTIMIZATION -
  # It will overwrite best_hyperparameters and trials
  # Load best_hyperparameters and trials from files
  load_directory = '/content/drive/My Drive/Colab Notebooks'
  best_hyperparameters_path = os.path.join(load_directory, 'TPE_2025-11-10.pkl')

  with open(best_hyperparameters_path, 'rb') as f:
      data = pickle.load(f)

  best_hyperparameters = data['best_hyperparameters']
  trials = data['trials']
  return best_hyperparameters, trials

def save_model_history_list(final_model_list, final_history_list):
  # Save LIST of final model and history to file
  # Get today's date
  today_date = datetime.now().strftime("%Y-%m-%d")

  save_directory = '/content/drive/My Drive/Colab Notebooks'
  list_save_path = os.path.join(save_directory, f'NN_model_history_list_{today_date}.keras')

  data = {'final_model_list': final_model_list,
          'final_history_list': final_history_list}

  # Save best_hyperparameters to a file
  with open(list_save_path, 'wb') as f:
      pickle.dump(data, f)

def load_model_history_list():
  # - OPTIONAL -
  # Load list of NN models and corresponding histories
  load_directory = '/content/drive/My Drive/Colab Notebooks'
  list_save_path = os.path.join(load_directory, 'NN_model_history_list_2025-12-16.keras')

  with open(list_save_path, 'rb') as f:
      data = pickle.load(f)

  final_model_list = data['final_model_list']
  final_history_list = data['final_history_list']
  return final_model_list, final_history_list
