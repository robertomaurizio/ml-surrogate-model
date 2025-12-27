# src/train.py - for model training (search for best hyperparameters, NN training)

from hyperopt import fmin, tpe, Trials
from functools import partial

def run_tpe_optimization(objective, space, x_temp, y_temp):
  # Create a partial function for the objective function with data
  # NB: must pass original data (not pre-processed, i.e. before standardization/log transformation)
  objective_with_data = partial(objective,
                                x_combined=x_temp, y_combined=y_temp)

  # Run the TPE optimization to find the "best" hyperparameters
  trials = Trials()
  best_hyperparameters = fmin(fn=objective_with_data, # Use the partial function
                              space=space,
                              algo=tpe.suggest, # the TPE algorithm
                              max_evals=60,     # Number of different sets to try
                              trials=trials)    # record all results here

  print("\nBest hyperparameters found:")
  print(best_hyperparameters)
  return best_hyperparameters, trials

def train_multiple_seeds(best_hyperparameters, x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled):
  # Translate the index values from best_hyperparameters to actual values
  best_num_layers = num_layers_choices[best_hyperparameters['num_layers']]
  best_units = units_choices[best_hyperparameters['units']]
  best_activation = activation_choices[best_hyperparameters['activation']]
  best_learning_rate = learning_rate_choices[best_hyperparameters['learning_rate']]
  best_batch_size = batch_size_choices[best_hyperparameters['batch_size']]
  best_use_batch_norm = use_batch_norm_choices[best_hyperparameters['use_batch_norm']]

  # Create a dictionary with the best hyperparameters
  best_hp_dict = {
      'activation': best_activation,
      'batch_size': best_batch_size,
      'learning_rate': best_learning_rate,
      'num_layers': best_num_layers,
      'units': best_units,
      'use_batch_norm': best_use_batch_norm,
  }

  print("Best hyperparameters to be used for final training:")
  for par in best_hp_dict:
      print(f"{par}: {best_hp_dict[par]}")
  
  # Retrain the network 100 times using different random seeds
  N_models = 100

  final_model_list, final_history_list = [], []
  base_seed = 42
  seeds = [base_seed + i for i in range(N_models)]

  for i in range(N_models):
    fmod, fhist = build_and_train_model(best_hp_dict, x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, seed_choice = seeds[i])
    final_model_list.append(fmod)
    final_history_list.append(fhist)
    print(f"Model {i+1} trained.")
  
  return final_model_list, final_history_list
