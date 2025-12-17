# run_train.py - runner script to train the neural network(s)
from src.data import *
from src.model import *
from src.train import *

def main():
  # Load input features and corresponding output targets (simulation results)
  df, df_excluded = load_sobol_solps_samples()

  # Preprocess data: normalize, rescale, log-transform
  x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, x_temp, y_temp, scaler_y, x_train, y_train, x_test, y_test = \
    preprocess_data(df, df_excluded)

  # Define hyperparameter search space for TPE optimization
  num_layers_choices, units_choices, activation_choices, \
    learning_rate_choices, batch_size_choices, use_batch_norm_choices, space = define_hyperparameter_space()

  # Run TPE optimization with 5-fold cross-validation
  best_hyperparameters, trials = run_tpe_optimization(space, x_temp, y_temp)

  # Save TPE results
  save_TPE_results(trials)

  # Train neural network using multiple random seeds
  final_model_list, final_history_list = (
    train_multiple_seeds(best_hyperparameters, num_layers_choices, units_choices, activation_choices, learning_rate_choices, batch_size_choices, 
                         use_batch_norm_choices, x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, N_models = 100)
    )
  
  # Save neutral network (multiple seeds) results
  save_model_history_list(final_model_list, final_history_list)

if __name__ == "__main__":
    main()
