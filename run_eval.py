# run_eval.py - runner script to visualize results
from src.data import *
from src.model import *
from src.train import *

def main():
  # Load input features and corresponding output targets (simulation results)
  df, df_excluded = load_sobol_solps_samples()

  # Preprocess data: normalize, rescale, log-transform
  x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, \
      x_train, x_val, x_test, y_train, y_val, y_test, \
      x_temp, y_temp, scaler_y = preprocess_data(df)

  # Load TPE results
  best_hyperparameters, trials = load_TPE_results()

  # Visualize TPE results
  visualize_tpe_results(trials)

  # Load multiple seed neural network results
  final_model_list, final_history_list = load_model_history_list()

  # Visualize multiple seeds results (histograms)
  mse_list, r2_list = evaluate_multiple_seeds(final_model_list, x_test_scaled, y_test, scaler_y)

  # Visualize detailed results for single (representative) network
  plot_single_network_predictions(mse_list, r2_list, final_model_list, final_history_list, x_test_scaled, y_test, scaler_y)

  # Run linear regression fit
  y_predicted_test_linear, mse_test_linear, r2_test_linear = fit_linear_model(x_train, y_train, x_test, y_test)

  # Visualize linear regression fit results
  plot_linear_model_predictions(y_test, y_predicted_test_linear, r2_test_linear, mse_test_linear)

if __name__ == "__main__":
    main()
