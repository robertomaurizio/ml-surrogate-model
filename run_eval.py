# run_eval.py - runner script to visualize results

# run_train.py
from src.data import *
from src.model import *
from src.train import *

def main():
  # Load input features and corresponding output targets (simulation results)
  df, df_excluded = load_sobol_solps_samples()

  # Preprocess data: normalize, rescale, log-transform
  x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, x_temp, y_temp, scaler_y, x_train, y_train, x_test, y_test = \
    preprocess_data(df, df_excluded)

  # Load TPE results
  trials = load_TPE_results()

  # Visualize TPE results
  visualize_tpe_results(trials)

  # Load multiple seed neural network results
  final_model_list, final_history_list = load_model_history_list()

  # Visualize multiple seeds results (histograms)
  mean_mse, std_mse, mean_r2, std_r2, mse_list, r2_list = evaluate_multiple_seeds(final_model_list)

  # Visualize detailed results for single (representative) network
  mse_test_nn, r2_test_nn, y_predicted = plot_single_network_predictions(mse_list, r2_list, final_model_list, final_history_list, x_test_scaled, y_test, scaler_y)

  # Run linear regression fit
  y_predicted_test_linear = fit_linear_regression_model(x_train, y_train, x_test, y_test)

  # Visualize linear regression fit results
  plot_linear_regression_predictions(y_test, y_predicted_test_linear)

if __name__ == "__main__":
    main()
