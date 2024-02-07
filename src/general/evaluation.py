from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def evaluate_grid_search(grid_search, X_test, y_test):
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return best_model, best_params, mae, mape
