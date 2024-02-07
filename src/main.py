import sys
import time
import yaml
from datetime import datetime
from sklearn.model_selection import GridSearchCV

from apartments.pipeline import pipeline as apartments_pipeline, param_grid as apartments_param_grid
from houses.pipeline import pipeline as houses_pipeline, param_grid as houses_param_grid
from lands.pipeline import pipeline as lands_pipeline, param_grid as lands_param_grid

from general.load_data import load_data
from general.split_data import split_data
from general.evaluation import evaluate_grid_search
from general.save_results import save_results


property_type = sys.argv[1]
dump_target = sys.argv[2]
if property_type not in ["apartments", "houses", "lands"]:
    raise ValueError(f"Invalid property_type: {property_type}")

if dump_target not in ["cloud", "local"]:
    raise ValueError(f"Invalid dump_target: {dump_target}")

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

param_grid_dict = {"apartments": apartments_param_grid, "houses": houses_param_grid, "lands": lands_param_grid}
pipeline_dict = {"apartments": apartments_pipeline, "houses": houses_pipeline, "lands": lands_pipeline}


df = load_data(f"otodom_{property_type}", config["columns_to_load"][property_type])
X, y, X_test, y_test = split_data(df, config["dropna_columns"][property_type])

grid_search = GridSearchCV(pipeline_dict[property_type], param_grid_dict[property_type],
                           cv=10, verbose=1, scoring='neg_mean_absolute_error')


start = time.perf_counter()
grid_search.fit(X, y)
stop = time.perf_counter()
fit_time = stop - start


best_model, best_model_params, mae, mape = evaluate_grid_search(grid_search, X_test, y_test)

results = {
    "time [s]": round(fit_time),
    "prams_grid": param_grid_dict[property_type],
    "best_params": best_model_params,
    "metrics": {"MAE": round(mae, 3), "MAPE": round(mape, 3)}
}


results_label = f"{property_type}_{datetime.strftime(datetime.now(), '%Y-%m-%d')}_MAPE: {round(mape, 3)}"
save_results(results, best_model, results_label, target=dump_target)
