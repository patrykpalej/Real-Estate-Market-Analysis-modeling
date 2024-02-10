from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer

from houses.preprocessing import (houses_feature_engineering, province_ohencoder,
                                  subregion_ohencoder)


column_transformer = ColumnTransformer(
    transformers=[
        ('feature_engineering', FunctionTransformer(houses_feature_engineering), ["advert_type", "market", "utc_created_at", "location"]),
        ("province_encoder", province_ohencoder, ["province"]),
        ("subregion_encoder", subregion_ohencoder, ["subregion"]),
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('column_transformer', column_transformer),
    ('scaler', None),
    ('regressor', None)
])


def create_param_grid(model):
    regressor = (RandomForestRegressor(n_jobs=-1)
                 if model == "randomforest" else GradientBoostingRegressor())
    param_grid = {
        'regressor__n_estimators': [700, 750, 800, 850, 900],
        'regressor__max_depth': [110, 120, 130, 140],
        'scaler': [StandardScaler(), MinMaxScaler()],
        'regressor': [regressor]
    }
    return param_grid
