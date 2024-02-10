from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from lands.preprocessing import (lands_feature_engineering, province_ohencoder, subregion_ohencoder)


column_transformer = ColumnTransformer(
    transformers=[
        ('feature_engineering', FunctionTransformer(lands_feature_engineering), ["advert_type", "location", "utc_created_at"]),
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
        'regressor__n_estimators': [500, 600, 700, 800],
        'regressor__max_depth': [70, 80, 90, 100],
        'scaler': [StandardScaler(), MinMaxScaler()],
        'regressor': [regressor]
    }
    return param_grid
