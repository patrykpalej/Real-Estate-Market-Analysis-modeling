from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
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

param_grid = {
    'regressor__n_estimators': [600, 650, 700, 750, 800],
    'regressor__max_depth': [90, 100, 110, 120],
    'scaler': [StandardScaler(), MinMaxScaler()],
    'regressor': [RandomForestRegressor(n_jobs=-1)]
}
