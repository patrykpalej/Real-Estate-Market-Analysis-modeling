from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from apartments.preprocessing import (apartment_feature_engineering, province_ohencoder,
                                      subregion_ohencoder)


column_transformer = ColumnTransformer(
    transformers=[
        ('feature_engineering', FunctionTransformer(apartment_feature_engineering), ["advert_type", "market", "utc_created_at"]),
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
    'regressor__n_estimators': [500, 600, 700, 800],
    'regressor__max_depth': [70, 80, 90, 100],
    'scaler': [StandardScaler()],
    'regressor': [RandomForestRegressor(n_jobs=-1)]
}
