from datetime import datetime
from sklearn.preprocessing import OneHotEncoder


def apartment_feature_engineering(X):
    X = X.copy()

    X["advert_type"] = X["advert_type"] == "PRIVATE"
    X["market"] = X["market"] == "PRIMARY"
    X = X.rename({"advert_type": "is_advert_private", "market": "is_primary_market"}, axis=1)

    X.insert(3, "weekday", X["utc_created_at"].dt.weekday)
    X.insert(4, "season", X["utc_created_at"].dt.month.map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}))
    X.insert(5, "timeline", X["utc_created_at"].apply(lambda x: x - datetime(2023, 1, 1)).dt.days)
    X.drop(columns="utc_created_at", inplace=True)

    return X


province_ohencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
subregion_ohencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
