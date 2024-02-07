def split_data(df, dropna_columns):
    test_size = int(len(df) / 5)

    X_test = df.dropna(subset=dropna_columns).drop("price", axis=1).iloc[:test_size]
    y_test = df.dropna(subset=dropna_columns)["price"].iloc[:test_size]

    X = df.dropna(subset=dropna_columns).drop("price", axis=1).iloc[test_size:]
    y = df.dropna(subset=dropna_columns)["price"].iloc[test_size:]

    return X, y, X_test, y_test
