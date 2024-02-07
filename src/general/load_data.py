import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()


def load_data(table_name, columns_to_load, shuffle=True):
    username = os.environ["VPS_POSTGRESQL_USER"]
    password = os.environ["VPS_POSTGRESQL_PASSWORD"]
    host = os.environ["VPS_POSTGRESQL_HOST"]
    port = os.environ["VPS_POSTGRESQL_PORT"]
    dbname = os.environ["VPS_POSTGRESQL_DBNAME"]

    conn_str = f"postgresql://{username}:{password}@{host}:{port}/{dbname}"

    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 9999999", conn_str)[columns_to_load]

    if shuffle:
        df = df.sample(len(df), ignore_index=True)

    return df
