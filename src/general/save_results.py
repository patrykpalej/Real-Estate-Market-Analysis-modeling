import os
import json
import pickle
from google.cloud import storage


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../secrets/cloud-storage-sa-key.json"


def upload_json_to_gcs(bucket_name, json_data, blob_name):
    data_string = json.dumps(json_data)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(data_string, content_type='application/json')


def dump_object_to_gcs(bucket_name, object_data, blob_name):
    pickled_data = pickle.dumps(object_data)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(pickled_data, content_type='application/octet-stream')


def save_results(metadata, best_model, results_label, target="cloud"):
    if target == "cloud":
        bucket_name = "rea-metadata"
        blob_name = f"{results_label}.json"
        metadata = {k: str(v) for k, v in metadata.items()}
        upload_json_to_gcs(bucket_name, metadata, blob_name)

        bucket_name = "rea-models"
        blob_name = f"{results_label}.pickle"
        dump_object_to_gcs(bucket_name, best_model, blob_name)

    elif target == "local":
        with open(f"results/{results_label}.json", "w") as f:
            json.dump(metadata, f)

        with open(f"models/{results_label}.pickle", "wb") as f:
            pickle.dump(best_model, f)
