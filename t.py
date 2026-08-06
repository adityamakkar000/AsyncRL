from google.cloud import storage


def gcs_exists(gcs_path: str) -> bool:
    """
    Check whether a GCS object exists.

    Example:
        gcs_exists("gs://my-bucket/path/to/file.txt")
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError("Path must start with gs://")

    path = gcs_path[len("gs://") :]
    bucket_name, blob_name = path.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    return blob.exists(client)


if __name__ == "__main__":
    path = "gs://arl-experiments/runs/advan_41/checkpoints/50/"

    if gcs_exists(path):
        print(f"{path} exists")
    else:
        print(f"{path} does not exist")
