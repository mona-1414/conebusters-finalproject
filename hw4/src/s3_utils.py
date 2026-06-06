import io, json, pickle, zipfile, boto3
import numpy as np
import pandas as pd

_client = None

def s3():
    global _client
    if _client is None:
        _client = boto3.client("s3")
    return _client

def key_exists(bucket, key):
    from botocore.exceptions import ClientError
    try:
        s3().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False

def upload_bytes(data, bucket, key):
    s3().put_object(Bucket=bucket, Key=key, Body=data)

def download_bytes(bucket, key):
    return s3().get_object(Bucket=bucket, Key=key)["Body"].read()

def upload_json(obj, bucket, key):
    upload_bytes(json.dumps(obj, indent=2, default=str).encode(), bucket, key)

def upload_dataframe(df, bucket, key):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    upload_bytes(buf.getvalue().encode(), bucket, key)

def download_dataframe(bucket, key):
    return pd.read_csv(io.StringIO(download_bytes(bucket, key).decode()))

def upload_npy(arr, bucket, key):
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    upload_bytes(buf.read(), bucket, key)

def download_npy(bucket, key):
    return np.load(io.BytesIO(download_bytes(bucket, key)))

def upload_pickle(obj, bucket, key):
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    upload_bytes(buf.getvalue(), bucket, key)

def download_pickle(bucket, key):
    return pickle.loads(download_bytes(bucket, key))

def load_zip_data(bucket, zip_key):
    """Load movies and ratings directly from the ml-1m zip in S3."""
    obj = s3().get_object(Bucket=bucket, Key=zip_key)
    zip_bytes = io.BytesIO(obj["Body"].read())
    with zipfile.ZipFile(zip_bytes) as z:
        with z.open("ml-1m/movies.dat") as f:
            movies = pd.read_csv(f, sep="::", engine="python",
                                 names=["MovieID", "Title", "Genres"],
                                 encoding="latin-1")
        with z.open("ml-1m/ratings.dat") as f:
            ratings = pd.read_csv(f, sep="::", engine="python",
                                  names=["UserID", "MovieID", "Rating", "Timestamp"],
                                  encoding="latin-1")
    return movies, ratings