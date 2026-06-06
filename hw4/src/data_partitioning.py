from .config import S3_BUCKET, ZIP_S3_KEY, PARTITION_BOUNDS
from .s3_utils import load_zip_data

def load_all_ratings(bucket=S3_BUCKET):
    _, ratings = load_zip_data(bucket, ZIP_S3_KEY)
    return ratings.sort_values("Timestamp").reset_index(drop=True)

def get_new_users(ratings, iteration):
    current = set(get_cumulative_ratings(ratings, iteration)["UserID"])
    if iteration == 0:
        return current
    previous = set(get_cumulative_ratings(ratings, iteration - 1)["UserID"])
    return current - previous

def get_cumulative_ratings(ratings, iteration):
    lo = PARTITION_BOUNDS[0][0]
    hi = PARTITION_BOUNDS[iteration][1]
    mask = ratings["Timestamp"] >= lo
    if hi is not None:
        mask &= ratings["Timestamp"] <= hi
    return ratings[mask].copy()