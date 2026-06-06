from .config import S3_BUCKET, MOVIE_EMBEDDINGS_KEY, MOVIES_META_KEY

def _load_movie_embeddings(bucket):
    from .s3_utils import download_npy, download_dataframe
    import numpy as np
    embeddings = download_npy(bucket, MOVIE_EMBEDDINGS_KEY)
    meta = download_dataframe(bucket, MOVIES_META_KEY)
    return dict(zip(meta["MovieID"].astype(int), embeddings)), meta

def task_generate_movie_embeddings(**context):
    from .movie_embeddings import generate_movie_embeddings
    generate_movie_embeddings(bucket=S3_BUCKET)

def task_compute_user_embeddings(iteration: int, **context):
    from .data_partitioning import load_all_ratings, get_cumulative_ratings
    from .user_embeddings import compute_and_save_user_embeddings
    ratings = load_all_ratings(S3_BUCKET)
    cum = get_cumulative_ratings(ratings, iteration)
    movie_embeddings, _ = _load_movie_embeddings(S3_BUCKET)
    compute_and_save_user_embeddings(cum, movie_embeddings, S3_BUCKET, iteration)

def task_generate_recommendations(iteration: int, **context):
    from .data_partitioning import load_all_ratings, get_cumulative_ratings
    from .user_embeddings import load_user_embeddings
    from .recommendations import generate_and_save_recommendations
    ratings = load_all_ratings(S3_BUCKET)
    cum = get_cumulative_ratings(ratings, iteration)
    movie_embeddings, movies_df = _load_movie_embeddings(S3_BUCKET)
    user_embeddings = load_user_embeddings(S3_BUCKET, iteration)
    generate_and_save_recommendations(cum, movie_embeddings, user_embeddings, movies_df, S3_BUCKET, iteration)