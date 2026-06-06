from .config import S3_BUCKET, MOVIE_EMBEDDINGS_KEY
from .s3_utils import download_pickle

# ── Called once by the embeddings DAG ─────────────────────────────────────────

def task_generate_movie_embeddings(**context):
    from .movie_embeddings import generate_movie_embeddings
    generate_movie_embeddings(bucket=S3_BUCKET)

# ── Called 4 times by the recommendation DAG ──────────────────────────────────

def task_compute_user_embeddings(iteration: int, **context):
    from .data_partitioning import load_all_ratings, get_cumulative_ratings
    from .user_embeddings import compute_and_save_user_embeddings
    ratings = load_all_ratings(S3_BUCKET)
    cum = get_cumulative_ratings(ratings, iteration)
    movie_embeddings = download_pickle(S3_BUCKET, MOVIE_EMBEDDINGS_KEY)
    compute_and_save_user_embeddings(cum, movie_embeddings, S3_BUCKET, iteration)

def task_generate_recommendations(iteration: int, **context):
    from .data_partitioning import load_all_ratings, get_cumulative_ratings
    from .user_embeddings import load_user_embeddings
    from .recommendations import generate_and_save_recommendations
    ratings = load_all_ratings(S3_BUCKET)
    cum = get_cumulative_ratings(ratings, iteration)
    movie_embeddings = download_pickle(S3_BUCKET, MOVIE_EMBEDDINGS_KEY)
    user_embeddings = load_user_embeddings(S3_BUCKET, iteration)
    generate_and_save_recommendations(cum, movie_embeddings, user_embeddings, S3_BUCKET, iteration)