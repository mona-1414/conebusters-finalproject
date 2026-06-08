from .config import S3_BUCKET, MOVIE_EMBEDDINGS_KEY, MOVIES_META_KEY
from .s3_utils import download_pickle, download_dataframe, load_all_ratings
from .data_partitioning import get_cumulative_ratings
from .user_embeddings import load_user_embeddings
from .recommendations import generate_and_save_recommendations

# ── Called once by the embeddings DAG ─────────────────────────────────────────

def task_generate_movie_embeddings(**context):
    from .movie_embeddings import generate_movie_embeddings
    generate_movie_embeddings(bucket=S3_BUCKET)

# ── Called 4 times by the recommendation DAG ──────────────────────────────────

def task_compute_user_embeddings(iteration: int, **context):
    from .user_embeddings import compute_and_save_user_embeddings
    ratings = load_all_ratings(S3_BUCKET)
    cum = get_cumulative_ratings(ratings, iteration)
    movie_embeddings = download_pickle(S3_BUCKET, MOVIE_EMBEDDINGS_KEY)
    compute_and_save_user_embeddings(cum, movie_embeddings, S3_BUCKET, iteration)

def task_generate_recommendations(iteration: int, **context):
    ratings = load_all_ratings(S3_BUCKET)
    cum = get_cumulative_ratings(ratings, iteration)
    movie_embeddings = download_pickle(S3_BUCKET, MOVIE_EMBEDDINGS_KEY)
    user_embeddings = load_user_embeddings(S3_BUCKET, iteration)
    
    #grab metadata dataframe needed by recommendations backend
    movies_df = download_dataframe(S3_BUCKET, MOVIES_META_KEY)
    
    #movies_df as the 4th positional argument
    generate_and_save_recommendations(cum, movie_embeddings, user_embeddings, movies_df, S3_BUCKET, iteration)