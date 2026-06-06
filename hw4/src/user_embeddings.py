import numpy as np
from .config import S3_BUCKET, STATE_PREFIX, USER_SAMPLE_FRACTION, RANDOM_SEED
from .s3_utils import upload_pickle, download_pickle

def compute_user_embedding(user_ratings, movie_embeddings):
    
    liked_ids = user_ratings[user_ratings["Rating"] >= 4]["MovieID"].tolist()
    vecs = [movie_embeddings[mid] for mid in liked_ids if mid in movie_embeddings]
    if not vecs:
        return None
    avg = np.mean(np.stack(vecs), axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg

def compute_and_save_user_embeddings(cumulative_ratings, movie_embeddings, bucket, iteration):
    all_users = list(cumulative_ratings["UserID"].unique())
    rng = np.random.default_rng(RANDOM_SEED + iteration)
    n_sample = max(1, int(len(all_users) * USER_SAMPLE_FRACTION))
    sampled = rng.choice(all_users, size=n_sample, replace=False).tolist()
    print(f"Iteration {iteration}: {len(all_users)} total users, sampling {n_sample}.")

    user_embeddings = {}
    for uid in sampled:
        u_ratings = cumulative_ratings[cumulative_ratings["UserID"] == uid]
        emb = compute_user_embedding(u_ratings, movie_embeddings)
        if emb is not None:
            user_embeddings[uid] = emb

    upload_pickle(user_embeddings, bucket, f"{STATE_PREFIX}user_embeddings_{iteration}.pkl")
    print(f"Saved {len(user_embeddings)} user embeddings.")
    return user_embeddings

def load_user_embeddings(bucket, iteration):
    return download_pickle(bucket, f"{STATE_PREFIX}user_embeddings_{iteration}.pkl")