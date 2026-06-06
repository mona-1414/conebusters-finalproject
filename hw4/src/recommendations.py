import datetime
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .config import S3_BUCKET, RECOMMENDATIONS_PREFIX, TOP_N_RECOMMENDATIONS, TOP_USER_PERCENTILE, RANDOM_SEED
from .s3_utils import upload_json

def recommend_cold_user(cumulative_ratings, movies_df, n=TOP_N_RECOMMENDATIONS):
    popular = cumulative_ratings.groupby("MovieID").size().reset_index(name="count")
    popular = popular[popular["MovieID"].isin(movies_df["MovieID"])]
    top = popular.sort_values("count", ascending=False).head(n)
    return movies_df[movies_df["MovieID"].isin(top["MovieID"])]["Title"].tolist()

def select_top_user(cumulative_ratings):
    counts = cumulative_ratings.groupby("UserID")["Rating"].count()
    threshold = counts.quantile(TOP_USER_PERCENTILE / 100)
    top_users = counts[counts >= threshold].index.tolist()
    return int(np.random.default_rng(RANDOM_SEED).choice(top_users))

def generate_and_save_recommendations(cumulative_ratings, movie_embeddings, user_embeddings, movies_df, bucket, iteration):
    from .user_embeddings import compute_user_embedding

    now = datetime.datetime.utcnow()
    results = []

    # Cold user
    results.append({
        "User_Type": "cold",
        "Last_Interaction_Time": None,
        "Num_Ratings": 0,
        "Recommendations": recommend_cold_user(cumulative_ratings, movies_df),
        "iteration": iteration,
        "generated_at": now.isoformat() + "Z",
    })

    # Top user
    top_uid = select_top_user(cumulative_ratings)
    u_ratings = cumulative_ratings[cumulative_ratings["UserID"] == top_uid]
    already_rated = set(u_ratings["MovieID"].tolist())

    movie_ids = list(movie_embeddings.keys())
    emb_matrix = np.stack([movie_embeddings[mid] for mid in movie_ids])

    taste_vector = user_embeddings.get(top_uid)
    if taste_vector is None:
        taste_vector = compute_user_embedding(u_ratings, movie_embeddings)

    if taste_vector is not None:
        sims = cosine_similarity(taste_vector.reshape(1, -1), emb_matrix)[0]
        top_indices = [i for i in sims.argsort()[::-1]
                       if movie_ids[i] not in already_rated][:TOP_N_RECOMMENDATIONS]
        top_recs = movies_df[movies_df["MovieID"].isin([movie_ids[i] for i in top_indices])]["Title"].tolist()
    else:
        top_recs = recommend_cold_user(cumulative_ratings, movies_df)

    results.append({
        "User_Type": "top",
        "Last_Interaction_Time": str(pd.to_datetime(int(u_ratings["Timestamp"].max()), unit="s")),
        "Num_Ratings": int(len(u_ratings)),
        "Avg_Rating": round(float(u_ratings["Rating"].mean()), 2),
        "Recommendations": top_recs,
        "iteration": iteration,
        "generated_at": now.isoformat() + "Z",
    })

    key = f"{RECOMMENDATIONS_PREFIX}recs_iter{iteration}_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    upload_json(results, bucket, key)
    print(f"Saved → s3://{bucket}/{key}")
    return key