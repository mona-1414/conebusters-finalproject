import numpy as np
from sentence_transformers import SentenceTransformer
from .config import S3_BUCKET, ZIP_S3_KEY, MOVIE_EMBEDDINGS_KEY, MOVIES_META_KEY, BERT_MODEL_NAME
from .s3_utils import load_zip_data, upload_npy, upload_dataframe, download_npy, download_dataframe, key_exists

def generate_movie_embeddings(bucket=S3_BUCKET, force=False):
    if not force and key_exists(bucket, MOVIE_EMBEDDINGS_KEY):
        print("Embeddings already exist in S3, loading...")
        embeddings = download_npy(bucket, MOVIE_EMBEDDINGS_KEY)
        movies_df = download_dataframe(bucket, MOVIES_META_KEY)
        return dict(zip(movies_df["MovieID"].astype(int), embeddings))

    movies_df, _ = load_zip_data(bucket, ZIP_S3_KEY)
    movies_df["text"] = movies_df["Title"] + ": " + movies_df["Genres"].str.replace("|", " ", regex=False)

    print(f"Encoding {len(movies_df)} movies...")
    model = SentenceTransformer(BERT_MODEL_NAME)
    embeddings = model.encode(movies_df["text"].tolist(), show_progress_bar=True)

    upload_npy(embeddings, bucket, MOVIE_EMBEDDINGS_KEY)
    upload_dataframe(movies_df[["MovieID", "Title", "Genres"]], bucket, MOVIES_META_KEY)
    print(f"Saved {len(movies_df)} embeddings.")

    return dict(zip(movies_df["MovieID"].astype(int), embeddings))