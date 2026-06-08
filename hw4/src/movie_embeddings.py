import traceback
import boto3
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import S3_BUCKET, ZIP_S3_KEY, MOVIE_EMBEDDINGS_KEY, MOVIES_META_KEY, BERT_MODEL_NAME
from .s3_utils import load_zip_data, upload_pickle, download_pickle, upload_dataframe, key_exists

def generate_movie_embeddings(bucket=S3_BUCKET, force=False):
    try:
        if not force and key_exists(bucket, MOVIE_EMBEDDINGS_KEY):
            print("Embeddings already exist in S3, loading...")
            return download_pickle(bucket, MOVIE_EMBEDDINGS_KEY) #read as pickle dictionary

        movies_df, _ = load_zip_data(bucket, ZIP_S3_KEY)
        movies_df["text"] = movies_df["Title"] + ": " + movies_df["Genres"].str.replace("|", " ", regex=False)

        print(f"Encoding {len(movies_df)} movies...")
        model = SentenceTransformer(BERT_MODEL_NAME)
        embeddings = model.encode(movies_df["text"].tolist(), show_progress_bar=True)

        #combine keys and arrays into a dictionary map before storing
        movie_embeddings_dict = dict(zip(movies_df["MovieID"].astype(int), embeddings))

        #save via pickle to match pipeline expectations
        upload_pickle(movie_embeddings_dict, bucket, MOVIE_EMBEDDINGS_KEY)
        upload_dataframe(movies_df[["MovieID", "Title", "Genres"]], bucket, MOVIES_META_KEY)
        print(f"Saved {len(movies_df)} embeddings.")

        return movie_embeddings_dict

    except Exception as e:
        # Capture the full traceback error text
        error_msg = traceback.format_exc()
        
        # Upload the error log straight to S3 to bypass Airflow UI log restrictions
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=bucket,
            Key='homework4/mwaa_crash_report.txt',
            Body=error_msg
        )
        
        # Re-raise the exception so Airflow still correctly registers the task failure
        raise e