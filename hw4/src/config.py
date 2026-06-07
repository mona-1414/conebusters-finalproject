S3_BUCKET  = "ndm4080"
ZIP_S3_KEY = "homework4/ml-1m.zip"

MOVIE_EMBEDDINGS_KEY   = "homework4/embeddings/movie_embeddings.npy"
MOVIES_META_KEY        = "homework4/embeddings/movies_meta.csv"
STATE_PREFIX           = "homework4/state/"
RECOMMENDATIONS_PREFIX = "homework4/recommendations/"

BERT_MODEL_NAME       = "all-MiniLM-L6-v2"
USER_SAMPLE_FRACTION  = 0.30
TOP_N_RECOMMENDATIONS = 5
TOP_USER_PERCENTILE   = 95
RANDOM_SEED           = 42

_tz = datetime.timezone.utc

def _ts(y, m, d, end_of_day=False):
    return int(datetime.datetime(y, m, d,
                                 23 if end_of_day else 0,
                                 59 if end_of_day else 0,
                                 59 if end_of_day else 0,
                                 tzinfo=_tz).timestamp())

PARTITION_BOUNDS = [
    (_ts(2000, 4, 25), _ts(2000, 8,  3,  end_of_day=True)),
    (_ts(2000, 8,  4), _ts(2000, 10, 31, end_of_day=True)),
    (_ts(2000, 11, 1), _ts(2000, 11, 25, end_of_day=True)),
    (_ts(2000, 11, 26), None),
]