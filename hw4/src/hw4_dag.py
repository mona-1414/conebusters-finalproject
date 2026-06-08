#DAG 1 (embeddings_dag):   generate movie embeddings once and store to S3.
#DAG 2 (recommendation_dag): for each of 4 iterations, compute user embeddings
#then generate recommendations (sequential within
#each iteration, iterations run in parallel).

from datetime import datetime, timedelta
 
from airflow import DAG
from airflow.operators.python import PythonOperator
 
# default args
 
DEFAULT_ARGS = {
    "owner": "conebusters",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
 
# genmovie embeddings
 
with DAG(
    dag_id="embeddings_dag",
    description="Generate and upload movie embeddings to S3 (run once).",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,          # triggered manually / by CI
    catchup=False,
    tags=["conebusters", "embeddings"],
) as embeddings_dag:
 
    # Helper to capture top-level package or script import crashes
    def run_with_logging(**ctx):
        import traceback
        import boto3
        import sys
        import subprocess
        try:
            # ── INLINE ESCAPE HATCH ──────────────────────────────────────────
            # Force the active worker node to install the ML library dynamically,
            # bypassing the broken and constrained global AWS console installer.
            print("Executing dynamic worker installation for sentence-transformers...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "--no-cache-dir"])
            # ─────────────────────────────────────────────────────────────────

            return __import__(
                "hw4.src.pipeline", fromlist=["task_generate_movie_embeddings"]
            ).task_generate_movie_embeddings(**ctx)
        except Exception as e:
            error_msg = traceback.format_exc()
            s3 = boto3.client('s3')
            s3.put_object(
                Bucket="yegon-jay-lab6",
                Key="homework4/mwaa_crash_report.txt",
                Body=error_msg,
                ServerSideEncryption='AES256'
            )
            raise e

    generate_movie_embeddings = PythonOperator(
        task_id="generate_movie_embeddings",
        python_callable=run_with_logging,
    )
 
# per user reccomendations and iterations 
 
NUM_ITERATIONS = 4
 
with DAG(
    dag_id="recommendation_dag",
    description="Compute user embeddings and generate recommendations for 4 iterations.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,          # triggered manually / downstream of embeddings_dag
    catchup=False,
    tags=["conebusters", "recommendations"],
) as recommendation_dag:
 
    for i in range(NUM_ITERATIONS):
 
        compute_user_embeddings = PythonOperator(
            task_id=f"compute_user_embeddings_iter_{i}",
            python_callable=lambda iteration=i, **ctx: __import__(
                "hw4.src.pipeline", fromlist=["task_compute_user_embeddings"]
            ).task_compute_user_embeddings(iteration=iteration, **ctx),
        )
 
        generate_recommendations = PythonOperator(
            task_id=f"generate_recommendations_iter_{i}",
            python_callable=lambda iteration=i, **ctx: __import__(
                "hw4.src.pipeline", fromlist=["task_generate_recommendations"]
            ).task_generate_recommendations(iteration=iteration, **ctx),
        )
 
        # Within each iteration: embeddings must finish before recommendations
        compute_user_embeddings >> generate_recommendations
        # Iterations are independent of each other (run in parallel by default)