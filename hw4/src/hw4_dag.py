
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
    schedule_interval=None,          # triggered manually / by CI
    catchup=False,
    tags=["conebusters", "embeddings"],
) as embeddings_dag:
 
    generate_movie_embeddings = PythonOperator(
        task_id="generate_movie_embeddings",
        python_callable=lambda **ctx: __import__(
            "hw4.src.pipeline", fromlist=["task_generate_movie_embeddings"]
        ).task_generate_movie_embeddings(**ctx),
    )
 
# per user reccomendations and iterations 
 
NUM_ITERATIONS = 4
 
with DAG(
    dag_id="recommendation_dag",
    description="Compute user embeddings and generate recommendations for 4 iterations.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,          # triggered manually / downstream of embeddings_dag
    catchup=False,
    tags=["conebusters", "recommendations"],
) as recommendation_dag:
 
    for i in range(1, NUM_ITERATIONS + 1):
 
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