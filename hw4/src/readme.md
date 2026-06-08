# Homework 4: Movie Recommendation Pipeline on Amazon MWAA

## Environment Setup
* **MWAA Environment:** conebusters-v2-hw4
* **S3 Bucket:** yegon-jay-lab6

## How to Run
1. **Upload Source Data:** Ensure the raw data zip is located at `s3://yegon-jay-lab6/homework4/ml-1m.zip` and code files are uploaded to the `dags/` folder.
2. **Execute Movie Embeddings:** Go to the Airflow Web UI, unpause and manually trigger `embeddings_dag` to encode the movie dataset.
3. **Execute Recommendation Engine:** Once the embeddings task completes successfully, trigger `recommendation_dag` to compute user states and generate recommendations across 4 parallel iterations.

## Expected Outputs (S3 Bucket)
* `homework4/embeddings/movie_embeddings.npy` (Movie vector matrix dictionary)
* `homework4/embeddings/movies_meta.csv` (Processed movie metadata DataFrame)
* `homework4/state/user_embeddings_[0-3].pkl` (Sampled user preference vectors per iteration)
* `homework4/recommendations/recs_iter[0-3]_[timestamp].json` (Final top-5 recommendations for cold and active users)

## AI Usage Note
AI Usage: We used AI (Gemini and Claude) for brainstorming pipeline structure, drafting S3 helper functions, and debugging MWAA dependency issues (specifically implementing an inline fallback block to resolve a `ModuleNotFoundError` for `sentence-transformers`). All code was reviewed and verified by us before execution.
