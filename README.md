# Conebusters Final Project: NYC Taxi Fare Prediction

## Team

- Mona Gomaa
- Jay Yegon
- Nicholas Melo

## Infrastructure

- **S3 Bucket:** `de300-project7` —> all data lives here, never commit data locally
- **EC2 Instance:** `conebusters-project` (us-east-2) —> all code runs here

## Local Setup

1. Clone the repo: `git clone https://github.com/mona-1414/conebusters-finalproject`
2. Create virtual env: `python -m venv .venv`
3. Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`

## EC2 Setup

1. Get the `conebusters-key.pem` from the team drive.
2. SSH in: `ssh -i "conebusters-key.pem" ec2-user@<ec2-public-ip>`
3. Navigate to repo: `cd conebusters-finalproject`
4. Pull latest code: `git pull`

## Daily Workflow

```
# Write code locally in VSCode, then:
git add .
git commit -m "your message"
git push

# On EC2:
git pull
python3 src/your_script.py
```

## Data (S3)

- `s3://de300-project7/raw/tlc/` — Yellow taxi trip records (Jan-Mar 2026)
- `s3://de300-project7/raw/atc/` — Automated Traffic Volume Counts (~1.8M rows)

## General Pipeline Outline

1. `src/ingestion/` — Load TLC + ATC data from S3 into PySpark
2. `src/preprocessing/` — Clean and impute missing values
3. `src/fare_analysis/` — Congestion vs fare analysis
4. `src/modeling/` — PySpark Random Forest fare prediction

## Running the Code

Follow the EC2 and local setup above and run the below code in EC2 in order.

`python3 src/ingestion/load_tlc.py`  

`python3 src/ingestion/load_atc.py`  

`python3 src/preprocessing/clean_tlc_data.py`  

`python3 src/preprocessing/clean_atc_data.py`  

`consolidate_features.ipynb` Run All

`python3 src/fare_analysis/inspect_data.py`  

`python3 src/fare_analysis/congestion_analysis.py`  

`python3 src/fare_analysis/generate_plots.py`  

`python3 src/modeling/random_forest.py`
