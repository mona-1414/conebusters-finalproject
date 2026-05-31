from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# build spark
spark = SparkSession.builder \
    .appName("Conebusters Modeling") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
    .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000") \
    .getOrCreate()

#inputs
S3_DATA = 's3a://de300-project7/processed/consolidated_features/'
#outputs
S3_MODEL_OUT    = "s3a://de300-project7/models/random_forest_fare/"
S3_IMPORTANCE   = "s3a://de300-project7/models/feature_importance.csv"

# load features

print("Loading consolidated feature store...")
df = spark.read.parquet(S3_DATA)
print(f"Total rows loaded: {df.count():,}")
df.printSchema()

# Feature and Target Selection

# Numeric... (commented out old features)

NUMERIC_FEATURES = [
    "trip_distance",
    "trip_duration",
    "passenger_count",
    "RatecodeID",
    "pickup_hour", 
    "Is_Weekend",             # 0/1 flag
    "pickup_avg_median_volume",   # ATC traffic at pickup borough/dow/hour
    "dropoff_avg_median_volume",  # ATC traffic at dropoff borough/dow/hour
    "traffic_distance_interaction",      # distance * pickup traffic volume
    "tolls_amount",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
    "extra",
    "mta_tax",
]

CATEGORICAL_FEATURES = [
    "pickup_boro",   # borough name string
    "dropoff_boro",
    "Hour_Bin",         # Morning_Rush / Off_Peak_Day / Evening_Rush / Late_Night_Off_Peak
    "payment_type",
    "store_and_fwd_flag",
]
'''
NUMERIC_FEATURES = [
    "trip_distance",
    "trip_duration",
    "passenger_count",
    "RatecodeID",
    "pickup_hour", 
    "Is_Weekend",             # 0/1 flag
    "pickup_avg_median_volume",   # ATC traffic at pickup borough/dow/hour
    "dropoff_avg_median_volume",  # ATC traffic at dropoff borough/dow/hour
    "tolls_amount",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
    "extra",
    "mta_tax",
]
CATEGORICAL_FEATURES = [
    "pickup_boro",   # borough name string
    "dropoff_boro",
    "Hour_Bin",         # Morning_Rush / Off_Peak_Day / Evening_Rush / Late_Night_Off_Peak
]
'''

TARGET = "fare_amount"

# Keep only rows where every feature and the target are non-null
all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
df_model = df.select(*all_cols).dropna()
print(f"Rows after null purge: {df_model.count():,}")

# 80-20 train test split
train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)
print(f"Train rows: {train_df.count():,}  |  Test rows: {test_df.count():,}")

# Build pipeline
# StringIndexer -> VectorAssembler -> RfRegressor
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in CATEGORICAL_FEATURES
]
 
indexed_cat_cols = [f"{c}_idx" for c in CATEGORICAL_FEATURES]
assembler_inputs = NUMERIC_FEATURES + indexed_cat_cols
 
assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features",
    handleInvalid="keep"
)
 
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=TARGET,
    numTrees=100,         # large ensemble → stable importances
    maxDepth=10,          # deep enough to capture interactions
    minInstancesPerNode=5,
    featureSubsetStrategy="auto",  # sqrt(p) for regression
    seed=42,
)
 
pipeline = Pipeline(stages=indexers + [assembler, rf])

#train

print("Training Random Forest (100 trees, maxDepth=10)...")
model = pipeline.fit(train_df)
print("Training complete.")

#eval on test set

predictions = model.transform(test_df)
 
rmse_eval = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction",
                                metricName="rmse")
r2_eval   = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction",
                                metricName="r2")
mae_eval  = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction",
                                metricName="mae")
 
rmse = rmse_eval.evaluate(predictions)
r2   = r2_eval.evaluate(predictions)
mae  = mae_eval.evaluate(predictions)

print("MODEL EVALUATION RESULTS (Test Set)")
print(f"  RMSE  : ${rmse:.4f}")
print(f"  MAE   : ${mae:.4f}")
print(f"  R²    : {r2:.6f}")

rf_model       = model.stages[-1]          # RandomForestRegressionModel
importances    = rf_model.featureImportances.toArray()
feature_names  = assembler_inputs          # same order VectorAssembler used
 
importance_pairs = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)
 
print("FEATURE IMPORTANCE RANKING")

# Persist importance table to S3 as a single CSV for downstream reporting
importance_df = spark.createDataFrame(
    [(name, float(imp)) for name, imp in importance_pairs],
    schema=["feature", "importance"]
)
importance_df \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(S3_IMPORTANCE)
 
print(f"Feature importance CSV saved to: {S3_IMPORTANCE}")

#Congestion Analysis
print("\nCONGESTION INFLUENCE ANALYSIS")
print("Comparing fares across Hour_Bin categories on predictions...\n")
 
congestion_summary = predictions.groupBy("Hour_Bin").agg(
    F.count("*").alias("num_trips"),
    F.round(F.avg("fare_amount"), 2).alias("avg_actual_fare"),
    F.round(F.avg("prediction"), 2).alias("avg_predicted_fare"),
    F.round(F.avg("pickup_avg_median_volume"), 1).alias("avg_pickup_traffic_vol"),
).orderBy("avg_pickup_traffic_vol", ascending=False)
 
congestion_summary.show(truncate=False)


 
# Borough-level breakdown
print("BOROUGH-LEVEL FARE vs CONGESTION BREAKDOWN\n")
borough_summary = predictions.groupBy("pickup_boro").agg(
    F.count("*").alias("num_trips"),
    F.round(F.avg("fare_amount"), 2).alias("avg_actual_fare"),
    F.round(F.avg("pickup_avg_median_volume"), 1).alias("avg_traffic_vol"),
    F.round(F.avg("trip_duration"), 2).alias("avg_trip_duration_min"),
).orderBy("avg_traffic_vol", ascending=False)
 
borough_summary.show(truncate=False)
 
# ---------------------------------------------------------------------------
# 10. Save trained pipeline to S3
# ---------------------------------------------------------------------------
model.write().overwrite().save(S3_MODEL_OUT)
print(f"Full ML pipeline saved to: {S3_MODEL_OUT}")
 
print("\nDone! Pipeline complete.")
spark.stop()
 