from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Conebusters_CongestionAnalysis") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
    .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

INPUT  = "s3a://de300-project7/processed/consolidated_features/"
OUTPUT = "s3a://de300-project7/processed/analysis/"

df = spark.read.parquet(INPUT)

df = df.withColumn(
    "period",
    F.when(F.col("Hour_Bin").isin("Morning_Rush", "Evening_Rush"), "Peak")
     .otherwise("Off_Peak")
)

# Fare and congestion metrics by time period
by_hour_bin = df.groupBy("Hour_Bin").agg(
    F.count("*").alias("trip_count"),
    F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
    F.round(F.avg("trip_duration"), 2).alias("avg_duration_min"),
    F.round(F.avg("average_velocity"), 4).alias("avg_velocity"),
    F.round(F.avg("pickup_avg_median_volume"), 2).alias("avg_traffic_volume"),
    F.round(F.avg("congestion_surcharge"), 3).alias("avg_congestion_surcharge"),
    F.round(F.avg("cbd_congestion_fee"), 3).alias("avg_cbd_fee"),
).orderBy("avg_fare", ascending=False)

print("Fare and congestion metrics by time period:")
by_hour_bin.show()

# Fare and traffic by pickup borough
by_boro = df.groupBy("pickup_boro").agg(
    F.count("*").alias("trip_count"),
    F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
    F.round(F.avg("trip_duration"), 2).alias("avg_duration_min"),
    F.round(F.avg("average_velocity"), 4).alias("avg_velocity"),
    F.round(F.avg("pickup_avg_median_volume"), 2).alias("avg_traffic_volume"),
    F.round(F.avg("congestion_surcharge"), 3).alias("avg_congestion_surcharge"),
).orderBy("avg_fare", ascending=False)

print("Fare and traffic by borough:")
by_boro.show()

# Peak vs off-peak split by borough 
peak_by_boro = df.groupBy("pickup_boro", "period").agg(
    F.count("*").alias("trip_count"),
    F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
    F.round(F.avg("trip_duration"), 2).alias("avg_duration_min"),
    F.round(F.avg("average_velocity"), 4).alias("avg_velocity"),
    F.round(F.avg("pickup_avg_median_volume"), 2).alias("avg_traffic_volume"),
).orderBy("pickup_boro", "period")

print("Peak vs Off-Peak by borough:")
peak_by_boro.show(20)

# Pearson correlations: how much does each feature actually move the fare?
print("Pearson correlations with fare_amount:")
corr_cols = [
    "pickup_avg_median_volume",
    "average_velocity",
    "trip_duration",
    "trip_distance",
    "traffic_distance_interaction",
]
for col in corr_cols:
    r = df.stat.corr(col, "fare_amount")
    print(f"  {col:<35} r = {r:.4f}")

by_hour_bin.coalesce(1).write.mode("overwrite").csv(OUTPUT + "fare_by_hour_bin", header=True)
by_boro.coalesce(1).write.mode("overwrite").csv(OUTPUT + "fare_by_borough", header=True)
peak_by_boro.coalesce(1).write.mode("overwrite").csv(OUTPUT + "peak_vs_offpeak_by_borough", header=True)

print("Saved results to", OUTPUT)
spark.stop()