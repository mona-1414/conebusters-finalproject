from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder \
    .appName("ATC-Historical-Profiling-Pipeline") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
    .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000") \
    .getOrCreate()

#raw ATC data from S3 bucket
S3_RAW_ATC = "s3a://de300-project7/raw/atc/Automated_Traffic_Volume_Counts_20260518.csv"
atc = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(S3_RAW_ATC)

print(f"Total raw traffic logs ingested: {atc.count()}")

#strip commas from volume strings so numbers like '1,147' cast to integers cleanly
atc_clean = atc.withColumn("Vol_numeric", F.regexp_replace(F.col("Vol"), ",", ""))
atc_clean = atc_clean.withColumn("Vol_numeric", F.col("Vol_numeric").cast("int"))

#nullify municipal hardware error sentinel codes (-1 and 999) 
#true structural traffic zeros (like midnight drops) are preserved completely untouched
atc_clean = atc_clean.withColumn(
    "Vol_numeric",
    F.when((F.col("Vol_numeric") < 0) | (F.col("Vol_numeric") == 999), F.lit(None))
     .otherwise(F.col("Vol_numeric"))
)

#drop entries missing critical spatial anchors
atc_clean = atc_clean.filter(F.col("SegmentID").isNotNull()).filter(F.col("Boro").isNotNull())

#parse unified dates to isolate cyclical features
timestamp_str_expr = F.concat_ws("-", F.col("Yr"), F.col("M"), F.col("D"))
time_str_expr = F.concat_ws(":", F.col("HH"), F.col("MM"), F.lit("00"))
combined_datetime_expr = F.concat_ws(" ", timestamp_str_expr, time_str_expr)

atc_clean = atc_clean \
    .withColumn("traffic_timestamp", F.to_timestamp(combined_datetime_expr, "yyyy-M-d H:m:ss")) \
    .withColumn("day_of_week", F.dayofweek(F.col("traffic_timestamp")))

print("Compiling 24-year footprints into Spatio-Temporal feature maps...")

#group by core spatio-temporal indices to extract the definitive baseline traffic signature
#using median (approx percentile 0.5) to protect against skewness from extreme rush hours
atc_feature_lookup = atc_clean \
    .groupBy("SegmentID", "Boro", "street", "day_of_week", "HH") \
    .agg(F.percentile_approx("Vol_numeric", 0.5).alias("historical_median_volume"))

print(f"Profile compilation complete. Unique feature matrix rows: {atc_feature_lookup.count()}")


#finalized feature store map to S3
S3_PROCESSED_ATC = "s3a://de300-project7/processed/atc/"
atc_feature_lookup.write \
    .mode("overwrite") \
    .parquet(S3_PROCESSED_ATC)

print(f"Pipeline complete! Historical feature lookup safely stored at: {S3_PROCESSED_ATC}")