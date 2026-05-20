from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("TLC Preprocessing") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
    .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000") \
    .getOrCreate()

tlc = spark.read.parquet("s3a://de300-project7/raw/tlc/")

print("Rows before cleaning:", tlc.count())

# Drop rows with invalid values: negative/zero fares, zero distance, 
# zero passengers, and extreme outliers likely caused by data entry errors
tlc_clean = tlc \
    .filter(F.col("fare_amount") > 0) \
    .filter(F.col("fare_amount") <= 800) \
    .filter(F.col("trip_distance") > 0) \
    .filter(F.col("trip_distance") <= 100) \
    .filter(F.col("passenger_count") > 0) \
    .filter(F.col("passenger_count").isNotNull())

# Impute nulls for remaining columns with standard values

# Impute nulls with constant values based on domain knowledge.
# EDA revealed that ~3M null values across RatecodeID, store_and_fwd_flag,
# congestion_surcharge, and Airport_fee all originate from the same vendor,
# making this MNAR (Missing Not At Random). Since the missingness is vendor-driven
# and not related to the trip itself, constant imputation with reasonable defaults is appropriate:
# - RatecodeID: 1.0 (standard rate, most common code)
# - store_and_fwd_flag: "N" (trip was not held in vehicle memory, most common value)
# - congestion_surcharge: 0.0 (assume no surcharge if not reported)
# - Airport_fee: 0.0 (assume no airport fee if not reported)

tlc_clean = tlc_clean \
    .fillna({"RatecodeID": 1.0, "store_and_fwd_flag": "N"}) \
    .fillna({"congestion_surcharge": 0.0, "Airport_fee": 0.0})

print("Rows after cleaning:", tlc_clean.count())

# Save cleaned data to S3
tlc_clean.write.mode("overwrite").parquet("s3a://de300-project7/processed/tlc/")

print("Done! Cleaned TLC data saved to S3.")