from pyspark.sql import SparkSession
import os

spark = SparkSession.builder \
    .appName("TLC Data Ingestion") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
    .getOrCreate()

S3_BUCKET = "s3://de300-project7/raw/tlc/"

df = spark.read.parquet(S3_BUCKET)

print(f"Total rows: {df.count()}")
print(f"Columns: {df.columns}")
df.printSchema()
df.show(5)