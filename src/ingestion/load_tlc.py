from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TLC Data Ingestion") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
    .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000") \
    .getOrCreate()

S3_BUCKET = "s3a://de300-project7/raw/tlc/"

df = spark.read.parquet(S3_BUCKET)

print(f"Total rows: {df.count()}")
print(f"Columns: {df.columns}")
df.printSchema()
df.show(5)