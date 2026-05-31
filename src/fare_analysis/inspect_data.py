from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("Conebusters_Step1_Inspect")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.InstanceProfileCredentialsProvider")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet("s3a://de300-project7/processed/consolidated_features/")

print(f"Rows: {df.count():,}  |  Cols: {len(df.columns)}")
df.printSchema()
print("\nColumn list:", df.columns)
df.show(5, truncate=True)

analysis_cols = [c for c in ["fare_amount", "trip_distance", "trip_duration",
                              "Average_velocity", "Traffic_distance_interaction"]
                 if c in df.columns]
df.select(analysis_cols).describe().show()

traffic_cols = [c for c in df.columns if "traffic" in c.lower() or "volume" in c.lower()]
print("Traffic-related columns:", traffic_cols)

borough_cols = [c for c in df.columns if "borough" in c.lower()]
print("Borough-related columns:", borough_cols)

if "Hour_Bin" in df.columns:
    df.groupBy("Hour_Bin").count().orderBy("count", ascending=False).show()

if "Is_Weekend" in df.columns:
    df.groupBy("Is_Weekend").count().orderBy("Is_Weekend").show()

for col in borough_cols:
    df.groupBy(col).count().orderBy(F.desc("count")).show()

# null check
key_cols = analysis_cols + traffic_cols + borough_cols
df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in key_cols]).show(truncate=False)

spark.stop()