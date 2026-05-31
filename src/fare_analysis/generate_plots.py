import io
import os
import boto3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKET   = "de300-project7"
ANALYSIS = "processed/analysis/"
PLOTS_S3 = "processed/analysis/plots/"
PLOTS_LOCAL = "/tmp/plots"

os.makedirs(PLOTS_LOCAL, exist_ok=True)
s3 = boto3.client("s3")


def read_spark_csv(folder):
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=ANALYSIS + folder + "/")
    key = next(o["Key"] for o in response["Contents"] if o["Key"].endswith(".csv"))
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def save(fig, filename):
    path = os.path.join(PLOTS_LOCAL, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    s3.upload_file(path, BUCKET, PLOTS_S3 + filename)
    plt.close(fig)
    print(f"saved {filename}")


hour_df   = read_spark_csv("fare_by_hour_bin")
boro_df   = read_spark_csv("fare_by_borough")
peak_df   = read_spark_csv("peak_vs_offpeak_by_borough")

# Staten Island has 211 trips out of 7.5M which is not enough to plot meaningfully
boro_df  = boro_df[boro_df["pickup_boro"] != "Staten Island"]
peak_df  = peak_df[peak_df["pickup_boro"] != "Staten Island"]

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"


# Average fare by borough
fig, ax = plt.subplots(figsize=(8, 5))
sorted_boro = boro_df.sort_values("avg_fare", ascending=True)
bars = ax.barh(sorted_boro["pickup_boro"], sorted_boro["avg_fare"], color=BLUE)
ax.bar_label(bars, fmt="$%.2f", padding=4)
ax.set_xlabel("Average Fare ($)")
ax.set_title("Average Taxi Fare by Pickup Borough")
ax.set_xlim(0, sorted_boro["avg_fare"].max() * 1.18)
save(fig, "fare_by_borough.png")


# Peak vs off-peak fare by borough
boroughs = sorted(peak_df["pickup_boro"].unique())
peak_vals    = peak_df[peak_df["period"] == "Peak"].set_index("pickup_boro")["avg_fare"]
offpeak_vals = peak_df[peak_df["period"] == "Off_Peak"].set_index("pickup_boro")["avg_fare"]

x, w = np.arange(len(boroughs)), 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, [peak_vals[b]    for b in boroughs], w, label="Peak",     color=ORANGE)
ax.bar(x + w/2, [offpeak_vals[b] for b in boroughs], w, label="Off-Peak", color=BLUE)
ax.set_xticks(x)
ax.set_xticklabels(boroughs)
ax.set_ylabel("Average Fare ($)")
ax.set_title("Peak vs Off-Peak Average Fare by Borough")
ax.legend()
save(fig, "peak_vs_offpeak_by_borough.png")


# Pearson correlations with fare
corr_labels = ["Trip Distance", "Traffic × Distance", "Trip Duration", "Velocity", "Traffic Volume"]
corr_values = [0.9400, 0.8710, 0.8138, 0.5889, -0.3019]
colors = [BLUE if v > 0 else RED for v in corr_values]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(corr_labels, corr_values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.bar_label(bars, fmt="%.3f", padding=4)
ax.set_xlabel("Pearson r")
ax.set_title("Pearson Correlation with Fare Amount")
ax.set_xlim(-0.5, 1.15)
save(fig, "correlations_with_fare.png")


# Average velocity by time period (lower velocity = more congestion)
hour_order    = ["Morning_Rush", "Off_Peak_Day", "Evening_Rush", "Late_Night_Off_Peak"]
display_names = ["Morning Rush", "Off Peak Day", "Evening Rush", "Late Night"]
hour_sorted   = hour_df.set_index("Hour_Bin").reindex(hour_order).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(display_names, hour_sorted["avg_velocity"], color=GREEN)
ax.bar_label(bars, fmt="%.4f", padding=3)
ax.set_ylabel("Average Velocity (miles/min)")
ax.set_title("Average Trip Velocity by Time Period\n(lower = more congested)")
save(fig, "velocity_by_hour_bin.png")


# Traffic volume vs avg fare by borough (bubble size = trip count)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(
    boro_df["avg_traffic_volume"],
    boro_df["avg_fare"],
    s=boro_df["trip_count"] / 3000,
    color=BLUE, alpha=0.7, edgecolors="white", linewidth=1.5
)
for _, row in boro_df.iterrows():
    ax.annotate(row["pickup_boro"],
                (row["avg_traffic_volume"], row["avg_fare"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_xlabel("Avg Traffic Volume")
ax.set_ylabel("Average Fare ($)")
ax.set_title("Traffic Volume vs Average Fare by Borough\n(bubble size = trip count)")
save(fig, "traffic_vs_fare_by_borough.png")

print("all plots saved to s3://" + BUCKET + "/" + PLOTS_S3)