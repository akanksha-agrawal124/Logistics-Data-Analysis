import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("logistics_data.csv")

print("First 5 rows of the dataset:")
print(df.head())


print("\nDataset Information:")
print(df.info())


print("\nDataset Shape:")
print(df.shape)


print("\nColumn Names:")
print(df.columns.tolist())







print("\nMissing Values:")
print(df.isnull().sum())

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


print("\nDuplicate Rows:")
print(df[df.duplicated()])






print("\nDescriptive Statistics:")
print(df.describe())

print("\nMean Values:")
print(df[["Delivery_Time", "Transportation_Cost", "Quantity", "Distance"]].mean())


print("\nMedian Values:")
print(df[["Delivery_Time", "Transportation_Cost", "Quantity", "Distance"]].median())


print("\nStandard Deviation:")
print(df[["Delivery_Time", "Transportation_Cost", "Quantity", "Distance"]].std())



plt.figure(figsize=(8, 5))

plt.hist(
    df["Delivery_Time"].dropna(),
    bins=8,
    edgecolor="black"
)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")

plt.tight_layout()


plt.savefig("delivery_time_distribution.png")


plt.show()


plt.figure(figsize=(8, 5))

plt.hist(
    df["Transportation_Cost"].dropna(),
    bins=8,
    edgecolor="black"
)

plt.title("Distribution of Transportation Cost")
plt.xlabel("Transportation Cost")
plt.ylabel("Number of Shipments")

plt.tight_layout()


plt.savefig("transportation_cost_distribution.png")


plt.show()



plt.figure(figsize=(8, 5))

plt.scatter(
    df["Distance"],
    df["Transportation_Cost"],
    alpha=0.7
)

plt.title("Distance vs Transportation Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost")

plt.tight_layout()


plt.savefig("distance_vs_transportation_cost.png")


plt.show()






plt.figure(figsize=(8, 5))

plt.scatter(
    df["Distance"],
    df["Delivery_Time"],
    alpha=0.7
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (Days)")

plt.tight_layout()

plt.savefig("distance_vs_delivery_time.png")


plt.show()




status_counts = df["Delivery_Status"].value_counts()

plt.figure(figsize=(7, 5))

plt.bar(
    status_counts.index,
    status_counts.values,
    edgecolor="black"
)

plt.title("Delivery Status of Shipments")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Shipments")

plt.tight_layout()


plt.savefig("delivery_status.png")


plt.show()




mode_counts = df["Shipping_Mode"].str.title().value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    mode_counts.index,
    mode_counts.values,
    edgecolor="black"
)

plt.title("Number of Shipments by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Number of Shipments")

plt.tight_layout()


plt.savefig("shipping_mode_analysis.png")


plt.show()



numerical_data = df[
    ["Delivery_Time", "Transportation_Cost", "Quantity", "Distance"]
]


correlation_matrix = numerical_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Logistics Variables")
plt.tight_layout()

plt.savefig("correlation_heatmap.png")


plt.show()



status_analysis = df.groupby("Delivery_Status").agg({
    "Delivery_Time": "mean",
    "Transportation_Cost": "mean",
    "Distance": "mean",
    "Quantity": "mean"
})

print("\nStatus-wise Average Analysis:")
print(status_analysis)


Q1 = df["Delivery_Time"].quantile(0.25)
Q3 = df["Delivery_Time"].quantile(0.75)


IQR = Q3 - Q1


lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nOutlier Analysis for Delivery Time:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)


outliers = df[
    (df["Delivery_Time"] < lower_limit) |
    (df["Delivery_Time"] > upper_limit)
]

print("\nDelivery Time Outliers:")
print(outliers[["Shipment_ID", "Delivery_Time", "Delivery_Status"]])






print("\n========== FINAL LOGISTICS SUMMARY ==========")

print("\nTotal Shipments:", len(df))

print("\nDelivery Status:")
print(df["Delivery_Status"].value_counts())

print("\nAverage Delivery Time:",
      round(df["Delivery_Time"].mean(), 2), "days")

print("\nAverage Transportation Cost:",
      round(df["Transportation_Cost"].mean(), 2))

print("\nAverage Distance:",
      round(df["Distance"].mean(), 2), "km")

print("\nAverage Quantity:",
      round(df["Quantity"].mean(), 2))

print("\nMost Common Shipping Mode:")
print(df["Shipping_Mode"].str.title().mode()[0])

print("\nHighest Delivery Time:")
print(df["Delivery_Time"].max(), "days")

print("\nHighest Transportation Cost:")
print(df["Transportation_Cost"].max())