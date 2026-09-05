import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("logistics_data.csv")
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


df = df.drop_duplicates()


df["Delivery_Time"] = pd.to_numeric(
    df["Delivery_Time"], errors="coerce"
)

df["Transportation_Cost"] = pd.to_numeric(
    df["Transportation_Cost"], errors="coerce"
)


df["Delivery_Time"] = df["Delivery_Time"].fillna(
    df["Delivery_Time"].median()
)

df["Transportation_Cost"] = df["Transportation_Cost"].fillna(
    df["Transportation_Cost"].median()
)


df["Shipping_Mode"] = df["Shipping_Mode"].fillna(
    df["Shipping_Mode"].mode()[0]
)


df["Shipping_Mode"] = df["Shipping_Mode"].str.strip().str.title()


Q1 = df["Delivery_Time"].quantile(0.25)
Q3 = df["Delivery_Time"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df = df[
    (df["Delivery_Time"] >= lower_limit) &
    (df["Delivery_Time"] <= upper_limit)
]


scaler = MinMaxScaler()
numerical_columns = ["Delivery_Time", "Transportation_Cost"]

df[numerical_columns] = scaler.fit_transform(
    df[numerical_columns]
)

df.to_csv("cleaned_logistics_data.csv", index=False)

print("\nData preprocessing completed successfully!")