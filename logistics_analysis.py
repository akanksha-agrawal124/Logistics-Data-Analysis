import pandas as pd
import matplotlib.pyplot as plt


data = {
    "Distance": [10, 25, 15, 40, 30],
    "Delivery_Time": [1, 3, 2, 5, 4],
    "Transportation_Cost": [100, 250, 150, 400, 300],
    "Delivery_Status": ["On-Time", "On-Time", "Late", "On-Time", "Late"]
}


df = pd.DataFrame(data)


print("Logistics Data:")
print(df)


average_delivery_time = df["Delivery_Time"].mean()
print("\nAverage Delivery Time:", average_delivery_time)


on_time_deliveries = (df["Delivery_Status"] == "On-Time").sum()
total_deliveries = len(df)

on_time_delivery_rate = (on_time_deliveries / total_deliveries) * 100
print("On-Time Delivery Rate:", on_time_delivery_rate, "%")


average_cost = df["Transportation_Cost"].mean()
print("Average Transportation Cost:", average_cost)


plt.bar(df["Distance"], df["Delivery_Time"])
plt.xlabel("Distance")
plt.ylabel("Delivery Time")
plt.title("Distance vs Delivery Time")
plt.show()