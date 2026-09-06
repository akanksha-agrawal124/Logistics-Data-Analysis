import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("logistics_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())



df = df.dropna(subset=["Delivery_Time"])



features = [
    "Transportation_Cost",
    "Quantity",
    "Distance",
    "Shipping_Mode"
]

target = "Delivery_Time"

X = df[features]
y = df[target]



categorical_features = ["Shipping_Mode"]

numerical_features = [
    "Transportation_Cost",
    "Quantity",
    "Distance"
]



numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\n-----------------------------")
print("Linear Regression Results")
print("-----------------------------")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)


rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ]
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)



rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_predictions)
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\n-----------------------------")
print("Random Forest Results")
print("-----------------------------")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2
    ]
})

print("\n-----------------------------")
print("Model Comparison")
print("-----------------------------")
print(comparison)



parameter_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [3, 5, 10],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    rf_model,
    parameter_grid,
    cv=5,
    scoring="neg_mean_absolute_error"
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


print("\n-----------------------------")
print("Best Parameters")
print("-----------------------------")
print(grid_search.best_params_)

final_predictions = best_model.predict(X_test)

final_mae = mean_absolute_error(
    y_test,
    final_predictions
)

final_rmse = np.sqrt(
    mean_squared_error(y_test, final_predictions)
)

final_r2 = r2_score(
    y_test,
    final_predictions
)

print("\n-----------------------------")
print("Final Model Results")
print("-----------------------------")
print("MAE:", final_mae)
print("RMSE:", final_rmse)
print("R2 Score:", final_r2)



new_shipment = pd.DataFrame({
    "Transportation_Cost": [250],
    "Quantity": [10],
    "Distance": [150],
    "Shipping_Mode": ["Road"]
})

predicted_time = best_model.predict(new_shipment)

print("\n-----------------------------")
print("New Shipment Prediction")
print("-----------------------------")
print("Predicted Delivery Time:",
      predicted_time[0], "hours")


print("\n-----------------------------")
print("Optimization Recommendations")
print("-----------------------------")

print("1. Optimize routes for shipments with high predicted delivery time.")
print("2. Allocate resources to high-distance shipments.")
print("3. Select suitable shipping modes to reduce delays.")
print("4. Monitor transportation costs.")
print("5. Use predictions for better shipment scheduling.")