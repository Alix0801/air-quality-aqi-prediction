import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pathlib import Path

df = pd.read_csv(r"C:\Users\User\Downloads\Air Quality Data in India (2015 - 2020)\Air Quality Data in India (2015 - 2020)\city_day.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Feature engineering
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["dayofweek"] = df["Date"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
df["quarter"] = df["Date"].dt.quarter

df = df.dropna(subset=["AQI"]).copy()

train = df[df["year"] <= 2018]
val   = df[df["year"] == 2019]
test  = df[df["year"] == 2020]

drop_cols = ["AQI", "AQI_Bucket", "Date"]
X_train, y_train = train.drop(columns=drop_cols), train["AQI"]
X_val,   y_val   = val.drop(columns=drop_cols),   val["AQI"]
X_test,  y_test  = test.drop(columns=drop_cols),  test["AQI"]

cat_cols = ["City"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

def eval_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)  # MSE
    rmse = np.sqrt(mse)                        # RMSE
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

models = {
    "BenchmarkMean": Pipeline([("prep", preprocess),
                              ("model", DummyRegressor(strategy="mean"))]),
    "Ridge": Pipeline([("prep", preprocess),
                       ("model", Ridge())]),
    "RandomForest": Pipeline([("prep", preprocess),
                              ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
    "GradBoost": Pipeline([("prep", preprocess),
                           ("model", GradientBoostingRegressor(random_state=42))]),
}

param_grids = {
    "Ridge": {"model__alpha": [0.1, 1, 10, 100]},
    "RandomForest": {"model__n_estimators": [200, 500],
                     "model__max_depth": [None, 20],
                     "model__min_samples_leaf": [1, 2]},
    "GradBoost": {"model__n_estimators": [200, 500],
                  "model__learning_rate": [0.05, 0.1],
                  "model__max_depth": [2, 3]},
}

# Train + tune (validation-based tuning)
results = []

# benchmark
m = models["BenchmarkMean"]
m.fit(X_train, y_train)
val_pred = m.predict(X_val)
test_pred = m.predict(X_test)
results.append(("BenchmarkMean", "-", *eval_metrics(y_val, val_pred), *eval_metrics(y_test, test_pred)))

# tuned models
for name in ["Ridge", "RandomForest", "GradBoost"]:
    grid = GridSearchCV(models[name], param_grids[name],
                        scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    val_pred = best.predict(X_val)
    test_pred = best.predict(X_test)

    results.append((name, str(grid.best_params_),
                    *eval_metrics(y_val, val_pred),
                    *eval_metrics(y_test, test_pred)))

res_df = pd.DataFrame(results, columns=[
    "Model", "BestParams",
    "Val_RMSE", "Val_MAE", "Val_R2",
    "Test_RMSE", "Test_MAE", "Test_R2"
])
print(res_df.sort_values("Val_RMSE"))
pd.set_option("display.max_colwidth", None)
print(res_df[["Model","BestParams"]].to_string(index=False))