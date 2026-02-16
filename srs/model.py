import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target = "Calories"

train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)


numeric_features = train_df.drop(columns=[target]).select_dtypes(include=np.number)

corr = numeric_features.corr().abs()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Матрица корреляции (train)")
plt.tight_layout()
plt.show()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [c for c in upper.columns if any(upper[c] > 0.9)]


train_df = train_df.drop(columns=high_corr)
test_df = test_df.drop(columns=high_corr)


full = pd.concat(
    [train_df.drop(columns=[target]), test_df],
    axis=0
)

full = pd.get_dummies(full, drop_first=True)

X_train = full.iloc[:len(train_df)].copy()
X_test = full.iloc[len(train_df):].copy()
y_train = train_df[target]


num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
    if X_train[col].nunique() > 2:
        base_corr = X_train[col].corr(y_train)
        log_corr = np.log1p(X_train[col]).corr(y_train)

        if abs(log_corr) > abs(base_corr):
            X_train.loc[:, col] = np.log1p(X_train[col])
            X_test.loc[:, col] = np.log1p(X_test[col])
            print(f"[LOG] применено к {col}")

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print("\nПример полиномиальных признаков:")
print(pd.DataFrame(X_train_poly, columns=poly.get_feature_names_out()).head())


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_poly = StandardScaler()
X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
X_test_poly_scaled = scaler_poly.transform(X_test_poly)


def evaluate(model, X, y, name):
    model.fit(X, y)
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    print(f"\n{name}")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    return model, preds


lr = LinearRegression()
lr_model, lr_pred = evaluate(
    lr, X_train_poly_scaled, y_train, "Linear Regression"
)

rf = RandomForestRegressor(random_state=42, n_estimators=150, max_depth=10)
rf_model, rf_pred = evaluate(
    rf, X_train_scaled, y_train, "Random Forest"
)

xgb = XGBRegressor(
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb_model, xgb_pred = evaluate(
    xgb, X_train_scaled, y_train, "XGBoost"
)


param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print("\nЛучшие параметры XGBoost:")
print(grid.best_params_)

best_pred = best_model.predict(X_train_scaled)

print("\nМетрики после GridSearch:")
print("MAE :", mean_absolute_error(y_train, best_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train, best_pred)))
print("R2  :", r2_score(y_train, best_pred))


errors = y_train - best_pred

plt.figure(figsize=(7, 5))
plt.hist(errors, bins=30, edgecolor="black")
plt.xlabel("Ошибка")
plt.ylabel("Количество")
plt.title("Распределение ошибок")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_train, best_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()],
         "r--")
plt.xlabel("Истинные Calories")
plt.ylabel("Предсказанные Calories")
plt.title("Истина vs Предсказание")
plt.tight_layout()
plt.show()

importances = best_model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8, 5))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X_train.columns[indices])
plt.title("Топ-10 важных признаков")
plt.tight_layout()
plt.show()
