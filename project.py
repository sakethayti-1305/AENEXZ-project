import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



print("========== TITANIC DATASET ==========")

df = pd.read_csv("titanic.csv")

df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

accuracies = []
names = []

for name, model in models.items():
    model.fit(X_train_t, y_train_t)
    predictions = model.predict(X_test_t)
    acc = accuracy_score(y_test_t, predictions)
    accuracies.append(acc)
    names.append(name)
    print(f"{name} Accuracy: {acc:.4f}")

print("\n--- Cross Validation ---")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} CV Mean Accuracy: {scores.mean():.4f}")


best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train_t, y_train_t)
y_pred = best_model.predict(X_test_t)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_t, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_t, y_pred))


plt.figure()
plt.bar(names, accuracies)
plt.xticks(rotation=45)
plt.title("Titanic Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()



print("\n========== IRIS DATASET ==========")

from sklearn.datasets import load_iris

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

rf_iris = RandomForestClassifier(random_state=42)
rf_iris.fit(X_train_i, y_train_i)

print("Iris Accuracy:", rf_iris.score(X_test_i, y_test_i))


print("\n========== HOUSE PRICE DATASET ==========")

df_house = pd.read_csv("house_price.csv")

df_house = df_house.drop("Id", axis=1)

y_house = df_house["SalePrice"]
X_house = df_house.drop("SalePrice", axis=1)

categorical_cols = X_house.select_dtypes(include=["object"]).columns
X_house[categorical_cols] = X_house[categorical_cols].fillna("Unknown")

numeric_cols = X_house.select_dtypes(exclude=["object"]).columns
X_house[numeric_cols] = X_house[numeric_cols].fillna(X_house[numeric_cols].median())

X_house = pd.get_dummies(X_house, drop_first=True)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train_h, y_train_h)

predictions = rf_reg.predict(X_test_h)

print("House Price R2 Score:", r2_score(y_test_h, predictions))

import joblib
joblib.dump(rf_reg, "house_price_model.pkl")