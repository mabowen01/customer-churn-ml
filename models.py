import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import all 7 models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# STEP 1: Load Your Dataset
# ==========================================
df = pd.read_csv("BCCP.csv")

# ==========================================
# STEP 2: Separate Features (X) and Target (y)
# ==========================================
X = df[["credit_score", "age", "tenure", "balance", "products_number", "credit_card", "active_member", "estimated_salary"]]
y = df["churn"]

# ==========================================
# STEP 3: Train / Test Split (80% train, 20% test)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

print(f"Total customers: {len(df)}")
print(f"Training set: {len(X_train)} customers")
print(f"Testing set: {len(X_test)} customers\n")

# ==========================================
# STEP 4: Feature Scaling (For models that need it)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# MODEL 1: Logistic Regression (Uses Scaled Data)
# ==========================================
print("=" * 50)
print("MODEL 1: Logistic Regression")
print("=" * 50)
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, log_pred) * 100:.2f}%")
print(classification_report(y_test, log_pred))


# ==========================================
# MODEL 2: Decision Tree (Uses Unscaled Data)
# ==========================================
print("=" * 50)
print("MODEL 2: Decision Tree")
print("=" * 50)
tree_model = DecisionTreeClassifier(random_state=52)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, tree_pred) * 100:.2f}%")
print(classification_report(y_test, tree_pred))


# ==========================================
# MODEL 3: K-Nearest Neighbors (Uses Scaled Data)
# ==========================================
print("=" * 50)
print("MODEL 3: K-Nearest Neighbors (KNN)")
print("=" * 50)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, knn_pred) * 100:.2f}%")
print(classification_report(y_test, knn_pred))


# ==========================================
# MODEL 4: Random Forest (Uses Unscaled Data)
# ==========================================
print("=" * 50)
print("MODEL 4: Random Forest")
print("=" * 50)
rf_model = RandomForestClassifier(n_estimators=100, random_state=52)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")
print(classification_report(y_test, rf_pred))


# ==========================================
# MODEL 5: Gradient Boosting (Uses Unscaled Data)
# ==========================================
print("=" * 50)
print("MODEL 5: Gradient Boosting")
print("=" * 50)
gb_model = GradientBoostingClassifier(random_state=52)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, gb_pred) * 100:.2f}%")
print(classification_report(y_test, gb_pred))


# ==========================================
# MODEL 6: Support Vector Machine / SVM (Uses Scaled Data)
# ==========================================
print("=" * 50)
print("MODEL 6: Support Vector Machine (SVM)")
print("=" * 50)
svm_model = SVC(random_state=52)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, svm_pred) * 100:.2f}%")
print(classification_report(y_test, svm_pred))


# ==========================================
# MODEL 7: Naive Bayes (Uses Unscaled Data)
# ==========================================
print("=" * 50)
print("MODEL 7: Naive Bayes")
print("=" * 50)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, nb_pred) * 100:.2f}%")
print(classification_report(y_test, nb_pred))