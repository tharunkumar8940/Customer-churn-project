import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load Dataset
df = pd.read_csv("data/raw/customer_churn_1000.csv")

# Encode categorical columns
encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "ContractType",
    "InternetService",
    "TechSupport",
    "PaymentMethod",
    "Churn"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Test Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save Model
joblib.dump(model, "models/customer_churn_model.pkl")

print("Model saved successfully!")