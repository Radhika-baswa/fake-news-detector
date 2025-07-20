# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Create a data folder if not exists
os.makedirs("data", exist_ok=True)

# Load your combined fake and real news dataset
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels: 0 = fake, 1 = real
fake_df["label"] = 0
true_df["label"] = 1

# Combine datasets
df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42)  # shuffle

# Use only the title/text column (you can choose one or merge both)
X = df["title"].fillna("")  # or use df["text"] or df["title"] + " " + df["text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorizer
print("🔠 Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75, min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("🎯 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("📦 Saved model.pkl and vectorizer.pkl")
