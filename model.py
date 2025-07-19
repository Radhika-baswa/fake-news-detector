# model.py
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Create a data folder if not exists
os.makedirs("data", exist_ok=True)

# Load LIAR dataset from Hugging Face
print("🔄 Loading 'liar' dataset...")
dataset = load_dataset("liar")

# Extract relevant data from training set
train_df = pd.DataFrame(dataset["train"])
val_df = pd.DataFrame(dataset["validation"])
test_df = pd.DataFrame(dataset["test"])

# Combine train/val for training, use test set for final evaluation
combined_df = pd.concat([train_df, val_df])
X = combined_df["statement"].fillna("")
y = combined_df["label"]

# Map multi-class labels to binary: 0 = fake, 1 = real
# fake = ['false', 'pants-fire', 'barely-true'] -> 0
# real = ['half-true', 'mostly-true', 'true'] -> 1
label_map = {
    0: 0,  # pants-fire
    1: 0,  # false
    2: 0,  # barely-true
    3: 1,  # half-true
    4: 1,  # mostly-true
    5: 1   # true
}
y = y.map(label_map)

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
