import pandas as pd

# Step 1: Load the datasets
try:
	df_fake = pd.read_csv("Fake.csv")   # This is the file with fake news
	df_real = pd.read_csv("True.csv")   # This is the file with real news
except FileNotFoundError as e:
	print(f"❌ File not found: {e.filename}")
	exit(1)

# Step 2: Add a label column to each
df_fake['label'] = 'FAKE'
df_real['label'] = 'REAL'

# Step 3: Combine both into one DataFrame
df_combined = pd.concat([df_fake, df_real], ignore_index=True)

# Step 4: Optional – shuffle the dataset
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 5: Save the combined dataset
df_combined.to_csv("fake_or_real_news.csv", index=False)

print("✅ Combined dataset saved as 'fake_or_real_news.csv'")
