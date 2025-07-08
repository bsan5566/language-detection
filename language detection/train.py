import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset with relative path
dataset_path = os.path.join('data', 'dataset.csv')
df = pd.read_csv(dataset_path)

# Drop missing values
df.dropna(subset=['Text', 'language'], inplace=True)

# Define features and target
X = df['Text']
y = df['language']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create faster pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=500, solver='liblinear'))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
model_path = os.path.join('app', 'model.joblib')
joblib.dump(pipeline, model_path)
print(f"✅ Model saved to {model_path}")
