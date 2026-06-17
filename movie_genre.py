import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (format: text + genre separated by :::)
data = pd.read_csv("train_data.txt", sep=":::", engine="python", names=["title", "genre", "plot"])

# Remove missing values
data.dropna(inplace=True)

# Features and labels
X = data["plot"]
y = data["genre"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Output
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test custom input
sample = ["A young boy discovers he has magical powers and goes to a wizard school"]
print("\nPredicted Genre:", model.predict(sample)[0])