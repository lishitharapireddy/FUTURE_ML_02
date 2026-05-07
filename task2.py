import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = {
    'ticket_text': [
        'Server is down',
        'Password reset needed',
        'Payment failed',
        'Website not loading',
        'Need account help',
        'System crash issue'
    ],
    'priority': [
        'High',
        'Low',
        'High',
        'High',
        'Low',
        'High'
    ]
}

df = pd.DataFrame(data)

# Features and labels
X = df['ticket_text']
y = df['priority']

# Convert text into vectors
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

# Test custom ticket
sample = ["Customer payment failed"]
sample_vector = vectorizer.transform(sample)

result = model.predict(sample_vector)

print("Predicted Priority:", result[0])