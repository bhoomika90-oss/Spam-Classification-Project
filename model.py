import pandas as pd
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("spam_dataset.csv")

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

df['message'] = df['message'].apply(preprocess)

# Vectorization (BIG IMPROVEMENT)
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Better model
model = LogisticRegression()
model.fit(X, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained!")