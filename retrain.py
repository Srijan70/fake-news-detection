import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import nltk
import json
from datetime import datetime

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean(text):
    text = re.sub(r'\W', ' ', str(text).lower())
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def retrain(dataset_path="WELFake_Dataset.csv"):
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting retraining...")
    print(f"Loading dataset: {dataset_path}")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Drop nulls
    df = df.dropna(subset=['text'])
    df = df.dropna(subset=['label'])

    # WELFake: label 1 = Fake, 0 = Real
    # Our model: label 0 = Fake, 1 = Real — so flip
    df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)

    # Combine title + text
    df['title'] = df['title'].fillna('')
    df['combined'] = (df['title'] + ' ' + df['text']).apply(clean)

    print(f"Total samples: {len(df)}")
    print(f"Fake: {(df['label'] == 0).sum()} | Real: {(df['label'] == 1).sum()}")

    X = df['combined']
    y = df['label']

    # TF-IDF
    print("Applying TF-IDF vectorization...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_vec = tfidf.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train Naive Bayes
    print("Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_acc = round(accuracy_score(y_test, nb.predict(X_test)) * 100, 2)
    print(f"Naive Bayes Accuracy: {nb_acc}%")

    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = round(accuracy_score(y_test, lr_pred) * 100, 2)
    print(f"Logistic Regression Accuracy: {lr_acc}%")
    print("\nClassification Report:")
    print(classification_report(y_test, lr_pred))

    # Save model and vectorizer
    joblib.dump(lr, "fakenews_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("Model and vectorizer saved!")

    # Save metadata
    meta = {
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_path,
        "total_samples": len(df),
        "naive_bayes_accuracy": nb_acc,
        "logistic_regression_accuracy": lr_acc,
        "features": 5000
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nRetraining complete! LR Accuracy: {lr_acc}%")
    print(f"Metadata saved to model_meta.json")
    return meta

if __name__ == "__main__":
    retrain("WELFake_Dataset.csv")
