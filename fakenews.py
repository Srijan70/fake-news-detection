import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
fake = pd.read_csv("fakenews/Fake.csv")
real = pd.read_csv("fakenews/True.csv")

# Label them
fake["label"] = 0
real["label"] = 1

# Combine
df = pd.concat([fake, real])
df = df.sample(frac=1).reset_index(drop=True)

# Clean text
stop_words = set(stopwords.words('english'))
def clean(text):
    text = re.sub(r'\W', ' ', str(text).lower())
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

df["text"] = (df["title"] + " " + df["text"]).apply(clean)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["text"])
y = df["label"]

# SMOTE
sm = SMOTE()
X, y = sm.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred))
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Save the model
joblib.dump(lr, "fakenews_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("\nModel saved successfully!")

# Confusion Matrix
cm = confusion_matrix(y_test, lr_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion Matrix saved!")

# Predict your own news
def predict_news(news):
    cleaned = clean(news)
    vectorized = tfidf.transform([cleaned])
    prediction = lr.predict(vectorized)[0]
    confidence = lr.predict_proba(vectorized)[0]
    label = "REAL" if prediction == 1 else "FAKE"
    score = round(max(confidence) * 100, 2)
    print(f"\nNews: {news}")
    print(f"Prediction: {label}")
    print(f"Confidence: {score}%")

# Test with sample news
predict_news("NASA discovers water on Mars surface confirmed by scientists")
predict_news("Government secretly putting chips in vaccines to control people")
