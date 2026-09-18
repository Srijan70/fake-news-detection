import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import joblib
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# ─── CLEAN TEXT ───────────────────────────────────────────
def clean(text):
    text = re.sub(r'\W', ' ', str(text).lower())
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# ─── EXTRA FEATURES ──────────────────────────────────────
def extract_extra_features(df, text_col='combined'):
    features = pd.DataFrame()
    features['word_count']   = df[text_col].apply(lambda x: len(str(x).split()))
    features['char_count']   = df[text_col].apply(lambda x: len(str(x)))
    features['exclamation']  = df[text_col].apply(lambda x: str(x).count('!'))
    features['question']     = df[text_col].apply(lambda x: str(x).count('?'))
    features['caps_count']   = df[text_col].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    features['avg_word_len'] = df[text_col].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)
    features['unique_words'] = df[text_col].apply(lambda x: len(set(str(x).split())))
    return features

# ─── LOAD ISOT DATASET ───────────────────────────────────
print("Loading ISOT dataset...")
try:
    fake = pd.read_csv("Fake.csv")
    real = pd.read_csv("True.csv")
    fake["label"] = 0
    real["label"] = 1
    isot_df = pd.concat([fake, real]).reset_index(drop=True)
    isot_df['title'] = isot_df['title'].fillna('')
    isot_df['text']  = isot_df['text'].fillna('')
    isot_df['combined'] = (isot_df['title'] + ' ' + isot_df['text']).apply(clean)
    print(f"ISOT loaded: {len(isot_df)} samples")
except Exception as e:
    print(f"ISOT not found, skipping: {e}")
    isot_df = pd.DataFrame()

# ─── LOAD WELFAKE DATASET ────────────────────────────────
print("Loading WELFake dataset...")
try:
    wel = pd.read_csv("WELFake_Dataset.csv")
    wel = wel.dropna(subset=['text', 'label'])
    wel['label'] = wel['label'].apply(lambda x: 1 if x == 0 else 0)
    wel['title'] = wel['title'].fillna('')
    wel['combined'] = (wel['title'] + ' ' + wel['text']).apply(clean)
    wel = wel[['combined', 'label']]
    print(f"WELFake loaded: {len(wel)} samples")
except Exception as e:
    print(f"WELFake not found, skipping: {e}")
    wel = pd.DataFrame()

# ─── COMBINE DATASETS ────────────────────────────────────
if not isot_df.empty and not wel.empty:
    df = pd.concat([isot_df[['combined', 'label']], wel], ignore_index=True)
    print(f"\nCombined dataset: {len(df)} samples")
elif not isot_df.empty:
    df = isot_df[['combined', 'label']]
else:
    df = wel

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Fake (0): {(df['label']==0).sum()} | Real (1): {(df['label']==1).sum()}")

# ─── FEATURE EXTRACTION ──────────────────────────────────
print("\nExtracting features...")
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['combined'])

extra = extract_extra_features(df)
scaler = MinMaxScaler()
X_extra = csr_matrix(scaler.fit_transform(extra))

X = hstack([X_tfidf, X_extra])
y = df['label']

# ─── TRAIN TEST SPLIT ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── NAIVE BAYES ─────────────────────────────────────────
print("\nTraining Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc  = round(accuracy_score(y_test, nb_pred) * 100, 2)
print(f"Naive Bayes Accuracy: {nb_acc}%")

# ─── LOGISTIC REGRESSION ─────────────────────────────────
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc  = round(accuracy_score(y_test, lr_pred) * 100, 2)
print(f"Logistic Regression Accuracy: {lr_acc}%")
print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred))

# ─── RANDOM FOREST ───────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = round(accuracy_score(y_test, rf_pred) * 100, 2)
print(f"Random Forest Accuracy: {rf_acc}%")

# ─── ENSEMBLE VOTING (NB + LR + RF) ─────────────────────
print("\nEnsemble Voting (NB + LR + RF)...")
ensemble_pred = []
for nb_p, lr_p, rf_p in zip(nb_pred, lr_pred, rf_pred):
    votes = [nb_p, lr_p, rf_p]
    ensemble_pred.append(max(set(votes), key=votes.count))
ensemble_acc = round(accuracy_score(y_test, ensemble_pred) * 100, 2)
print(f"Ensemble Accuracy: {ensemble_acc}%")

# ─── RESULTS SUMMARY ─────────────────────────────────────
print("\n========== RESULTS SUMMARY ==========")
print(f"Naive Bayes Accuracy      : {nb_acc}%")
print(f"Logistic Regression       : {lr_acc}%")
print(f"Random Forest             : {rf_acc}%")
print(f"Ensemble (NB + LR + RF)   : {ensemble_acc}%")
print("======================================")

# ─── SAVE MODELS ─────────────────────────────────────────
joblib.dump(lr, "fakenews_model.pkl")
joblib.dump(nb, "nb_model.pkl")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nAll models saved!")

# ─── CONFUSION MATRIX ────────────────────────────────────
cm = confusion_matrix(y_test, ensemble_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Ensemble (NB + LR + RF)")
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion Matrix saved!")

# ─── PREDICT FUNCTION ────────────────────────────────────
def predict_news(news):
    cleaned = clean(news)
    tfidf_feat = tfidf.transform([cleaned])
    temp_df = pd.DataFrame({'combined': [cleaned]})
    extra_feat = extract_extra_features(temp_df)
    extra_scaled = csr_matrix(scaler.transform(extra_feat))
    features = hstack([tfidf_feat, extra_scaled])

    nb_p  = nb.predict(features)[0]
    lr_p  = lr.predict(features)[0]
    rf_p  = rf.predict(features)[0]
    lr_prob = lr.predict_proba(features)[0]

    votes = [nb_p, lr_p, rf_p]
    final = max(set(votes), key=votes.count)
    confidence = round(max(lr_prob) * 100, 2)

    label    = "REAL" if final == 1 else "FAKE"
    nb_label = "REAL" if nb_p  == 1 else "FAKE"
    lr_label = "REAL" if lr_p  == 1 else "FAKE"
    rf_label = "REAL" if rf_p  == 1 else "FAKE"

    print(f"\nNews        : {news}")
    print(f"Naive Bayes : {nb_label}")
    print(f"Log Reg     : {lr_label}")
    print(f"Rand Forest : {rf_label}")
    print(f"Final Vote  : {label} ({confidence}% confidence)")

# ─── LIVE PREDICTION ─────────────────────────────────────
print("\n--- LIVE FAKE NEWS DETECTOR ---")
while True:
    news = input("\nEnter news headline (or type 'exit' to quit): ")
    if news.lower() == 'exit':
        break
    predict_news(news)
