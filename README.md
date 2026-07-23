# Fake News Detection Using Machine Learning

## Project Overview
A Machine Learning project that detects whether a news article is Fake or Real using Naive Bayes and Logistic Regression algorithms.

## Dataset
- **Source:** ISOT Fake News Dataset (Kaggle)
- **Link:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **Files:** Fake.csv and True.csv
- **Total Samples:** 44,898 articles

## Technologies Used
- Python 3.14
- Scikit-learn
- NLTK
- Pandas
- Matplotlib
- Imbalanced-learn (SMOTE)
- Joblib

## How to Run
1. Clone this repository
2. Install dependencies:
   pip install pandas scikit-learn nltk imbalanced-learn matplotlib joblib
3. Place Fake.csv and True.csv inside fakenews folder
4. Run:
   python fakenews.py

## Results
| Model | Accuracy |
|-------|----------|
| Naive Bayes | 93.7% |
| Logistic Regression | 98.9% |

## What Makes This Different
- Two algorithms compared instead of one
- TF-IDF applied on title and body separately
- SMOTE used to handle class imbalance
- Confidence score output along with Fake/Real prediction

## Future Scope
- Add LSTM deep learning model
- Build a web interface
- Support Tamil and Hindi news detection
