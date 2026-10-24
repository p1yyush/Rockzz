"""Train the style model. Put Fake.csv and True.csv (Kaggle: Fake and Real News) next to this file, then run:
    python train_model.py
Creates tfidf.pkl and model.pkl (label 1 = real, 0 = fake)."""
import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

fake = pd.read_csv("Fake.csv"); fake["label"] = 0
real = pd.read_csv("True.csv"); real["label"] = 1
df = pd.concat([fake, real]).sample(frac=1, random_state=42)
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

X_tr, X_te, y_tr, y_te = train_test_split(df["content"], df["label"], test_size=0.2, random_state=42)
vec = TfidfVectorizer(stop_words="english", max_df=0.7)
model = LogisticRegression(max_iter=1000)
model.fit(vec.fit_transform(X_tr), y_tr)
print("Test accuracy:", model.score(vec.transform(X_te), y_te))

joblib.dump(vec, "tfidf.pkl")
joblib.dump(model, "model.pkl")
print("Saved tfidf.pkl and model.pkl")
