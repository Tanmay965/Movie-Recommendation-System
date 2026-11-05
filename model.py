"""
ML logic for the movie recommender, taken from the notebook and wrapped in a class
so the Flask app can load everything once at start-up.

Three parts:
  1. Content based recommender   (TF-IDF + cosine similarity)
  2. Sentiment analysis          (CountVectorizer + Multinomial Naive Bayes)
  3. Popularity prediction       (Linear Regression)
"""
import ast
import os

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from reviews import NEGATIVE, POSITIVE

_ps = PorterStemmer()


# ---------- small helpers (same as notebook) ----------
def _names(text):
    return [d["name"] for d in ast.literal_eval(text)]


def _top3_cast(text):
    return [d["name"] for d in ast.literal_eval(text)[:3]]


def _director(text):
    for d in ast.literal_eval(text):
        if d.get("job") == "Director":
            return [d["name"]]
    return []


def _squash(words):
    return [w.replace(" ", "") for w in words]


def _stem(text):
    return " ".join(_ps.stem(w) for w in text.split())


class MovieEngine:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self._build_recommender()
        self._build_sentiment()
        self._build_popularity()

    # ------------------------------------------------------------------
    # 1. Recommender
    # ------------------------------------------------------------------
    def _build_recommender(self):
        movies = pd.read_csv(os.path.join(self.data_dir, "tmdb_5000_movies.csv"))
        credits = pd.read_csv(os.path.join(self.data_dir, "tmdb_5000_credits.csv"))

        # Merge on the movie id (not the title) so remakes with the same title
        # do not create duplicate rows.
        credits = credits.drop(columns=["title"]).rename(columns={"movie_id": "id"})
        df = movies.merge(credits, on="id")

        # keep full info for the UI, and for the popularity model
        self.raw = df.copy()

        df = df[["id", "title", "overview", "genres", "keywords", "cast", "crew",
                 "release_date", "vote_average"]].dropna(subset=["overview"])
        df = df.drop_duplicates(subset="id").reset_index(drop=True)

        df["genres_l"] = df["genres"].apply(_names)
        df["keywords_l"] = df["keywords"].apply(_names)
        df["cast_l"] = df["cast"].apply(_top3_cast)
        df["director_l"] = df["crew"].apply(_director)

        tags = (
            df["overview"].apply(str.split)
            + df["genres_l"].apply(_squash)
            + df["keywords_l"].apply(_squash)
            + df["cast_l"].apply(_squash)
            + df["director_l"].apply(_squash)
        )
        df["tags"] = tags.apply(lambda x: " ".join(x)).apply(_stem)

        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        self.vectors = tfidf.fit_transform(df["tags"])  # stays sparse

        self.movies = df[["id", "title", "overview", "genres_l", "cast_l",
                          "director_l", "release_date", "vote_average"]]
        self._title_to_idx = {t.lower(): i for i, t in enumerate(self.movies["title"])}
        self.titles = sorted(self.movies["title"].tolist())

    def _card(self, i, score=None):
        r = self.movies.iloc[i]
        year = str(r["release_date"])[:4] if pd.notna(r["release_date"]) else ""
        card = {
            "id": int(r["id"]),
            "title": r["title"],
            "year": year,
            "rating": round(float(r["vote_average"]), 1),
            "genres": r["genres_l"],
            "cast": r["cast_l"],
            "director": r["director_l"][0] if r["director_l"] else "",
            "overview": r["overview"],
        }
        if score is not None:
            card["match"] = round(float(score) * 100, 1)
        return card

    def search_titles(self, q, limit=8):
        q = q.strip().lower()
        if not q:
            return []
        starts = [t for t in self.titles if t.lower().startswith(q)]
        contains = [t for t in self.titles if q in t.lower() and t not in starts]
        return (starts + contains)[:limit]

    def recommend(self, title, n=5):
        idx = self._title_to_idx.get(title.strip().lower())
        if idx is None:
            return None
        # similarity of ONE movie against all the others (no 4806x4806 matrix needed)
        sims = linear_kernel(self.vectors[idx], self.vectors).ravel()
        order = np.argsort(-sims)
        order = [i for i in order if i != idx][:n]
        return {
            "selected": self._card(idx),
            "recommendations": [self._card(i, sims[i]) for i in order],
        }

    # ------------------------------------------------------------------
    # 2. Sentiment analysis
    # ------------------------------------------------------------------
    def _build_sentiment(self):
        reviews = pd.DataFrame({
            "review": POSITIVE + NEGATIVE,
            "sentiment": ["positive"] * len(POSITIVE) + ["negative"] * len(NEGATIVE),
        })
        self.cv = CountVectorizer()
        X = self.cv.fit_transform(reviews["review"])
        y = reviews["sentiment"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.nb = MultinomialNB().fit(X_tr, y_tr)
        self.sentiment_accuracy = float(accuracy_score(y_te, self.nb.predict(X_te)))
        self.sentiment_samples = int(len(reviews))

    def predict_sentiment(self, review):
        vec = self.cv.transform([review])
        proba = self.nb.predict_proba(vec)[0]
        classes = list(self.nb.classes_)
        label = classes[int(np.argmax(proba))]
        known = int(vec.sum())  # how many words the model actually recognised
        return {
            "sentiment": label,
            "confidence": round(float(proba.max()) * 100, 1),
            "known_words": known,
        }

    # ------------------------------------------------------------------
    # 3. Popularity prediction
    # ------------------------------------------------------------------
    FEATURES = ["vote_count", "vote_average", "budget", "runtime"]

    def _build_popularity(self):
        df = self.raw[["title", "popularity"] + self.FEATURES].fillna(0)
        self.pop_df = df.drop_duplicates(subset="title").reset_index(drop=True)
        X, y = df[self.FEATURES], df["popularity"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        lr = LinearRegression().fit(X_tr, y_tr)
        pred = lr.predict(X_te)
        self.pop_r2 = float(r2_score(y_te, pred))
        self.pop_mae = float(mean_absolute_error(y_te, pred))
        # final model uses all the data so every movie can be queried
        self.lr = LinearRegression().fit(X, y)

    @staticmethod
    def _level(score):
        if score <= 50:
            return "Low"
        if score <= 150:
            return "Medium"
        return "High"

    def predict_popularity_by_title(self, title):
        row = self.pop_df[self.pop_df["title"].str.lower() == title.strip().lower()]
        if row.empty:
            return None
        row = row.iloc[0]
        feats = {f: float(row[f]) for f in self.FEATURES}
        out = self.predict_popularity(feats)
        out["actual_popularity"] = round(float(row["popularity"]), 1)
        out["title"] = row["title"]
        out["inputs"] = feats
        return out

    def predict_popularity(self, feats):
        x = pd.DataFrame([[float(feats[f]) for f in self.FEATURES]], columns=self.FEATURES)
        score = max(float(self.lr.predict(x)[0]), 0.0)
        return {"predicted_popularity": round(score, 1), "level": self._level(score)}

    def stats(self):
        return {
            "movies": int(len(self.movies)),
            "sentiment_accuracy": round(self.sentiment_accuracy * 100, 1),
            "sentiment_samples": self.sentiment_samples,
            "popularity_r2": round(self.pop_r2, 3),
            "popularity_mae": round(self.pop_mae, 2),
        }
