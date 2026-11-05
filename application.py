"""
Flask backend for the Movie Recommendation System.

Put this file in the same folder as tmdb_5000_movies.csv and tmdb_5000_credits.csv.

Run:   python application.py
Open:  http://127.0.0.1:5000
"""
import os

from flask import Flask, jsonify, render_template, request

from model import MovieEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Built once at start-up (takes a few seconds: stemming + TF-IDF on ~4800 movies)
engine = MovieEngine(data_dir=BASE_DIR)


@app.route("/")
def home():
    return render_template("index.html", stats=engine.stats())


@app.route("/api/titles")
def titles():
    return jsonify(engine.search_titles(request.args.get("q", "")))


@app.route("/api/recommend")
def recommend():
    title = request.args.get("title", "")
    result = engine.recommend(title, 5)
    if result is None:
        return jsonify({"error": f"'{title}' not found. Pick a title from the suggestions."}), 404
    return jsonify(result)


@app.route("/api/sentiment", methods=["POST"])
def sentiment():
    review = (request.get_json(silent=True) or {}).get("review", "").strip()
    if not review:
        return jsonify({"error": "Please write a review first."}), 400
    return jsonify(engine.predict_sentiment(review))


@app.route("/api/popularity", methods=["GET", "POST"])
def popularity():
    if request.method == "GET":
        title = request.args.get("title", "")
        result = engine.predict_popularity_by_title(title)
        if result is None:
            return jsonify({"error": f"'{title}' not found."}), 404
        return jsonify(result)

    data = request.get_json(silent=True) or {}
    try:
        feats = {f: float(data[f]) for f in engine.FEATURES}
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Send numeric vote_count, vote_average, budget and runtime."}), 400
    return jsonify(engine.predict_popularity(feats))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
