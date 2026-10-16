# Movie Recommendation System

## Project Overview

Developed a Movie Recommendation System that recommends similar movies based on their overviews, genres, keywords, cast, and directors using Python, NLP, TF-IDF, and cosine similarity to recommend the top 5 similar movies. Also implemented sentiment analysis using Multinomial Naive Bayes and movie popularity prediction using Linear Regression.

## Tools & Technologies

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
- **Machine Learning:** Multinomial Naive Bayes, Linear Regression
- **NLP & Text Processing:** TF-IDF Vectorization, Count Vectorization, Porter Stemming
- **Similarity Measurement:** Cosine Similarity
- **Environment:** Jupyter Notebook

## Dataset

- **TMDB 5000 Movies Dataset:** Contains movie information such as titles, overviews, genres, keywords, budgets, popularity, ratings, and runtime.
- **TMDB 5000 Credits Dataset:** Contains cast and crew information, including actors and directors.
- **Movie Review Dataset:** A collection of positive and negative movie reviews used for sentiment analysis.

## Workflow

1. **Data Loading & Preprocessing:** Loaded movie and credits datasets, merged them, handled missing values, and selected relevant features.
2. **Feature Extraction:** Extracted genres, keywords, top three actors, and directors. Combined these features with movie overviews to create movie tags.
3. **Text Processing:** Applied stemming using NLTK's PorterStemmer and converted text into numerical vectors using TF-IDF.
4. **Movie Recommendation:** Calculated cosine similarity between movie vectors and recommended the top five similar movies based on the selected title.
5. **Sentiment Analysis:** Converted movie reviews into numerical features using CountVectorizer and trained a Multinomial Naive Bayes classifier to classify sentiment.
6. **Popularity Prediction:** Used Linear Regression with features such as vote count, vote average, budget, and runtime to predict movie popularity.

## Key Features

- **Content-Based Recommendations:** Suggests five movies similar to a selected movie.
- **NLP-Based Similarity:** Uses movie metadata and textual features to identify similar content.
- **Sentiment Analysis:** Classifies movie reviews as positive or negative.
- **Popularity Prediction:** Predicts a movie's popularity score and categorizes it as Low, Medium, or High.
- **Data Preprocessing:** Handles missing values, extracts relevant information, and prepares text for machine learning.

## Outcome

Developed a Movie Recommendation System using content-based filtering, NLP, and cosine similarity. Extended the project with movie review sentiment classification and popularity prediction to explore multiple machine learning applications in the movie domain.

## Skills Demonstrated

Python • Data Preprocessing • Exploratory Data Analysis • Natural Language Processing (NLP) • Feature Engineering • TF-IDF • Cosine Similarity • Naive Bayes • Linear Regression • Machine Learning
