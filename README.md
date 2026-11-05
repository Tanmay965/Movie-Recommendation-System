# Movie Recommendation System (NLP & Machine Learning)

A Machine Learning-based Movie Recommendation System that recommends the **top 5 similar movies** based on movie overviews, genres, keywords, cast, and directors using Natural Language Processing (NLP), TF-IDF, and Cosine Similarity.

The project also includes **Movie Review Sentiment Analysis** using Multinomial Naive Bayes and **Movie Popularity Prediction** using Linear Regression.

Initially developed and tested in Jupyter Notebook, the project was extended with an interactive **Flask web application** that integrates all three machine learning functionalities into a single interface.

## Application Screenshot

![Movie Recommendation System](UI.png)

---

## Files in This Project

| File / Folder | Description |
|---|---|
| `movie_recommender.ipynb` | Main Jupyter Notebook containing data exploration, preprocessing, feature engineering, movie recommendations, sentiment analysis, and popularity prediction. |
| `application.py` | Flask backend that handles application routes, API requests, and communication with the movie recommendation engine. |
| `model.py` | Contains the `MovieEngine` class responsible for building the recommendation system, training ML models, and generating predictions. |
| `reviews.py` | Contains positive and negative movie review samples used to train the sentiment analysis model. |
| `templates/` | Contains HTML templates used to render the web application interface. |
| `static/` | Contains frontend assets such as CSS, JavaScript, and images. |
| `tmdb_5000_movies.csv` | Dataset containing movie titles, overviews, genres, keywords, budgets, popularity, ratings, and runtime. |
| `tmdb_5000_credits.csv` | Dataset containing cast and crew information, including actors and directors. |
| `requirements.txt` | Contains the Python libraries required to run the project. |
| `How to run.txt` | Quick setup and execution instructions. |


---

## Dataset

The project uses the **TMDB 5000 Movies Dataset** and **TMDB 5000 Credits Dataset**.

### 1. TMDB 5000 Movies Dataset

Contains movie-related information such as:
- Movie titles and overviews.
- Genres and keywords.
- Budget and revenue.
- Popularity scores.
- Vote count and vote average.
- Movie runtime.

### 2. TMDB 5000 Credits Dataset

Contains additional movie information, including:
- Cast members and actors.
- Crew information.
- Director details.

### 3. Movie Review Dataset

A collection of positive and negative movie reviews stored in `reviews.py`, used for training and evaluating the sentiment analysis model.

---

## How It Works

The project combines NLP and machine learning techniques to provide three different movie-related functionalities.

### 1. Data Loading & Preprocessing

- Loaded movie and credits datasets using Pandas.
- Merged both datasets using movie IDs instead of movie titles to avoid duplicate entries for remakes.
- Handled missing movie overviews and removed duplicate movie IDs.
- Selected relevant features for recommendation and popularity prediction.

### 2. Feature Extraction & Engineering

Extracted the following movie features:
- Movie overview.
- Genres.
- Keywords.
- Top three actors.
- Director.

Combined these features into a single text-based representation called **movie tags**, capturing the important characteristics of each movie.

### 3. Text Processing

Applied NLP techniques to prepare movie information for similarity calculations.

- Used NLTK's `PorterStemmer` to reduce words to their root forms.
- Applied `TfidfVectorizer` with a maximum of 5,000 features.
- Removed English stop words during vectorization.
- Converted movie tags into numerical vectors.
- Stored the resulting TF-IDF matrix in sparse format to reduce memory usage.

### 4. Movie Recommendation System

Implemented a content-based movie recommendation system using Cosine Similarity.

- Calculates similarity between movie feature vectors.
- Compares movie overviews, genres, keywords, cast, and directors.
- Uses `linear_kernel` to calculate similarity between a selected movie and all other movies.
- Excludes the selected movie from the results.
- Returns the top 5 most similar movies.

The system also provides additional movie information, including:
- Movie title and release year.
- Rating.
- Genres.
- Cast.
- Director.
- Overview.
- Similarity match percentage.

### 5. Movie Review Sentiment Analysis

Implemented a sentiment classification model using **Multinomial Naive Bayes**.

**Workflow:**

1. Combined positive and negative movie reviews from `reviews.py`.
2. Assigned sentiment labels to the reviews.
3. Converted review text into numerical features using `CountVectorizer`.
4. Split the dataset into training and testing sets using an 80:20 ratio.
5. Used stratified sampling to maintain class distribution.
6. Trained a `MultinomialNB` classifier.
7. Evaluated the model using accuracy score.

The application predicts:
- Positive sentiment.
- Negative sentiment.
- Prediction confidence.
- Number of recognized words in the submitted review.

### 6. Movie Popularity Prediction

Implemented a **Linear Regression** model to predict movie popularity scores.

**Input Features:**

| Feature | Description |
|---|---|
| `vote_count` | Number of votes received by the movie. |
| `vote_average` | Average rating of the movie. |
| `budget` | Movie production budget. |
| `runtime` | Duration of the movie. |

**Workflow:**

1. Selected the four numerical features.
2. Handled missing values by replacing them with zero.
3. Split the data into training and testing sets using an 80:20 ratio.
4. Trained a Linear Regression model.
5. Evaluated predictions using R² score and Mean Absolute Error (MAE).
6. Trained the final model on the complete dataset for making predictions.

**Popularity Categorization:**

| Predicted Popularity Score | Category |
|---|---|
| 0–50 | Low |
| Above 50–150 | Medium |
| Above 150 | High |

The system supports popularity prediction through both movie title selection and manually entered numerical features.

---

## Key Features

- **Content-Based Movie Recommendations:** Recommends the top 5 movies similar to a selected movie.
- **NLP-Based Similarity:** Uses movie metadata, genres, keywords, cast, directors, and overviews to identify similar movies.
- **Movie Title Search:** Allows users to search and select movie titles with matching suggestions.
- **Sentiment Analysis:** Classifies movie reviews as positive or negative using Multinomial Naive Bayes.
- **Sentiment Confidence:** Displays prediction confidence and recognized words.
- **Popularity Prediction:** Predicts movie popularity using Linear Regression.
- **Popularity Categorization:** Classifies predicted popularity into Low, Medium, or High.
- **Interactive Web Application:** Provides access to all three functionalities through a Flask-based interface.
- **API Integration:** Uses Flask routes and JSON responses to connect the frontend with machine learning models.

---

## Flask Web Application

Extended the original Jupyter Notebook project by integrating a Flask backend and a web-based user interface.

### Application Architecture

The application uses Flask to connect the frontend with the machine learning functionality implemented in `model.py`.

**Workflow:**

1. User interacts with the web interface.
2. Frontend sends requests to the Flask backend.
3. Flask processes requests through the `MovieEngine` class.
4. The required recommendation or prediction functionality is executed.
5. Results are returned to the frontend in JSON format.

### API Endpoints

| Endpoint | Method | Functionality |
|---|---|---|
| `/` | GET | Loads the homepage with movie and model statistics. |
| `/api/titles` | GET | Searches and returns matching movie titles. |
| `/api/recommend` | GET | Returns the top 5 similar movies for the selected title. |
| `/api/sentiment` | POST | Analyzes a movie review and predicts its sentiment. |
| `/api/popularity` | GET | Predicts popularity using an existing movie title. |
| `/api/popularity` | POST | Predicts popularity using manually provided numerical features. |

### Model Initialization

The `MovieEngine` class initializes all three machine learning components when the Flask application starts.

- Builds the TF-IDF movie recommendation engine.
- Trains the Multinomial Naive Bayes sentiment classifier.
- Trains and evaluates the Linear Regression popularity model.
- Keeps the initialized models available for subsequent API requests.

This avoids rebuilding the models for every individual request.

---

## Model Evaluation

The project evaluates the sentiment analysis and popularity prediction models using separate evaluation metrics.

### Sentiment Analysis

| Evaluation Metric | Implementation |
|---|---|
| Model | Multinomial Naive Bayes |
| Feature Extraction | CountVectorizer |
| Train-Test Split | 80:20 |
| Random State | 42 |
| Stratification | Enabled |
| Evaluation Metric | Accuracy Score |

### Popularity Prediction

| Evaluation Metric | Implementation |
|---|---|
| Model | Linear Regression |
| Input Features | 4 |
| Train-Test Split | 80:20 |
| Random State | 42 |
| Evaluation Metrics | R² Score, Mean Absolute Error (MAE) |

The application also displays the calculated sentiment accuracy, popularity R² score, and popularity MAE on its homepage.

---

## How to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-folder>
```

### 2. Install Dependencies

Open a terminal in the project folder and execute:

```bash
pip install -r requirements.txt
```

### 3. Run the Flask Application

```bash
python application.py
```

### 4. Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

The Movie Recommendation System will be available through the local web application.

---


## Tech Stack

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
- **Machine Learning:** Multinomial Naive Bayes, Linear Regression
- **NLP & Text Processing:** TF-IDF Vectorization, CountVectorizer, Porter Stemming
- **Similarity Measurement:** Cosine Similarity (`linear_kernel`)
- **Backend Framework:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** TMDB 5000 Movies & Credits Dataset
- **Environment:** Jupyter Notebook, Python

## Skills Demonstrated

Python • Data Preprocessing • Exploratory Data Analysis • Natural Language Processing (NLP) • Feature Engineering • TF-IDF • CountVectorizer • Cosine Similarity • Content-Based Filtering • Sentiment Analysis • Multinomial Naive Bayes • Linear Regression • Machine Learning • Model Evaluation • Flask • REST API Development • Web Application Integration
