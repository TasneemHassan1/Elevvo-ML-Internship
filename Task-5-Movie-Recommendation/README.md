# Movie Recommendation System

This project builds a movie recommendation system using the MovieLens 100K dataset.

The system uses collaborative filtering to recommend movies that a user has not rated before. I implemented user-based collaborative filtering as the required approach, then explored item-based collaborative filtering and SVD as bonus approaches.

## Project Structure

```text
Task-5-Movie-Recommendation/
│
├── .streamlit/
│   └── secrets.toml.example
│
├── artifacts/
│   └── movie_recommendation_svd.pkl
│
├── data/
│   ├── u.data
│   └── u.item
│
├── app.py
├── movie_recommendation.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The project uses the MovieLens 100K dataset.

* 100,000 ratings
* 943 users
* 1,682 movies
* Ratings from 1 to 5

The ratings were split for each user into:

* 80% training data
* 20% test data

The training data was used to build the recommendation models, while the test data was used to evaluate recommendation quality.

## Methods

### 1. User-Based Collaborative Filtering

A user-item matrix was created from the training data.

Cosine similarity was used to find users with similar rating patterns.

For a selected user, the system:

1. Finds similar users.
2. Uses ratings from the most similar users.
3. Calculates recommendation scores for unseen movies.
4. Removes movies already rated by the user.
5. Returns the top 10 recommendations.

The user-based model achieved:

**Precision@10: 6.81%**

### 2. Item-Based Collaborative Filtering

As a bonus, I implemented item-based collaborative filtering.

Instead of finding similar users, this approach finds movies with similar rating patterns.

The model recommends movies that are similar to movies the user previously rated highly.

The item-based model achieved:

**Precision@10: 22.63%**

### 3. SVD

As another bonus, I implemented Singular Value Decomposition (SVD) using the Surprise library.

The SVD model was trained on the training ratings and used to predict ratings for movies that the user had not rated.

The baseline SVD model achieved:

**Precision@10: 7.39%**

#### SVD Hyperparameter Tuning

Three SVD configurations were tested:

| Model | Factors | Epochs | Regularization | Precision@10 |
| ----- | ------: | -----: | -------------: | -----------: |
| SVD 1 |      50 |     20 |           0.02 |        7.18% |
| SVD 2 |     100 |     20 |           0.02 |        7.39% |
| SVD 3 |     100 |     30 |           0.05 |        7.63% |

Among the tested SVD configurations, SVD 3 achieved the highest Precision@10 and was selected as the final SVD model for the application.

## Evaluation

Precision@10 was used to evaluate the recommendation quality.

The evaluation was performed on 935 users with held-out test ratings.

| Approach                           | Precision@10 |
| ---------------------------------- | -----------: |
| User-Based Collaborative Filtering |        6.81% |
| Item-Based Collaborative Filtering |       22.63% |
| Tuned SVD                          |        7.63% |

## Streamlit Application

The project includes a Streamlit application where the user can enter a MovieLens user ID and receive 10 personalized movie recommendations.

For each recommendation, the application displays:

* Movie title
* Predicted rating
* Movie poster when available

The application uses the tuned SVD model saved in:

```text
artifacts/movie_recommendation_svd.pkl
```

## Movie Posters with TMDB

Movie posters and movie metadata displayed in the application are retrieved from TMDB.

TMDB is only used to display movie posters and metadata. It is not used for the recommendation model or its evaluation.

A TMDB API Read Access Token is required for posters to appear.

### Getting a TMDB API Read Access Token

1. Create an account or log in to TMDB.
2. Open your account settings.
3. Go to the **API** section.
4. Follow the instructions to create API credentials if you do not already have them.
5. From the API settings, copy the **API Read Access Token (v4 auth)**.

### Adding the Token to the Application

A template file is provided in:

```text
.streamlit/secrets.toml.example
```

Copy this file and rename the copy to:

```text
.streamlit/secrets.toml
```

Then open `secrets.toml` and replace the placeholder with your own TMDB API Read Access Token:

```toml
TMDB_API_TOKEN = "YOUR_TMDB_API_READ_ACCESS_TOKEN"
```

The actual `secrets.toml` file is excluded from GitHub using `.gitignore`.

If no TMDB token is provided, the recommendation system still works, but movie posters will not be displayed.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then run the Streamlit application:

```bash
streamlit run app.py
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Surprise
* Matplotlib
* Joblib
* Streamlit
* Requests

## Conclusion

This project demonstrates how collaborative filtering can be used to build a movie recommendation system.

The required user-based collaborative filtering approach was implemented and evaluated using Precision@10. Item-based collaborative filtering and SVD were also explored as bonus approaches.

The tuned SVD model was saved as an artifact and integrated into the Streamlit application to generate personalized movie recommendations.
