import streamlit as st
import pandas as pd
import joblib
import requests
import re


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ---------- Styling ----------

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(126, 87, 194, 0.20),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 85%,
                rgba(54, 180, 190, 0.16),
                transparent 32%
            ),
            linear-gradient(
                135deg,
                #0b0e14,
                #111522,
                #0b0e14
            );
    }

    .block-container {
        max-width: 1050px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 46px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 8px;

        background: linear-gradient(
            90deg,
            #ffffff,
            #b9a7ff,
            #72d6e8,
            #ffffff
        );

        background-size: 250% auto;

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        animation: titleGlow 5s linear infinite;
    }

    @keyframes titleGlow {

        0% {
            background-position: 0% center;
        }

        50% {
            background-position: 100% center;
        }

        100% {
            background-position: 0% center;
        }

    }

    .subtitle {
        text-align: center;
        color: #c4c8d4;
        font-size: 17px;
        margin-bottom: 42px;
    }

    .section-title {
        color: #ffffff;
        font-size: 23px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    div[data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 10px;
    }

    div[data-baseweb="input"] input {
        color: #111111 !important;
        font-weight: 600;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 11px;
        padding: 11px 20px;

        font-size: 16px;
        font-weight: 700;

        color: white;

        background: linear-gradient(
            90deg,
            #7657d9,
            #4faec2,
            #7657d9
        );

        background-size: 200% auto;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;

        animation: buttonGradient 4s linear infinite;
    }

    @keyframes buttonGradient {

        0% {
            background-position: 0% center;
        }

        50% {
            background-position: 100% center;
        }

        100% {
            background-position: 0% center;
        }

    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 8px 25px rgba(110, 90, 220, 0.35);
    }

    /* Recommendation cards */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(
            145deg,
            rgba(29, 35, 56, 0.98),
            rgba(19, 23, 37, 0.98)
        ) !important;

        border: 1px solid rgba(118, 87, 217, 0.65) !important;

        border-radius: 16px !important;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.35) !important;

        padding: 12px !important;

        transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-3px);

        border-color: rgba(114, 214, 232, 0.75) !important;

        box-shadow:
            0 14px 35px rgba(0, 0, 0, 0.45) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] p {
        color: #c4c8d4 !important;
    }

    .recommendation-number {
        color: #b9a7ff;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .movie-title {
        color: #ffffff !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        margin-bottom: 10px !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .rating {
        color: #72d6e8;
        font-weight: 800;
        font-size: 16px;
    }

    .predicted-label {
        color: #c4c8d4 !important;
        font-size: 13px !important;
        margin-top: 2px !important;
    }

    .tmdb-credit {
        text-align: center;
        color: #747b8e;
        font-size: 12px;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------- Header ----------

st.markdown(
    '<div class="main-title">'
    '🎬 Movie Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Get personalized movie recommendations using '
    'collaborative filtering and SVD.'
    '</div>',
    unsafe_allow_html=True
)


# ---------- Load Model ----------

model_data = joblib.load(
    "artifacts/movie_recommendation_svd.pkl"
)

svd_model = model_data["model"]
train_movie_ids = model_data["train_movie_ids"]
movies = model_data["movies"]


# ---------- TMDB Poster Function ----------

@st.cache_data(show_spinner=False)
def get_movie_poster(title):

    if pd.isna(title):
        return None

    title = str(title)

    api_token = st.secrets.get(
        "TMDB_API_TOKEN",
        ""
    ).strip()

    if not api_token:
        return None

    # Extract year from MovieLens title
    year_match = re.search(
        r"\((\d{4})\)\s*$",
        title
    )

    if year_match:

        year = year_match.group(1)

        clean_title = re.sub(
            r"\s*\(\d{4}\)\s*$",
            "",
            title
        )

    else:

        year = None
        clean_title = title

    try:

        params = {
            "query": clean_title,
            "language": "en-US",
            "include_adult": False
        }

        if year:
            params["primary_release_year"] = year

        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            headers={
                "Authorization": f"Bearer {api_token}",
                "accept": "application/json"
            },
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return None

        results = response.json().get(
            "results",
            []
        )

        if not results:
            return None

        selected_movie = results[0]

        # Prefer exact release year
        if year:

            for result in results:

                release_date = result.get(
                    "release_date",
                    ""
                )

                if release_date.startswith(year):

                    selected_movie = result
                    break

        poster_path = selected_movie.get(
            "poster_path"
        )

        if not poster_path:
            return None

        return (
            "https://image.tmdb.org/t/p/w342"
            + poster_path
        )

    except Exception:

        return None


# ---------- User Selection ----------

st.markdown(
    '<div class="section-title">'
    'Choose a User'
    '</div>',
    unsafe_allow_html=True
)

user_id = st.number_input(
    "User ID",
    min_value=1,
    max_value=943,
    value=1,
    step=1
)


# ---------- Get Recommendations ----------

if st.button("Get Recommendations"):

    watched_movies = set(
        inner_movie_id
        for inner_movie_id, _ in svd_model.trainset.ur[
            svd_model.trainset.to_inner_uid(user_id)
        ]
    )

    watched_movies = {
        svd_model.trainset.to_raw_iid(
            inner_movie_id
        )
        for inner_movie_id in watched_movies
    }

    unseen_movies = (
        train_movie_ids
        - watched_movies
    )

    predictions = []

    for movie_id in unseen_movies:

        prediction = svd_model.predict(
            user_id,
            movie_id
        )

        predictions.append(
            (
                movie_id,
                prediction.est
            )
        )

    predictions = sorted(
        predictions,
        key=lambda x: x[1],
        reverse=True
    )

    top_recommendations = predictions[:10]

    recommendations_df = pd.DataFrame(
        top_recommendations,
        columns=[
            "movie_id",
            "predicted_rating"
        ]
    )

    recommendations_df = recommendations_df.merge(
        movies,
        on="movie_id",
        how="left"
    )


    # ---------- Get Posters ----------

    recommendations_df["poster_url"] = (
        recommendations_df["title"]
        .apply(get_movie_poster)
    )


    # ---------- Results ----------

    st.markdown(
        '<div class="section-title">'
        f'Recommended Movies for User {user_id}'
        '</div>',
        unsafe_allow_html=True
    )


    # ---------- Recommendation Cards ----------

    for row_index in range(
        0,
        len(recommendations_df),
        2
    ):

        left_column, right_column = st.columns(2)

        for column, index in zip(
            [left_column, right_column],
            [row_index, row_index + 1]
        ):

            if index >= len(recommendations_df):
                continue

            movie = recommendations_df.iloc[index]

            with column:

                with st.container(border=True):

                    st.markdown(
                        f'<div class="recommendation-number">'
                        f'Recommendation #{index + 1}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


                    # ---------- Poster ----------

                    poster_url = movie["poster_url"]

                    if (
                        pd.notna(poster_url)
                        and poster_url
                    ):

                        st.image(
                            poster_url,
                            width=150
                        )

                    else:

                        st.markdown(
                            """
                            <div style="
                                width:150px;
                                height:220px;
                                margin:0 auto 14px auto;
                                border-radius:10px;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                text-align:center;
                                color:#8f96a8;
                                background:
                                    linear-gradient(
                                        145deg,
                                        rgba(255,255,255,0.06),
                                        rgba(255,255,255,0.02)
                                    );
                                border:1px solid
                                    rgba(255,255,255,0.08);
                                font-size:14px;
                            ">
                                Poster not available
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # ---------- Movie Title ----------

                    st.markdown(
                        f'<div class="movie-title">'
                        f'{movie["title"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


                    # ---------- Predicted Rating ----------

                    st.markdown(
                        f'<div class="rating">'
                        f'⭐ {movie["predicted_rating"]:.2f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="predicted-label">'
                        'Predicted Rating'
                        '</div>',
                        unsafe_allow_html=True
                    )


# ---------- TMDB Attribution ----------

st.markdown(
    '<div class="tmdb-credit">'
    'Movie posters and movie metadata provided by TMDB.'
    '</div>',
    unsafe_allow_html=True
)