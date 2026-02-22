# import pickle
# import streamlit as st
# import requests

# def fetch_poster(movie_id):
#     url = "https://api.themoviedb.org/3/movie/{}?api_key=c077bec43e62466c735aad01495a1812&language=en-US".format(movie_id)
#     data = requests.get(url)
#     data = data.json()
#     poster_path = data['poster_path']
#     full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
#     return full_path

# def recommend(movie):
#     index = movies[movies['title'] == movie].index[0]
#     distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
#     recommended_movie_names = []
#     recommended_movie_posters = []
#     for i in distances[1:6]:
#         # fetch the movie poster
#         movie_id = movies.iloc[i[0]].movie_id
#         recommended_movie_posters.append(fetch_poster(movie_id))
#         recommended_movie_names.append(movies.iloc[i[0]].title)

#     return recommended_movie_names,recommended_movie_posters


# st.header('Movie Recommender System')
# movies = pickle.load(open('movie_list.pkl','rb'))
# similarity = pickle.load(open('similarity.pkl','rb'))

# movie_list = movies['title'].values
# selected_movie = st.selectbox(
#     "Type or select a movie from the dropdown",
#     movie_list
# )

# if st.button('Show Recommendation'):
#     recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
#     col1, col2, col3, col4, col5 = st.beta_columns(5)
#     with col1:
#         st.text(recommended_movie_names[0])
#         st.image(recommended_movie_posters[0])
#     with col2:
#         st.text(recommended_movie_names[1])
#         st.image(recommended_movie_posters[1])

#     with col3:
#         st.text(recommended_movie_names[2])
#         st.image(recommended_movie_posters[2])
#     with col4:
#         st.text(recommended_movie_names[3])
#         st.image(recommended_movie_posters[3])
#     with col5:
#         st.text(recommended_movie_names[4])
#         st.image(recommended_movie_posters[4])


# import pickle
# import streamlit as st
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# # ============================
# # TMDB API KEY
# # ============================
# API_KEY = "c077bec43e62466c735aad01495a1812"

# # ============================
# # SESSION WITH RETRY
# # ============================
# session = requests.Session()
# retries = Retry(
#     total=5,
#     backoff_factor=1,
#     status_forcelist=[429, 500, 502, 503, 504],
#     allowed_methods=["GET"]
# )
# session.mount("https://", HTTPAdapter(max_retries=retries))

# # ============================
# # FETCH POSTER (SAFE)
# # ============================
# def fetch_poster(movie_id):
#     url = f"https://api.themoviedb.org/3/movie/{movie_id}"
#     params = {"api_key": API_KEY, "language": "en-US"}

#     try:
#         response = session.get(url, params=params, timeout=10)
#         response.raise_for_status()
#         data = response.json()

#         poster_path = data.get("poster_path")
#         if poster_path:
#             return "https://image.tmdb.org/t/p/w500/" + poster_path
#         else:
#             return "https://via.placeholder.com/300x450?text=No+Poster"

#     except Exception as e:
#         print("Poster Error:", e)
#         return "https://via.placeholder.com/300x450?text=No+Poster"

# # ============================
# # RECOMMENDER FUNCTION
# # ============================
# def recommend(movie):
#     index = movies[movies['title'] == movie].index[0]
#     distances = sorted(
#         list(enumerate(similarity[index])),
#         reverse=True,
#         key=lambda x: x[1]
#     )

#     recommended_movie_names = []
#     recommended_movie_posters = []

#     for i in distances[1:6]:
#         movie_id = movies.iloc[i[0]].movie_id
#         recommended_movie_posters.append(fetch_poster(movie_id))
#         recommended_movie_names.append(movies.iloc[i[0]].title)

#     return recommended_movie_names, recommended_movie_posters

# # ============================
# # STREAMLIT UI
# # ============================
# st.header("🎬 Movie Recommender System")

# movies = pickle.load(open("movie_list.pkl", "rb"))
# similarity = pickle.load(open("similarity.pkl", "rb"))

# movie_list = movies["title"].values

# selected_movie = st.selectbox(
#     "Type or select a movie from the dropdown",
#     movie_list
# )

# if st.button("Show Recommendation"):
#     names, posters = recommend(selected_movie)

#     col1, col2, col3, col4, col5 = st.columns(5)

#     with col1:
#         st.text(names[0])
#         st.image(posters[0])

#     with col2:
#         st.text(names[1])
#         st.image(posters[1])

#     with col3:
#         st.text(names[2])
#         st.image(posters[2])

#     with col4:
#         st.text(names[3])
#         st.image(posters[3])

#     with col5:
#         st.text(names[4])
#         st.image(posters[4])





import pickle
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================
# TMDB API KEY
# ============================
# API_KEY = "c077bec43e62466c735aad01495a1812"
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ============================
# SESSION WITH RETRY
# ============================
@st.cache_resource
def get_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

session = get_session()

# ============================
# LOAD ARTIFACTS (TFIDF + KNN)
# ============================
@st.cache_resource
def load_artifacts():
    movies = pickle.load(open("movie_list.pkl", "rb"))          # DataFrame: movie_id,title,tags
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))     # TfidfVectorizer
    knn = pickle.load(open("knn_model.pkl", "rb"))              # NearestNeighbors

    tfidf_matrix = tfidf.transform(movies["tags"])              # sparse matrix
    return movies, knn, tfidf_matrix

movies, knn, tfidf_matrix = load_artifacts()

# ============================
# FETCH POSTER (SAFE + CACHED)
# ============================
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int) -> str:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": API_KEY, "language": "en-US"}

    try:
        resp = session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        return "https://via.placeholder.com/300x450?text=No+Poster"

    except Exception:
        return "https://via.placeholder.com/300x450?text=No+Poster"

# ============================
# FIND MOVIE INDEX (EXACT -> PARTIAL)
# ============================
def find_movie_index(title: str):
    q = title.strip().lower()

    exact = movies[movies["title"].str.lower() == q]
    if not exact.empty:
        return exact.index[0]

    partial = movies[movies["title"].str.lower().str.contains(q, na=False)]
    if not partial.empty:
        return partial.index[0]

    return None

# ============================
# RECOMMEND EXACTLY 20 MOVIES
# ============================
def recommend_20(movie_title: str):
    idx = find_movie_index(movie_title)
    if idx is None:
        return [], []

    # Always fetch 20 recommendations (+1 for itself)
    distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=21)

    rec_names, rec_posters = [], []
    for i in indices[0][1:]:  # skip itself
        movie_id = int(movies.iloc[i]["movie_id"])
        rec_names.append(movies.iloc[i]["title"])
        rec_posters.append(fetch_poster(movie_id))

    return rec_names, rec_posters

# ============================
# UI
# ============================
st.title("🎬 Movie Recommender System ")
# st.caption("Select a movie and get exactly 20 similar movies using TF-IDF vectors and KNN (cosine distance).")

movie_list = movies["title"].values
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button("Show Recommendation"):
    names, posters = recommend_20(selected_movie)

    if not names:
        st.error("Movie not found. Try another title.")
    else:
        cols_per_row = 5
        idx = 0
        total = len(names)  # should be 20

        while idx < total:
            row_cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                if idx >= total:
                    break
                with row_cols[c]:
                    st.image(posters[idx], use_container_width=True)
                    st.markdown(f"<p style='text-align:center; font-weight:600; margin-top:8px;'>{names[idx]}</p>", unsafe_allow_html=True)

                idx += 1
