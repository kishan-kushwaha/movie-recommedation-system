🎬 Movie Recommendation System


Overview

This project is a Content-Based Movie Recommendation System built using TF-IDF Vectorization and K-Nearest Neighbors (KNN) to recommend similar movies based on textual features.

It analyzes movie metadata such as overview, genres, keywords, cast, and crew to compute similarity and suggest relevant movies.




Tech Stack

Python

Pandas

NumPy

Scikit-Learn

TF-IDF Vectorizer

K-Nearest Neighbors (KNN)

Git LFS (for large model files)




How It Works
Step 1: Data Preprocessing

Cleaned dataset

Combined important features into single text column

Step 2: Feature Extraction

Applied TF-IDF Vectorizer

Converted text data into numerical vectors

Step 3: Similarity Model

Trained KNN model

Used cosine similarity to find nearest movies

Step 4: Recommendation

Input: Movie Name

Output: Top 20 Similar Movies





Model Files 

Due to GitHub file size limitations, large .pkl model files are stored using Git LFS.






How to Run Locally
pip install -r requirements.txt
python app.py




Example Output

Input:

The Dark Knight

Output:

Batman Begins
The Dark Knight Rises
Man of Steel
Joker
Watchmen





Future Improvements

Hybrid recommendation system

User-based collaborative filtering

Deep Learning based recommendation

Streamlit deployment
