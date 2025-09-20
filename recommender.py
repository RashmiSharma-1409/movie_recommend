import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def get_recommendations(ott_choice, movie_name, k=5):
    # Load dataset based on OTT choice
    ott_files = {
        '1': 'netflix_trimmed.csv',
        '2': 'prime_trimmed.csv',
        '3': 'disney_trimmed.csv'
    }

    if ott_choice not in ott_files:
        return ["Bloody hell! That OTT isn’t even on the list 😾"]

    data = pd.read_csv(ott_files[ott_choice])

    # Create user-item matrix
    def create_matrix(df):
        df = df.dropna(subset=['userId', 'movieId', 'title', 'rating'])

        N = len(df['userId'].unique())
        M = len(df['movieId'].unique())

        user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
        movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

        user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
        movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

        user_index = [user_mapper[i] for i in df['userId']]
        movie_index = [movie_mapper[i] for i in df['movieId']]

        X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

        return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(data)

    # Movie titles mapping
    movie_titles = dict(zip(data['movieId'], data['title']))
    reverse_titles = {v.strip().lower(): k for k, v in movie_titles.items()}

    # Find similar movies function
    def find_similar_movies(movie_id, X, k, metric='cosine'):
        neighbour_ids = []
        if movie_id not in movie_mapper:
            return []

        movie_ind = movie_mapper[movie_id]
        movie_vec = X[movie_ind]
        k += 1
        kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
        kNN.fit(X)
        movie_vec = movie_vec.reshape(1, -1)
        neighbour = kNN.kneighbors(movie_vec, return_distance=False)

        for i in range(k):
            n = neighbour.item(i)
            neighbour_ids.append(movie_inv_mapper[n])

        neighbour_ids.pop(0)
        return neighbour_ids

    # Prepare recommendations
    movie_name_lower = movie_name.strip().lower()
    if movie_name_lower not in reverse_titles:
        return [f"Aiyooo! Movie *{movie_name}* not found in my OTT treasure box 😿"]

    movie_id = reverse_titles[movie_name_lower]
    similar_ids = find_similar_movies(movie_id, X, k)

    recommendations = []
    for mid in similar_ids:
        if mid in movie_titles:
            recommendations.append(movie_titles[mid])

    if not recommendations:
        return ["No cheeky recommendations found ! 😢"]

    return recommendations
