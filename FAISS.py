import sqlite3
import numpy as np
import os
import pickle
import time


class DatabaseHelper:
    """
    Separate helper class for SQL operations
    """
    def __init__(self, db_path):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

    def fetch_distinct_movies(self):
        return {row[0]: idx for idx, row in enumerate(self.cursor.execute("SELECT DISTINCT movie_id FROM movies"))}

    def fetch_all_users(self):
        return [row[0] for row in self.cursor.execute("SELECT user_id FROM users")]

    def fetch_ratings_for_user(self, user_id):
        return self.cursor.execute("SELECT movie_id, rating FROM ratings WHERE user_id=?", (user_id,))

    def fetch_movie_avg_ratings(self):
        return {row[0]: row[2] for row in self.cursor.execute("SELECT movie_id, total_ratings, rating FROM movies")}

    def fetch_movie_title(self, movie_id):
        self.cursor.execute("SELECT title FROM movies WHERE movie_id=?", (movie_id,))
        result = self.cursor.fetchone()
        return result[0] if result else "Unknown Movie"

    def close(self):
        self.connection.close()


def cosine_similarity(u, v):
    """
    Compute the cosine similarity between two vectors u and v.
    """
    common_indices = np.where(~np.isnan(u) & ~np.isnan(v))[0]

    if len(common_indices) == 0:
        return 0

    u_common = u[common_indices]
    v_common = v[common_indices]

    dot_product = np.dot(u_common, v_common)
    norm_u = np.linalg.norm(u_common)
    norm_v = np.linalg.norm(v_common)

    return dot_product / (norm_u * norm_v)


def create_vectors_for_users(db_helper):
    """
    Creates vector data for cosine similarity operation
    """
    movies = db_helper.fetch_distinct_movies()
    users = db_helper.fetch_all_users()

    total_users = len(users)
    vectors = {}

    start_time = time.time()

    for idx, user in enumerate(users):
        vec = np.full(len(movies), np.nan)
        ratings = db_helper.fetch_ratings_for_user(user)
        for movie_id, rating in ratings:
            vec[movies[movie_id]] = rating
        vectors[user] = vec

        _display_progress(idx, total_users, start_time)
    print("\n")

    return vectors


def _display_progress(current_idx, total, start_time):
    """Displays progress, elapsed, and estimated time."""
    elapsed_time = time.time() - start_time
    avg_time_per_iteration = elapsed_time / (current_idx + 1)
    remaining_iterations = total - current_idx - 1
    estimated_time_left = avg_time_per_iteration * remaining_iterations

    minutes_left, seconds_left = divmod(estimated_time_left, 60)
    progress_percentage = ((current_idx + 1) / total) * 100

    print(f"\rProgress: {progress_percentage:.2f}% done - "
          f"Est. time left: {int(minutes_left)}m {int(seconds_left)}s", end='')


def find_similar_users(target_user_id, user_vectors_data):
    """
    Find users similar to the target_user_id based on cosine similarity.
    """
    target_vector = user_vectors_data[target_user_id]
    similarities = [
        (user_id, cosine_similarity(target_vector, user_vector))
        for user_id, user_vector in user_vectors_data.items()
        if user_id != target_user_id
    ]
    # Sort users based on similarity and return the top user_ids
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:100]


def recommend_movies_for_user(target_user_id, similar_users_list, db_helper, num_recommendations=5):
    """
    Recommends movies based on similar user preferences
    """
    movie_recommendations = {}
    target_user_rated_movies = set(row[0] for row in db_helper.fetch_ratings_for_user(target_user_id))
    movie_avg_ratings = db_helper.fetch_movie_avg_ratings()

    total_sim_users = len(similar_users_list)
    start_time = time.time()

    for idx, (sim_user, similarity) in enumerate(similar_users_list):
        rated_movies = db_helper.fetch_ratings_for_user(sim_user)

        for movie_id, rating in rated_movies:
            if movie_id not in target_user_rated_movies:
                weighted_rating = rating * similarity
                movie_key = movie_id  # Using movie_id alone
                if movie_key in movie_recommendations:
                    movie_recommendations[movie_key].append(weighted_rating)
                else:
                    movie_recommendations[movie_key] = [weighted_rating]

        _display_progress(idx, total_sim_users, start_time)
    print("\r" + " " * 100 + "\r", end='')

    bayesian_c = 1
    sorted_recommendations = sorted(
        [(movie_key, (movie_avg_ratings[movie_key] * bayesian_c + sum(ratings)) / (bayesian_c + len(ratings)))
         for movie_key, ratings in movie_recommendations.items()], key=lambda x: x[1], reverse=True
    )

    sorted_recommendations_with_title = [
        (movie_key, db_helper.fetch_movie_title(movie_key), rating)
        for movie_key, rating in sorted_recommendations
    ]

    return sorted_recommendations_with_title[:num_recommendations]


def save_user_vectors(vectors, filename):
    """ Save user vectors to a file using pickle. """
    with open(filename, 'wb') as f:
        pickle.dump(vectors, f)


def load_user_vectors(filename):
    """ Load user vectors from a file using pickle. """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():

    db_helper = DatabaseHelper('tmdb_sample_ratings.db')

    user_vectors_path = "path_to_save_user_vectors.pkl"

    if os.path.exists(user_vectors_path):
        print("Loading pre-computed user vectors...")
        user_vectors = load_user_vectors(user_vectors_path)
    else:
        print("Computing user vectors...")
        user_vectors = create_vectors_for_users(db_helper)
        print("Saving user vectors for future use...")
        save_user_vectors(user_vectors, user_vectors_path)

    # Input user ID
    while True:
        user_input_id = int(input("What user_id do you want to search?"))
        if user_input_id in user_vectors:
            break
        else:
            print(f"No data available for user {user_input_id}.")

    similar_users_list = find_similar_users(user_input_id, user_vectors)

    print(f"The 5 most similar users to user {user_input_id} are:")
    for user_id, similarity in similar_users_list[:5]:
        print(f"User {user_id} (Similarity rating: {similarity * 100:.2f}%)")

    recommended_movies = recommend_movies_for_user(user_input_id, similar_users_list, db_helper)
    formatted_movies = '\n'.join([f"{movie[1]} (Rating: {movie[2]:.2f})" for movie in recommended_movies])

    print(
        f"\nTop {len(recommended_movies)} movie recommendations for user {user_input_id}:\n{formatted_movies}"
    )

    db_helper.close()


if __name__ == '__main__':
    main()
