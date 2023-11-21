import requests
import sqlite3
import random
import time

# Constants
API_KEY = '321de2d058c5d1f5ab5b4940b099b169'
TMDB_MOVIE_ENDPOINT = "https://api.themoviedb.org/3/discover/movie"

# Connect to SQLite database
conn = sqlite3.connect('tmdb_sample_ratings.db')
cursor = conn.cursor()

# Create tables for users, movies, and ratings
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    total_ratings INTEGER,
    rating REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS ratings (
    user_id INTEGER,
    movie_id INTEGER,
    title TEXT NOT NULL,
    rating REAL,
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(movie_id) REFERENCES movies(movie_id)
)
''')

# Create 10,000 users
for i in range(10000):
    cursor.execute("INSERT INTO users (user_id) VALUES (?)", (i + 1,))

# Fetch 100 movies with between 100 and 5000 ratings using TMDb API
params = {
    'api_key': API_KEY,
    'vote_count.gte': 100,
    'vote_count.lte': 5000,
    'sort_by': 'popularity.desc',
    'page': 1
}

movies_fetched = 0
total_movies = 500
start_time = time.time()

while movies_fetched < total_movies:
    response = requests.get(TMDB_MOVIE_ENDPOINT, params=params).json()
    results = response.get('results', [])

    for movie in results:
        movie_id = movie['id']
        title = movie['title']
        total_ratings = movie['vote_count']
        rating = round(movie['vote_average'] / 2, 1)

        cursor.execute("INSERT OR REPLACE INTO movies (movie_id, title, total_ratings, rating) VALUES (?, ?, ?, ?)",
                       (movie_id, title, total_ratings, rating))
        movies_fetched += 1
        elapsed_time = time.time() - start_time
        estimated_remaining_time = elapsed_time / (movies_fetched + 1) * (total_movies - movies_fetched)
        minutes, seconds = divmod(estimated_remaining_time, 60)

        percentage_done = (movies_fetched / total_movies) * 100
        print(
            f"\rFetching movies: {percentage_done:.2f}% done - Est. time left: {int(minutes)}m "
            f"{int(seconds)}s", end='')

        if movies_fetched >= total_movies:
            break

    params['page'] += 1

# Distribute ratings based on TMDb average rating and vote count
users = [i for i in range(1, 10001)]
movies = [movie for movie in cursor.execute("SELECT movie_id, title, total_ratings, rating FROM movies").fetchall()]

for movie_id, title, total_ratings, rating in movies:
    average_rating = rating / 2.0  # Convert rating to a scale of 0.5 to 5

    ratings = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    ratings_count = {rating: 0 for rating in ratings}

    accumulated_ratings = 0.0
    remaining_votes = total_ratings

    for _ in range(total_ratings):
        possible_ratings = list(ratings)  # Clone the ratings list

        while possible_ratings:
            chosen_rating = random.choice(possible_ratings)
            new_accumulated_ratings = accumulated_ratings + chosen_rating
            new_avg = new_accumulated_ratings / (total_ratings - remaining_votes + 1)

            if 0.5 * remaining_votes + new_accumulated_ratings <= average_rating * total_ratings <= \
                    5 * remaining_votes + new_accumulated_ratings:
                ratings_count[chosen_rating] += 1
                accumulated_ratings = new_accumulated_ratings
                remaining_votes -= 1
                break
            else:
                possible_ratings.remove(chosen_rating)

    users_for_rating = random.sample(users, total_ratings)
    index = 0  # Use this to keep track of where you are in users_for_rating list

    for rating_instance, count in ratings_count.items():
        selected_users = users_for_rating[index:index + count]
        index += count

        for user in selected_users:
            cursor.execute("INSERT INTO ratings (user_id, title, movie_id, rating) VALUES (?, ?, ?, ?)",
                           (user, title, movie_id, rating_instance))

print("\nFinished!")
# Commit and close the connection
conn.commit()
conn.close()
