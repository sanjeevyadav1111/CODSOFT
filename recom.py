import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample movies dataset
movies = pd.DataFrame({
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Inception', 'Avatar', 'Titanic', 'The Matrix', 'Interstellar'],
    'genre': ['Sci-Fi', 'Sci-Fi', 'Romance', 'Sci-Fi', 'Sci-Fi']
})

# Sample ratings dataset
ratings = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 3],
    'movie_id': [1, 3, 2, 4, 1, 2, 5],
    'rating': [5, 4, 4, 5, 5, 3, 4]
})

# Create a User-Item Matrix
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Calculate User Similarity
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function for User-User Collaborative Filtering
def get_user_recommendations(user_id, user_item_matrix, user_similarity_df):
    similar_users = user_similarity_df[user_id].nlargest(3).index
    similar_users_ratings = user_item_matrix.loc[similar_users]
    recommended_movies = similar_users_ratings.mean(axis=0).sort_values(ascending=False)
    recommended_movies = recommended_movies[recommended_movies > 0]  # Filter out unwatched movies
    return recommended_movies.index.tolist()

# Calculate Movie Similarity for Content-Based Filtering
count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(movies['genre'])
movie_similarity = cosine_similarity(genre_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movies['title'], columns=movies['title'])

# Function for Content-Based Filtering
def get_content_based_recommendations(movie_title, movie_similarity_df):
    similar_movies = movie_similarity_df[movie_title].nlargest(3)
    return similar_movies.index.tolist()

# Main Recommendation Function
def recommend_movies(user_id, liked_movie=None):
    collaborative_recommendations = get_user_recommendations(user_id, user_item_matrix, user_similarity_df)
    content_based_recommendations = get_content_based_recommendations(liked_movie, movie_similarity_df) if liked_movie else []
    
    print(f"Collaborative Recommendations for User {user_id}:")
    print(movies[movies['movie_id'].isin(collaborative_recommendations)])
    
    if liked_movie:
        print(f"\nContent-Based Recommendations for '{liked_movie}':")
        print(content_based_recommendations)

# Example usage
if __name__ == "__main__":
    recommend_movies(1, liked_movie='Inception')
