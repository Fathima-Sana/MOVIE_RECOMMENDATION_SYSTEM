import pickle 
from flask import Flask, render_template, request, redirect, url_for,session

app = Flask(__name__)
app.secret_key = '123'  # Set a secret key for session management

# Load your recommendation system models and data from the pickle file here.
with open('movie_recommendation.pkl', 'rb') as file:
    data = pickle.load(file)
    model_knn1 = data['model_knn1']  # Load model_knn1
    user_features_df = data['user_features_df']  # Load user_features_df
    rating_popular_movies_df = data['rating_popular_movies_df']
    model_knn2 = data['model_knn2']
    movie_features_df = data['movie_features_df']

def recommenduserbased(user_id=10):
    n_users = 5
    rec_top_n = 10
    distances, indices = model_knn1.kneighbors(user_features_df.loc[user_features_df.index == user_id].values.reshape(1, -1), n_neighbors=n_users + 1)
    user_ids = []
    recommended_titles = []

    for index in range(0, len(distances.flatten())):
        user_ids.append(user_features_df.index[indices.flatten()[index]])

    # look only for movies highly rated by the similar users, not the current user
    candidate_user_ids = user_ids[1:]
    sel_ratings = rating_popular_movies_df.loc[rating_popular_movies_df.user_id.isin(candidate_user_ids)]
    # sort by best ratings and total rating count
    sel_ratings = sel_ratings.sort_values(by=["rating", "total_rating_count"], ascending=False)
    # eliminate from the selection movies that were ranked already by the current user
    movies_rated_by_targeted_user = list(rating_popular_movies_df.loc[rating_popular_movies_df.user_id==user_ids[0]]["movie_id"].values)
    sel_ratings = sel_ratings.loc[~sel_ratings.movie_id.isin(movies_rated_by_targeted_user)]
    # aggregate and count total ratings and total total_rating_count
    agg_sel_ratings = sel_ratings.groupby(["movie_title", "rating"])["total_rating_count"].max().reset_index()
    agg_sel_ratings.columns = ["movie_title", "rating", "total_ratings"]
    agg_sel_ratings = agg_sel_ratings.sort_values(by=["rating", "total_ratings"], ascending=False)
    # only select top n (default top 10 here)
    recommended_titles = agg_sel_ratings["movie_title"].head(10).values

    return recommended_titles

def recommend_item_based(movie_title, top_n=10):
    # Check if the movie_title exists in the dataset
    if movie_title not in movie_features_df.index:
        return []  # Return an empty list if the movie is not found

    # Find the query_index for the given movie_title
    query_index = movie_features_df.index.get_loc(movie_title)

    distances, indices = model_knn2.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=top_n + 1)

    recommended_movies = []
    for index in range(1, len(distances.flatten())):  # Start from 1 to exclude the input movie
        recommended_movies.append(movie_features_df.index[indices.flatten()[index]])

    return recommended_movies


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend')
def recommend():
    return render_template('input.html')

@app.route('/user_recommendations', methods=['POST'])
def user_recommendations():
    user_id = request.form['user_id']  
    error_message1 = None  # Initialize error_message
    if not user_id.isdigit():
        error_message1 = 'Please enter only numeric values for User ID.'
    if error_message1:
        # If there is an error, re-render the form with the error message
        return render_template('input.html', error_message1=error_message1)
    user_id = int(user_id)
    recommended_movies1 = recommenduserbased(user_id)
    return render_template('user_recommendations.html', user_id=user_id, recommended_movies=recommended_movies1,error_message1=error_message1)



@app.route('/movie_recommendations', methods=['POST'])
def movie_recommendations():
    movie_title = request.form['movie_title']
    error_message2 = None  # Initialize error_message

    if movie_title not in movie_features_df.index:
        error_message2 = f'The movie "{movie_title}" does not exist.'

    if error_message2:
        # If there is an error, re-render the form with the error message
        return render_template('input.html', error_message2=error_message2)

    # If no error, proceed with movie recommendations
    recommended_movies2 = recommend_item_based(movie_title)
    return render_template('movie_recommendations.html', movie_title=movie_title, recommended_movies=recommended_movies2, error_message2=error_message2)

if __name__ == '__main__':
    app.run(debug=True)
