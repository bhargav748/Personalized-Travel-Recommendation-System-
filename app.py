from flask import Flask, jsonify, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample user-item interaction matrix (ratings)
data = {'user1': [4, 0, 3, 5], 'user2': [5, 5, 0, 0], 'user3': [3, 2, 5, 4], 'user4': [0, 3, 4, 0]}
ratings = pd.DataFrame(data, index=['item1', 'item2', 'item3', 'item4'])

# Function to calculate recommendations
def recommend_items(user, user_similarity_df, ratings, num_recommendations=2):
    similar_users = user_similarity_df[user].sort_values(ascending=False).index[1:]
    recommendations = pd.Series(dtype='float64')

    for u in similar_users:
        for item in ratings.index:
            if ratings.at[item, user] == 0:
                recommendations[item] = ratings.at[item, u]

    return recommendations.sort_values(ascending=False).head(num_recommendations)

@app.route('/recommend', methods=['GET'])
def recommend():
    user = request.args.get('user')
    # Calculate cosine similarity between users
    user_similarity = cosine_similarity(ratings.T)
    user_similarity_df = pd.DataFrame(user_similarity, index=ratings.columns, columns=ratings.columns)
    
    recommendations = recommend_items(user, user_similarity_df, ratings)
    
    return jsonify(recommendations.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
