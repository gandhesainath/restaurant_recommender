from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the dataset
data_path = "C:/data sets/Dataset .csv"
data = pd.read_csv(data_path)

# Data preprocessing
data.dropna(subset=['Cuisines'], inplace=True)

# Encoding categorical variables
data_encoded = pd.get_dummies(data, columns=['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu'])

# Preprocessing for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data_encoded['Cuisines'])

# Function to recommend restaurants based on user preferences
def recommend_restaurants(user_preferences, top_n=5):
    user_preferences_vector = tfidf.transform([user_preferences])
    cosine_similarities = linear_kernel(user_preferences_vector, tfidf_matrix).flatten()
    
    # Get the indices of the top N similar restaurants
    top_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    
    # Select the top restaurants and include their similarity scores
    recommended_restaurants = data_encoded.iloc[top_indices].copy()
    recommended_restaurants['similarity_score'] = cosine_similarities[top_indices]
    
    # Sort by similarity score and then by aggregate rating for better results
    recommended_restaurants = recommended_restaurants.sort_values(by=['similarity_score', 'Aggregate rating'], ascending=False)

    # Return the relevant columns
    return recommended_restaurants[['Restaurant Name', 'City', 'Aggregate rating', 'similarity_score']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_preferences = request.form['preferences']
    recommended_restaurants = recommend_restaurants(user_preferences)
    return render_template('result.html', restaurants=recommended_restaurants)

if __name__ == '__main__':
    app.run(debug=True)
