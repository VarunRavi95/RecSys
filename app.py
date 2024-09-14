from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Set up CORS to allow requests from any origin
CORS(app, resources={r"/recommend": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def after_request(response):
    # Add necessary CORS headers for handling preflight and actual requests
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Load data
df = pd.read_csv('processed_listings_with_original_descriptions.csv')

# Load the saved embeddings
embeddings = np.load('gtr_t5_large_embeddings.npy').astype('float32')

# Normalize embeddings
faiss.normalize_L2(embeddings)

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/gtr-t5-large')

# Set up FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Define the recommendation endpoint
@app.route('/recommend', methods=['POST', 'OPTIONS'])
def recommend():
    if request.method == 'OPTIONS':
        # Send an empty response with the appropriate headers
        return jsonify(status="OK"), 200

    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    query_embedding = model.encode([query]).astype('float32')

    # Normalize the query embedding
    faiss.normalize_L2(query_embedding)

    # Ensure query embedding matches the dimension of index embeddings
    if query_embedding.shape[1] != embeddings.shape[1]:
        return jsonify({'error': 'Query embedding dimension does not match embeddings dimension'}), 400

    # Perform the search using FAISS
    distances, indices = index.search(query_embedding, 20)

    # Handle NaN values in the DataFrame by converting them to None
    recommendations = df.iloc[indices[0]].replace({np.nan: None}).to_dict(orient='records')
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)