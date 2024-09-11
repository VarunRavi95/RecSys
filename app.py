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

# Load data and model
df = pd.read_csv('processed_listings_with_original_descriptions.csv')

# Load the saved hybrid embeddings
embeddings = np.load('hybrid_embeddings.npy').astype('float32')

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Set up FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Define the recommendation endpoint
@app.route('/recommend', methods=['POST', 'OPTIONS'])
def recommend():
    if request.method == 'OPTIONS':
        # Send an empty response with the appropriate headers
        return jsonify(status="OK"), 200

    data = request.json
    query = data.get('query')
    query_embedding = model.encode([query]).astype('float32')

    # Ensure query embedding matches the dimension of index embeddings (768)
    target_dim = embeddings.shape[1]
    if query_embedding.shape[1] < target_dim:
        query_embedding = np.pad(query_embedding, ((0, 0), (0, target_dim - query_embedding.shape[1])), 'constant')
    elif query_embedding.shape[1] > target_dim:
        query_embedding = query_embedding[:, :target_dim]

    # Perform the search using FAISS
    distances, indices = index.search(query_embedding, 20)

    # Handle NaN values in the DataFrame by converting them to None
    recommendations = df.iloc[indices[0]].replace({np.nan: None}).to_dict(orient='records')
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
