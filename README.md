# Real Estate Listing Recommender

This project showcases a Real Estate Listing Recommender application using a content-based filtering approach with a sentence transformer model. The application allows users to input queries and receive relevant real estate listings based on their descriptions and other textual data.

## Demo

<p align="center">
  <img src="./demo-gif.gif" alt="Demo GIF">
</p>

## Features

- **Content-Based Filtering:** Recommends listings based on the semantic similarity of the user's query and listing descriptions using a sentence transformer model.
- **Interactive Frontend:** A React frontend with a clean, professional look and feel, showcasing the recommendations in a user-friendly interface.
- **Backend API:** A Flask backend serving the recommendation results, optimized for performance using FAISS for similarity search.
- **Deployment:** The application is set up for local testing and is demonstrated via a hosted GIF in the README.

## Setup and Installation

### Prerequisites

- **Node.js** and **npm** for frontend.
- **Python** for backend.
- **Git** for version control.

### Backend Setup (Flask)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>

2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Start the Flask server:
   ```bash
   python app.py

### Frontend Setup (React)
   
1. Install dependencies:
   ```bash
   npm install

2. Navigate to the app folder:
   ```bash
   cd my-react-app

2. Start the React application:
   ```bash
   npm start

## Key Components: 


1. Backend:

	•	Flask API: Handles requests and processes recommendations using FAISS for efficient similarity search.
	•	Sentence Transformer Model: Utilizes the all-mpnet-base-v2 model to embed listing descriptions and user queries.

2. Frontend:

	•	React Application: Provides a user interface for inputting queries and displaying recommendations.
	•	Styling: CSS for a modern, professional look with features like rounded edges and card sliders for a better UX.

## Development Process

1.	Data Preprocessing: Cleaned and merged relevant textual data from the Airbnb dataset for the Stockholm region.
2.	Model Training and Embedding: Created embeddings using a powerful sentence transformer and indexed them with FAISS for fast retrieval.
3.	Building the Frontend: Designed a responsive UI with React and CSS to display recommendations.

## Future Enhancements: 

	•	Additional Filters: Implement user filters for more refined recommendations.
	•	Expanded Dataset: Integrate more diverse real estate datasets for broader recommendations.
	•	Scalable Deployment: Move to cloud-based deployment solutions for better scalability and availability.
