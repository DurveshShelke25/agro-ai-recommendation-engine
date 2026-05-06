# Avdhut Agro AI Recommendation Engine 🌾

A full-stack machine learning application that predicts user interest in specialized agricultural products using Neural Collaborative Filtering.

## Overview
This project simulates an e-commerce backend where customer transaction data is used to train a deep learning model. The model calculates latent features between users and products to generate highly accurate, personalized product recommendations via a RESTful API.

## Tech Stack
* **Machine Learning:** TensorFlow, Keras, Pandas, Scikit-Learn
* **Backend API:** FastAPI, Uvicorn, Python
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)

## How to Run Locally

1. **Clone the repository:**
   `git clone https://github.com/YOUR_USERNAME/agro-ai-recommendation-engine.git`
2. **Set up the virtual environment:**
   `python -m venv venv`
   `source venv/bin/activate` (Mac/Linux) or `.\venv\Scripts\activate` (Windows)
3. **Install dependencies:**
   `pip install -r requirements.txt`
4. **Train the model (Generates `.h5` and `.pkl` files):**
   `python train_model.py`
5. **Start the FastAPI server:**
   `uvicorn main:app --reload`
6. **Test the API:** Open `index.html` in your web browser to interact with the model.
