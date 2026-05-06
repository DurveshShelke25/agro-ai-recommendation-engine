from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # NEW IMPORT
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle

app = FastAPI(
    title="Avdhut Agro AI Recommendation Engine",
    version="1.0"
)

# NEW: Configure CORS to allow your HTML file to fetch data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any webpage to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 2. Load the Model and Data globally so it only loads once when the server starts
try:
    model = tf.keras.models.load_model('ecommerce_model.h5')
    with open('product_map.pkl', 'rb') as f:
        product_map = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have run train_model.py first!")

# 3. Define the response format using Pydantic
class RecommendationResponse(BaseModel):
    product_name: str
    interest_score: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Recommendation API. Go to /docs to test the endpoints."}

# 4. Create the API Endpoint
@app.get("/recommend/{user_id}", response_model=list[RecommendationResponse])
def get_recommendations(user_id: int, num_recs: int = 2):
    """
    Pass a user_id to get top product recommendations for that user.
    """
    if user_id < 0 or user_id >= 100: # We simulated 100 users (0-99)
        raise HTTPException(status_code=404, detail="User not found")

    all_product_ids = np.array(list(product_map.keys()))
    user_ids_array = np.array([user_id] * len(all_product_ids))
    
    # Run the AI prediction
    predictions = model.predict([user_ids_array, all_product_ids], verbose=0).flatten()
    
    # Get top N results
    top_indices = predictions.argsort()[-num_recs:][::-1]
    
    # Format the results into a JSON-friendly list
    results = []
    for idx in top_indices:
        prod_id = all_product_ids[idx]
        results.append({
            "product_name": product_map[prod_id],
            "interest_score": round(float(predictions[idx]), 2) # Round to 2 decimals
        })
        
    return results