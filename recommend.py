import numpy as np
import tensorflow as tf
import pickle

# Load the saved model and the product mapping
print("Loading model and product data...")
model = tf.keras.models.load_model('ecommerce_model.h5')

with open('product_map.pkl', 'rb') as f:
    product_map = pickle.load(f)

def get_recommendations(target_user_id, num_recommendations=2):
    """Generates product recommendations for a specific user."""
    
    all_product_ids = np.array(list(product_map.keys()))
    user_ids_array = np.array([target_user_id] * len(all_product_ids))
    
    # Predict the rating the user would give to every product
    predictions = model.predict([user_ids_array, all_product_ids], verbose=0).flatten()
    
    # Sort by highest predicted rating
    top_indices = predictions.argsort()[-num_recommendations:][::-1]
    
    print(f"\n=========================================")
    print(f" TOP RECOMMENDATIONS FOR USER ID: {target_user_id}")
    print(f"=========================================")
    
    for idx in top_indices:
        prod_id = all_product_ids[idx]
        score = predictions[idx]
        print(f"-> {product_map[prod_id]} (Predicted Interest: {score:.2f}/5.0)")
    print(f"=========================================\n")

# Execute the function for a test user
if __name__ == "__main__":
    # Feel free to change the user ID here to test different outputs (0-99)
    test_user = 15
    get_recommendations(test_user)