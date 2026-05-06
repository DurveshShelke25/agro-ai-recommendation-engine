import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import pickle

print("1. Simulating Supabase Data Pull...")
# Simulating interaction data for the core product line
np.random.seed(42)
data = {
    'user_id': np.random.randint(0, 100, 500), 
    'product_id': np.random.choice([0, 1, 2, 3], 500), 
    'rating': np.random.randint(1, 6, 500) 
}
df = pd.DataFrame(data)

# Product mapping dictionary
product_map = {
    0: "Amino Casil",
    1: "Avdhut Sweeper",
    2: "Bio Fighter",
    3: "Root & Shoot"
}

# Save the product map so the prediction script can use it
with open('product_map.pkl', 'wb') as f:
    pickle.dump(product_map, f)

num_users = df['user_id'].nunique()
num_products = df['product_id'].nunique()

X = df[['user_id', 'product_id']].values
y = df['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Building the Neural Collaborative Filtering Model...")
embedding_size = 32

user_input = layers.Input(shape=(1,), name='user_input')
product_input = layers.Input(shape=(1,), name='product_input')

user_embedding = layers.Embedding(input_dim=100, output_dim=embedding_size)(user_input)
product_embedding = layers.Embedding(input_dim=4, output_dim=embedding_size)(product_input)

user_vec = layers.Flatten()(user_embedding)
product_vec = layers.Flatten()(product_embedding)

concat = layers.Concatenate()([user_vec, product_vec])

dense_1 = layers.Dense(128, activation='relu')(concat)
dropout_1 = layers.Dropout(0.2)(dense_1)
dense_2 = layers.Dense(64, activation='relu')(dropout_1)
output = layers.Dense(1, name='output')(dense_2)

model = Model(inputs=[user_input, product_input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='mean_squared_error', 
              metrics=['mae'])

print("3. Training the Model...")
model.fit(
    x=[X_train[:, 0], X_train[:, 1]], 
    y=y_train,
    batch_size=32,
    epochs=10,
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
    verbose=1
)

print("4. Saving the Model...")
model.save('ecommerce_model.h5')
print("Training complete. 'ecommerce_model.h5' and 'product_map.pkl' generated.")