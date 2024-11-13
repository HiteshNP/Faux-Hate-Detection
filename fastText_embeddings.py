import pandas as pd
import fasttext
import numpy as np
import pickle
from tqdm import tqdm

# Step 1: Load the training dataset
df = pd.read_csv('small_data.csv')  # Assuming 'small_data.csv' has 'FAUX' and 'Tweet' columns

# Step 2: Load the pretrained FastText model for Kannada (cc.kn.300.bin)
fasttext_model = fasttext.load_model('cc.kn.300.bin')  # Ensure this model file is in the directory or specify the path

# Step 3: Function to get FastText embeddings for each tweet
def get_fasttext_embeddings(text):
    tokens = text.split()  # Split the tweet into words
    vectors = [fasttext_model.get_word_vector(word) for word in tokens if word in fasttext_model.words]
    if vectors:
        return np.mean(vectors, axis=0)  # Take the average of all word vectors in the tweet
    else:
        return np.zeros(300)  # Return a zero vector if no valid words found

# Step 4: Function to generate embeddings for the dataset
def generate_embeddings(df):
    embeddings_data = []
    print("Generating FastText embeddings...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        tweet = row['Tweet']
        faux_label = row['FAUX']  # This is the combined label
        embedding = get_fasttext_embeddings(tweet)
        embeddings_data.append({
            'FAUX': faux_label,  # Keep the combined label
            'Tweet': tweet,
            'embedding': embedding
        })
    return embeddings_data

# Step 5: Generate embeddings for the dataset
embeddings_data = generate_embeddings(df)

# Step 6: Save embeddings as a .pkl file
with open('Test_Fasttext_embeddings.pkl', 'wb') as embeddings_file:
    pickle.dump(embeddings_data, embeddings_file)

print("Embeddings saved successfully.")
