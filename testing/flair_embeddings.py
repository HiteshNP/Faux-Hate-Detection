import pandas as pd
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import torch
import pickle
import numpy as np

# Load your dataset
df = pd.read_csv(r"C:\Users\Ankith Jain\Desktop\FAUX\cleaned_A.csv")  # Replace with the path to your dataset

# Remove rows with NaN or empty strings in the 'Tweet' column
df = df.dropna(subset=['Tweet'])
df = df[df['Tweet'].str.strip() != '']  # Keep only non-empty strings

# # Remove rows where both 'Target' and 'Severity' are NaN or empty
# df = df.dropna(subset=['Fake', 'Hate'], how='all')
# df = df[(df['Fake'].notna()) | (df['Hate'].notna())]  # Keep rows where at least one label is non-empty

print("ok")
# Initialize the Flair embeddings (using 'bert-base-uncased' as an example)
embedding = TransformerWordEmbeddings('bert-base-uncased')

# Function to generate embeddings for each tweet
def generate_embeddings(text):
    if isinstance(text, str) and text.strip():  # Check if text is a valid non-empty string
        sentence = Sentence(text)
        embedding.embed(sentence)
        if len(sentence) > 0:  # Check if there are tokens in the sentence
            # Get the mean of embeddings of all tokens to represent the whole sentence
            embeddings = torch.mean(torch.stack([token.embedding for token in sentence]), dim=0)
            return embeddings.cpu().detach().numpy()
    # Return a zero vector if text is NaN, empty, or has no tokens
    return np.zeros(embedding.embedding_length)

# Generate embeddings for each tweet in the dataset
df['Embeddings'] = df['Tweet'].apply(generate_embeddings)

# Select only the labels and embeddings columns
output_data = df[['Id','Hate', 'Fake', 'Embeddings']]

# Save the embeddings and labels to a pickle file
with open(r"C:\Users\Ankith Jain\Desktop\FAUX\Task-B\A_flair.pkl", "wb") as f:
    pickle.dump(output_data, f)

print("Labels and embeddings generated and saved!")