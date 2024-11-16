import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pickle
import numpy as np

# Load your dataset
df = pd.read_csv(r"C:\Users\Ankith Jain\Desktop\FAUX\Task-B\cleaned_B.csv")  # Replace with the path to your dataset

# Remove rows with NaN or empty strings in the 'Tweet' column
df = df.dropna(subset=['Tweet'])
df = df[df['Tweet'].str.strip() != '']  # Keep only non-empty strings

# # Remove rows where both 'Target' and 'Severity' labels are NaN or empty
# df = df.dropna(subset=['Target', 'Severity'], how='all')
# df = df[(df['Target'].notna()) | (df['Severity'].notna())]  # Keep rows where at least one label is non-empty

print("Data cleaned!")

# Load HateBERT model and tokenizer using Auto classes
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
model = AutoModelForMaskedLM.from_pretrained("GroNLP/hateBERT")

# Function to generate embeddings for each tweet
def generate_embeddings(text):
    if isinstance(text, str) and text.strip():  # Check if text is a valid non-empty string
        # Tokenize the text and get the input ids
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)

        # Generate embeddings using the model (without gradient computation)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)  # Request hidden states

        # Access the hidden states and use the last layer ([-1]) to get the embeddings
        hidden_states = outputs.hidden_states  # This is a tuple of all hidden states
        last_hidden_state = hidden_states[-1]  # The last hidden state is the last layer's embeddings

        # Take the mean of the token embeddings from the last hidden state
        embeddings = last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
    
    # Return a zero vector if text is NaN, empty, or has no tokens
    return np.zeros(model.config.hidden_size)

# Generate embeddings for each tweet in the dataset
df['Embeddings'] = df['Tweet'].apply(generate_embeddings)

# Select only the IDs, labels, and embeddings columns
output_data = df[['Id', 'Target', 'Severity', 'Embeddings']]  # Include Tweet_ID here

# Save the embeddings, IDs, and labels to a pickle file
with open(r"C:\Users\Ankith Jain\Desktop\FAUX\HateBert\B_hate.pkl", "wb") as f:
    pickle.dump(output_data, f)

print("Tweet IDs, labels, and embeddings generated and saved!")
