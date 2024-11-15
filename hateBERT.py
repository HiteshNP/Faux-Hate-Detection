import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import pickle
from tqdm import tqdm

# Initialize the tokenizer and model for HateBERT
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
model = AutoModel.from_pretrained("GroNLP/hateBERT")

def preprocess_data(tweets, max_length=178):
    encoding = tokenizer(
        tweets,
        padding=True,          # Pad all tweets to the same length
        truncation=True,       # Truncate longer tweets
        max_length=max_length, # Adjust this based on average tweet length
        return_tensors='pt'    # Return PyTorch tensors
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask

def get_tweet_embeddings(input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        # Get the embeddings of the [CLS] token
        tweet_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings
        return tweet_embeddings

def generate_embeddings(df):
    embeddings_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating HateBERT Embeddings"):
        tweet = row['Tweet']
        fake_label = row['Fake']
        hate_label = row['Hate']
        tweet_id = row['ID']
        
        # Preprocess and get embeddings
        input_ids, attention_mask = preprocess_data([tweet])  # List format for single tweet
        tweet_embedding = get_tweet_embeddings(input_ids, attention_mask)
        
        # Convert tensor to numpy array and store with ID and labels
        embeddings_list.append({
            'ID': tweet_id,
            'Fake': fake_label,
            'Hate': hate_label,
            'embedding': tweet_embedding[0].cpu().numpy()
        })
    
    return embeddings_list

# Load your dataset
df = pd.read_csv('val_data.csv')  # Adjust the file name as necessary

# Generate embeddings for the current subset
embeddings = generate_embeddings(df)

# Save embeddings to a .pkl file
with open('HateBERT_val_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print(f"Embeddings saved to 'HateBERT_val_embeddings.pkl'")
