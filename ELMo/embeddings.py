import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import os
import ast

# Load the ELMo model from TensorFlow Hub
elmo = hub.load("https://tfhub.dev/google/elmo/3")

# Function to generate ELMo embeddings for a list of sentences
def get_elmo_embeddings(texts):
    # Join the list of word tokens into sentences
    sentences = [' '.join(text) for text in texts]
    # Run the ELMo model on the input texts
    embeddings = elmo.signatures['default'](tf.constant(sentences))['elmo']
    # Calculate the mean of the embeddings across all words for each sentence
    return tf.reduce_mean(embeddings, axis=1).numpy()

# Function to process a single CSV file
def process_embeddings(df, batch_size=16):
    embeddings_list = []
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    
    for i in range(total_batches):
        start = i * batch_size
        end = min(start + batch_size, len(df))
        batch_texts = df['Tweet'].iloc[start:end].apply(ast.literal_eval).tolist()
        embeddings = get_elmo_embeddings(batch_texts)
        embeddings_list.extend(embeddings.tolist())
        
        # Print progress
        print(f"Processed batch {i + 1}/{total_batches} (Rows {start + 1} to {end})")
    
    return embeddings_list

# Function to process all CSV files in a directory
def process_all_files(input_dir, output_dir, batch_size=16):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing file: {filename}")
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Generate embeddings and add to DataFrame
            df['ELMo_Embeddings'] = process_embeddings(df, batch_size)
            
            # Create output filename for embeddings
            output_filename = f"{os.path.splitext(filename)[0]}_embeddings.pkl"
            output_file_path = os.path.join(output_dir, output_filename)
            
            # Save the DataFrame with embeddings to a pickle file
            df.to_pickle(output_file_path)
            
            print(f"Embeddings saved to: {output_file_path}")

# Specify input and output directories
input_dir = r"C:\Users\Ankith Jain\Desktop\faux hate\Embeddings\train_test_data"
output_dir = r"C:\Users\Ankith Jain\Desktop\faux hate\Embeddings\train_test_data"

# Process all files in the input directory and save embeddings in the output directory
process_all_files(input_dir, output_dir)

print("All files processed and ELMo embeddings generated successfully!")
