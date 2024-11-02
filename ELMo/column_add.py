import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download the NLTK tokenizer if not already downloaded
nltk.download('punkt')

# Load the dataset
df = pd.read_csv(r'C:\Users\Ankith Jain\Desktop\faux hate\Embeddings\cleaned.csv')

# Step 1: Create the Combined_Class column
def classify(row):
    if row['Hate'] == 1 and row['Fake'] == 1:
        return 'Fake-Hate'
    elif row['Hate'] == 1 and row['Fake'] == 0:
        return 'NonFake-Hate'
    elif row['Hate'] == 0 and row['Fake'] == 1:
        return 'Fake-NonHate'
    else:
        return 'NonFake-NonHate'

df['Combined_Class'] = df.apply(classify, axis=1)

# Step 2: Remove the Id, Fake, and Hate columns
# Step 3: Ensure all tweets are strings (to avoid errors during tokenization)
df['Tweet'] = df['Tweet'].astype(str)

# Step 4: Tokenize each tweet at the word level
df['Tokenized_Tweet'] = df['Tweet'].apply(word_tokenize)

df = df.drop(columns=['Id', 'Fake', 'Hate', 'Tweet'])  # Remove Tweet column

# Step 5: Keep only Combined_Class and Tokenized_Tweet
df = df[['Combined_Class', 'Tokenized_Tweet']]

# Step 6: Save the updated DataFrame to a new CSV file
output_path = r'C:\Users\Ankith Jain\Desktop\faux hate\Embeddings\tokenized data\tokenized_dataset.csv'
df.to_csv(output_path, index=False)

print(f"Tokenized data saved to {output_path}")
