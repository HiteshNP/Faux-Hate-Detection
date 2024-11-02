import re
import string
import pandas as pd

def preprocess_text(text):
    """
    Preprocess the input text by performing the following steps:
    1. Check if the input is a string
    2. Convert to lowercase
    3. Remove URLs
    4. Remove mentions (@user)
    5. Remove hashtags
    6. Remove punctuation
    7. Remove digits
    8. Remove extra whitespace (leaving only a single space between words)
    9. Handle encoding issues
    """
    # Check if the input is a string
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove digits
        text = ''.join([i for i in text if not i.isdigit()])
        
        # Remove extra whitespace, leaving only single spaces between words
        text = ' '.join(text.split())
        
        # Handle encoding issues
        text = text.encode('ascii', 'ignore').decode('ascii')
    else:
        text = ''
    
    return text

def preprocess_dataset(df):
    """
    Preprocess the entire dataset by applying the preprocess_text function to the 'Tweet' column.
    """
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    return df

# Read the data from the Excel file
data = pd.read_excel('Train_Task_A.xlsx', engine='openpyxl')

# Preprocess the dataset
preprocessed_data = preprocess_dataset(data)

# Save the preprocessed dataset to a new CSV file
preprocessed_data.to_csv('cleaned.csv', index=False)
