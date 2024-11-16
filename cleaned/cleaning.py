import re
import string
import pandas as pd

def preprocess_text(text):
    """
    Preprocess the input text by performing the following steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove mentions (@user)
    4. Remove hashtags
    5. Remove punctuation
    6. Remove digits
    7. Remove extra whitespace (leaving only a single space between words)
    8. Handle encoding issues
    """
    # Check if the input is a non-null string
    if pd.notna(text) and isinstance(text, str):
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
    
    return text

def preprocess_dataset(df):
    """
    Preprocess the entire dataset by applying the preprocess_text function to the 'Tweet' column.
    """
    # Apply the text preprocessing
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    
    # Explicitly set 'N/A' to Target and Severity if they are empty after cleaning
    df['Fake'] = df['Fake'].apply(lambda x: 'N/A' if pd.isna(x) or x == '' else x)
    df['Hate'] = df['Hate'].apply(lambda x: 'N/A' if pd.isna(x) or x == '' else x)
    
    return df

# Read the data from the Excel file, treating empty cells as strings
data = pd.read_excel(
    r'C:\Users\Ankith Jain\Desktop\FAUX\validation\Val_Task_A.xlsx', 
    engine='openpyxl', 
    dtype={'Fake': str, 'Hate': str}
)

# Preprocess only the 'Tweet' column, leaving 'Target' and 'Severity' intact
preprocessed_data = preprocess_dataset(data)

# Save the preprocessed dataset to a new CSV file
preprocessed_data.to_csv(r'C:\Users\Ankith Jain\Desktop\FAUX\validation\cleaned_val_A.csv', index=False)
