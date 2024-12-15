import re
import string
import pandas as pd

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ''.join([i for i in text if not i.isdigit()])
        text = ' '.join(text.split())
        text = text.encode('ascii', 'ignore').decode('ascii')
    else:
        text = ''
    return text

def preprocess_dataset(df):
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    return df

data = pd.read_excel('Train_Task_A.xlsx', engine='openpyxl')
preprocessed_data = preprocess_dataset(data)
preprocessed_data.to_csv('cleaned.csv', index=False)
