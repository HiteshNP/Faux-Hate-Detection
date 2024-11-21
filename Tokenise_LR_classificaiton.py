import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm

# Custom Dataset
class HateFakeDataset(Dataset):
    def __init__(self, texts, hate_labels, fake_labels, tokenizer, max_len=256):
        self.texts = texts
        self.hate_labels = hate_labels
        self.fake_labels = fake_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        hate_label = self.hate_labels[idx]
        fake_label = self.fake_labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'hate_label': torch.tensor(hate_label, dtype=torch.long),
            'fake_label': torch.tensor(fake_label, dtype=torch.long)
        }

# Function to extract BERT features
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels_hate = []
    labels_fake = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            features.extend(cls_embeddings)
            labels_hate.extend(batch['hate_label'].numpy())
            labels_fake.extend(batch['fake_label'].numpy())

    return np.array(features), np.array(labels_hate), np.array(labels_fake)

# Main function
def main():
    # Load and preprocess your data
    df_train = pd.read_csv('whole_data.csv')
    df_val = pd.read_csv('val_data.csv')

    for df in [df_train, df_val]:
        df.dropna(subset=['Hate', 'Fake', 'Tweet'], inplace=True)
        df['Hate'] = df['Hate'].astype(int)
        df['Fake'] = df['Fake'].astype(int)
        df = df[(df['Hate'].isin([0, 1])) & (df['Fake'].isin([0, 1]))]

    # Extract texts and labels
    texts_train = df_train['Tweet'].values
    hate_labels_train = df_train['Hate'].values
    fake_labels_train = df_train['Fake'].values

    texts_val = df_val['Tweet'].values
    hate_labels_val = df_val['Hate'].values
    fake_labels_val = df_val['Fake'].values

    # Initialize tokenizer and BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)

    # Create datasets and dataloaders
    train_dataset = HateFakeDataset(texts_train, hate_labels_train, fake_labels_train, tokenizer)
    val_dataset = HateFakeDataset(texts_val, hate_labels_val, fake_labels_val, tokenizer)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Extract features
    train_features, train_hate_labels, train_fake_labels = extract_features(bert_model, train_loader, device)
    val_features, val_hate_labels, val_fake_labels = extract_features(bert_model, val_loader, device)

    # Train Logistic Regression models
    hate_classifier = LogisticRegression(max_iter=1000, random_state=42)
    hate_classifier.fit(train_features, train_hate_labels)

    fake_classifier = LogisticRegression(max_iter=1000, random_state=42)
    fake_classifier.fit(train_features, train_fake_labels)

    # Validate models
    hate_preds = hate_classifier.predict(val_features)
    fake_preds = fake_classifier.predict(val_features)

    print("Classification Report for Hate Speech Task:")
    print(classification_report(val_hate_labels, hate_preds))

    print("Classification Report for Fake News Task:")
    print(classification_report(val_fake_labels, fake_preds))

if __name__ == '__main__':
    main()
