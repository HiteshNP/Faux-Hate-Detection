import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel  # Import XLM-R tokenizer and model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bars

# Custom Dataset
class HateFakeDataset(Dataset):
    def __init__(self, texts, hate_labels, fake_labels, tokenizer, max_len=128):
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

# Multi-task Model
class MultiTaskHateFakeClassifier(nn.Module):
    def __init__(self, xlm_model, dropout_rate=0.3):
        super().__init__()
        self.xlm = xlm_model
        hidden_size = xlm_model.config.hidden_size

        # Common layers
        self.dropout = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(hidden_size, 512)
        self.dense2 = nn.Linear(512, 256)

        # Task-specific layers
        self.hate_classifier = nn.Linear(256, 2)
        self.fake_classifier = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token

        x = self.dropout(pooled_output)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)

        hate_output = F.softmax(self.hate_classifier(x), dim=1)
        fake_output = F.softmax(self.fake_classifier(x), dim=1)

        return hate_output, fake_output

# Training function with early stopping
def train_model(model, train_loader, val_loader, device, num_epochs=30, learning_rate=2e-5, patience=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_val_accuracy = 0
    early_stop_counter = 0
    best_epoch = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hate_labels = batch['hate_label'].to(device)
            fake_labels = batch['fake_label'].to(device)

            optimizer.zero_grad()
            hate_output, fake_output = model(input_ids, attention_mask)
            hate_loss = criterion(hate_output, hate_labels)
            fake_loss = criterion(fake_output, fake_labels)
            total_batch_loss = hate_loss + fake_loss

            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()

        model.eval()
        val_loss = 0
        all_hate_preds, all_fake_preds = [], []
        all_hate_labels, all_fake_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                hate_labels = batch['hate_label'].to(device)
                fake_labels = batch['fake_label'].to(device)

                hate_output, fake_output = model(input_ids, attention_mask)
                hate_loss = criterion(hate_output, hate_labels)
                fake_loss = criterion(fake_output, fake_labels)
                val_loss += (hate_loss + fake_loss).item()

                all_hate_preds.extend(torch.argmax(hate_output, dim=1).cpu().numpy())
                all_fake_preds.extend(torch.argmax(fake_output, dim=1).cpu().numpy())
                all_hate_labels.extend(hate_labels.cpu().numpy())
                all_fake_labels.extend(fake_labels.cpu().numpy())

        # Calculate validation accuracy
        val_hate_accuracy = (torch.argmax(hate_output, dim=1) == hate_labels).float().mean().item()
        val_fake_accuracy = (torch.argmax(fake_output, dim=1) == fake_labels).float().mean().item()
        val_accuracy = (val_hate_accuracy + val_fake_accuracy) / 2

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')

        print("Classification Report for Hate Speech Task:")
        print(classification_report(all_hate_labels, all_hate_preds))
        print("Classification Report for Fake News Task:")
        print(classification_report(all_fake_labels, all_fake_preds))

        # Check if this is the best epoch
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_model_wts = model.state_dict()  # Save the best model weights

            # Reset early stop counter since we have a new best model
        #     early_stop_counter = 0
        # else:
        #     early_stop_counter += 1

        # # Early stopping check
        # if early_stop_counter >= patience:
        #     print(f"Early stopping triggered at epoch {epoch+1}.")
        #     break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_XLM_model.pt')  # Save the best model

    print(f"Training completed. Best epoch: {best_epoch+1} with validation accuracy: {best_val_accuracy:.4f}")
    print("Best Classification Report for Hate Speech Task:")
    print(classification_report(all_hate_labels, all_hate_preds))
    print("Best Classification Report for Fake News Task:")
    print(classification_report(all_fake_labels, all_fake_preds))

# Usage example
def main():
    # Load and preprocess your data
    df_train = pd.read_csv('A_trans.csv')  # Load whole training data
    df_val = pd.read_csv('A_val_trans.csv')  # Load validation data

    # Clean and preprocess the labels
    for df in [df_train, df_val]:
        df.dropna(subset=['Hate', 'Fake', 'Tweet'], inplace=True)
        df['Hate'] = df['Hate'].astype(int)
        df['Fake'] = df['Fake'].astype(int)
        df = df[(df['Hate'].isin([0, 1])) & (df['Fake'].isin([0, 1]))]

    # Extract texts and labels for training and validation
    texts_train = df_train['Tweet'].values
    hate_labels_train = df_train['Hate'].values
    fake_labels_train = df_train['Fake'].values

    texts_val = df_val['Tweet'].values
    hate_labels_val = df_val['Hate'].values
    fake_labels_val = df_val['Fake'].values

    # Initialize tokenizer and model for XLM-R
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')  # Use XLM-R tokenizer
    xlm_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')  # Use XLM-R model

    # Create datasets
    train_dataset = HateFakeDataset(texts_train, hate_labels_train, fake_labels_train, tokenizer)
    val_dataset = HateFakeDataset(texts_val, hate_labels_val, fake_labels_val, tokenizer)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskHateFakeClassifier(xlm_model).to(device)

    # Train model
    train_model(model, train_loader, val_loader, device)

# Run the main function
if __name__ == '__main__':
    main()
