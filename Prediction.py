import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bars

# Custom Dataset for Test Data
class HateFakeDatasetTest(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

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
            'attention_mask': encoding['attention_mask'].flatten()
        }

# Multi-task Model Definition (same as before)
class MultiTaskHateFakeClassifier(nn.Module):
    def __init__(self, bert_model, dropout_rate=0.3):
        super().__init__()
        self.bert = bert_model
        hidden_size = bert_model.config.hidden_size

        # Common layers
        self.dropout = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(hidden_size, 512)
        self.dense2 = nn.Linear(512, 256)

        # Task-specific layers
        self.hate_classifier = nn.Linear(256, 2)
        self.fake_classifier = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token

        x = self.dropout(pooled_output)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)

        hate_output = F.softmax(self.hate_classifier(x), dim=1)
        fake_output = F.softmax(self.fake_classifier(x), dim=1)

        return hate_output, fake_output

# Load the saved model
def load_model(model_path, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    model = MultiTaskHateFakeClassifier(bert_model).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, tokenizer

# Prediction function
def make_predictions(model, tokenizer, test_loader, device):
    all_predictions = []
    with torch.no_grad():
        # Use tqdm for progress bar with estimated time left
        for batch in tqdm(test_loader, desc="Making Predictions", leave=False, ncols=100, dynamic_ncols=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            hate_output, fake_output = model(input_ids, attention_mask)

            # Get predicted classes (indices with max probability)
            hate_preds = torch.argmax(hate_output, dim=1).cpu().numpy()
            fake_preds = torch.argmax(fake_output, dim=1).cpu().numpy()

            all_predictions.extend(zip(hate_preds, fake_preds))
    return all_predictions

# Main function for making predictions and saving to CSV
def main():
    # Load the test data
    df_test = pd.read_csv('Test_cleaned.csv')

    # Initialize tokenizer and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model, tokenizer = load_model('best_model.pt', device)

    # Create the test dataset and DataLoader
    test_dataset = HateFakeDatasetTest(df_test['Tweet'].values, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Make predictions
    predictions = make_predictions(model, tokenizer, test_loader, device)

    # Prepare the result DataFrame
    df_test['Predicted_Hate'] = [pred[0] for pred in predictions]
    df_test['Predicted_Fake'] = [pred[1] for pred in predictions]

    # Save predictions to CSV
    df_test[['Id', 'Predicted_Hate', 'Predicted_Fake']].to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'.")

# Run the main function
if __name__ == '__main__':
    main()
