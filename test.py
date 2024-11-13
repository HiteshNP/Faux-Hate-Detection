import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load Dataset
df = pd.read_csv('modified_small_data.csv')

# Load tokenizer and model (using mBERT or any other multilingual model)
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)
embedding_model.eval()  # Set model to evaluation mode

# Generate embeddings for each tweet
def generate_embeddings(df):
    embeddings = []
    with torch.no_grad():
        for tweet in tqdm(df['Tweet'], desc="Generating embeddings"):
            inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = embedding_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()  # CLS token embedding
            embeddings.append(cls_embedding.numpy())
    return embeddings

# Generate embeddings and add to DataFrame
df['embeddings'] = generate_embeddings(df)

# Split dataset
X = torch.tensor(df['embeddings'].tolist())
y_hate = torch.tensor(df['Hate'].values)
y_fake = torch.tensor(df['Fake'].values)
X_train, X_test, y_hate_train, y_hate_test, y_fake_train, y_fake_test = train_test_split(X, y_hate, y_fake, test_size=0.2, random_state=42)

# Define Multi-Task Model
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskModel, self).__init__()
        self.shared_layer = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.hate_head = nn.Linear(128, 1)
        self.fake_head = nn.Linear(128, 1)
    
    def forward(self, x):
        shared_output = self.relu(self.shared_layer(x))
        hate_output = torch.sigmoid(self.hate_head(shared_output))
        fake_output = torch.sigmoid(self.fake_head(shared_output))
        return hate_output, fake_output

# Initialize model, loss, and optimizer
input_dim = X.shape[1]  # CLS embedding size (e.g., 768 for BERT)
model = MultiTaskModel(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for i in tqdm(range(0, X_train.size(0), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_X = X_train[i:i+batch_size].to(device).float()
        batch_y_hate = y_hate_train[i:i+batch_size].to(device).float().unsqueeze(1)
        batch_y_fake = y_fake_train[i:i+batch_size].to(device).float().unsqueeze(1)
        
        # Forward pass
        hate_pred, fake_pred = model(batch_X)
        
        # Calculate loss for both tasks
        loss_hate = criterion(hate_pred, batch_y_hate)
        loss_fake = criterion(fake_pred, batch_y_fake)
        loss = loss_hate + loss_fake  # Combined loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / (X_train.size(0) / batch_size)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    hate_preds, fake_preds = [], []
    for i in range(0, X_test.size(0), batch_size):
        batch_X = X_test[i:i+batch_size].to(device).float()
        hate_pred, fake_pred = model(batch_X)
        hate_preds.extend(hate_pred.cpu().numpy().flatten())
        fake_preds.extend(fake_pred.cpu().numpy().flatten())
    
    # Binarize predictions
    hate_preds = [1 if x >= 0.5 else 0 for x in hate_preds]
    fake_preds = [1 if x >= 0.5 else 0 for x in fake_preds]
    
    # Calculate accuracy for each task
    hate_accuracy = accuracy_score(y_hate_test, hate_preds)
    fake_accuracy = accuracy_score(y_fake_test, fake_preds)
    
    # Calculate Macro F1 score for each task
    hate_f1 = f1_score(y_hate_test, hate_preds, average="macro")
    fake_f1 = f1_score(y_fake_test, fake_preds, average="macro")
    
    # Print results in percentage
    print(f"Hate Detection Accuracy: {hate_accuracy * 100:.2f}%")
    print(f"Hate Detection Macro F1 Score: {hate_f1 * 100:.2f}%")
    print(f"Fake Detection Accuracy: {fake_accuracy * 100:.2f}%")
    print(f"Fake Detection Macro F1 Score: {fake_f1 * 100:.2f}%")
