import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from tqdm import tqdm

# Custom Dataset class
class FauxHateDataset(Dataset):
    def __init__(self, embeddings, labels_fake, labels_hate):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels_fake = torch.LongTensor(labels_fake)
        self.labels_hate = torch.LongTensor(labels_hate)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'fake_label': self.labels_fake[idx],
            'hate_label': self.labels_hate[idx]
        }

# Multi-Task Learning Model
class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(MultiTaskClassifier, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific layers
        self.fake_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.hate_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        fake_output = self.fake_classifier(shared_features)
        hate_output = self.hate_classifier(shared_features)
        return fake_output, hate_output

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0
    all_fake_preds, all_hate_preds = [], []
    all_fake_labels, all_hate_labels = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False):
            embeddings = batch['embedding'].to(device)
            fake_labels = batch['fake_label'].to(device)
            hate_labels = batch['hate_label'].to(device)
            
            optimizer.zero_grad()
            fake_output, hate_output = model(embeddings)
            
            # Calculate losses
            fake_loss = criterion(fake_output, fake_labels)
            hate_loss = criterion(hate_output, hate_labels)
            total_loss = fake_loss + hate_loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_fake_preds, val_hate_preds = [], []
        val_fake_labels, val_hate_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False):
                embeddings = batch['embedding'].to(device)
                fake_output, hate_output = model(embeddings)
                
                val_fake_preds.extend(torch.argmax(fake_output, dim=1).cpu().numpy())
                val_hate_preds.extend(torch.argmax(hate_output, dim=1).cpu().numpy())
                val_fake_labels.extend(batch['fake_label'].numpy())
                val_hate_labels.extend(batch['hate_label'].numpy())
        
        # Calculate validation metrics
        fake_acc = accuracy_score(val_fake_labels, val_fake_preds)
        hate_acc = accuracy_score(val_hate_labels, val_hate_preds)
        avg_acc = (fake_acc + hate_acc) / 2
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Average Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Fake News Detection Accuracy: {fake_acc:.4f}')
        print(f'Hate Speech Detection Accuracy: {hate_acc:.4f}')
        
        # Save best model
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            torch.save(model.state_dict(), 'best_multitask_model.pt')
        
        # Collect all predictions for classification report at the end
        all_fake_preds.extend(val_fake_preds)
        all_hate_preds.extend(val_hate_preds)
        all_fake_labels.extend(val_fake_labels)
        all_hate_labels.extend(val_hate_labels)
    
    # Classification report after all epochs
    print("\nClassification Report (Fake News Detection):")
    print(classification_report(all_fake_labels, all_fake_preds))
    
    print("\nClassification Report (Hate Speech Detection):")
    print(classification_report(all_hate_labels, all_hate_preds))

# Main execution
def main():
    # Load embeddings
    with open('BERT_whole_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Prepare data
    embeddings = np.array([item['embedding'] for item in data])
    fake_labels = np.array([item['Fake'] for item in data])
    hate_labels = np.array([item['Hate'] for item in data])
    
    # Split data
    X_train, X_val, y_train_fake, y_val_fake, y_train_hate, y_val_hate = train_test_split(
        embeddings, fake_labels, hate_labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = FauxHateDataset(X_train, y_train_fake, y_train_hate)
    val_dataset = FauxHateDataset(X_val, y_val_fake, y_val_hate)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train model
    input_dim = embeddings.shape[1]  # BERT embedding dimension
    model = MultiTaskClassifier(input_dim)
    
    # Train the model
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
