import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm

# Custom Dataset class
class FauxHateDataset(Dataset):
    def __init__(self, embeddings, labels_target, labels_severity):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels_target = torch.LongTensor(labels_target)
        self.labels_severity = torch.LongTensor(labels_severity)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'target_label': self.labels_target[idx],
            'severity_label': self.labels_severity[idx]
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
        self.target_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3)  # (Target)
        )
        
        self.severity_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3)  #  (Severity)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        target_output = self.target_classifier(shared_features)
        severity_output = self.severity_classifier(shared_features)
        return target_output, severity_output

# Training function
def train_model(model, train_loader, val_loader, num_epochs=1000, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0
    # all_target_preds, all_severity_preds = [], []
    # all_target_labels, all_severity_labels = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False):
            embeddings = batch['embedding'].to(device)
            target_labels = batch['target_label'].to(device)
            severity_labels = batch['severity_label'].to(device)
            
            optimizer.zero_grad()
            target_output, severity_output = model(embeddings)
            
            # Calculate losses
            target_loss = criterion(target_output, target_labels)
            severity_loss = criterion(severity_output, severity_labels)
            total_loss = target_loss + severity_loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_target_preds, val_severity_preds = [], []
        val_target_labels, val_severity_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False):
                embeddings = batch['embedding'].to(device)
                target_output, severity_output = model(embeddings)
                
                val_target_preds.extend(torch.argmax(target_output, dim=1).cpu().numpy())
                val_severity_preds.extend(torch.argmax(severity_output, dim=1).cpu().numpy())
                val_target_labels.extend(batch['target_label'].numpy())
                val_severity_labels.extend(batch['severity_label'].numpy())
        
        # Calculate validation metrics
        target_acc = accuracy_score(val_target_labels, val_target_preds)
        severity_acc = accuracy_score(val_severity_labels, val_severity_preds)
        avg_acc = (target_acc + severity_acc) / 2
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Average Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Target Detection Accuracy: {target_acc:.4f}')
        print(f'Severity Detection Accuracy: {severity_acc:.4f}')
        
        # Save best model
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            torch.save(model.state_dict(), 'best_multitask_model.pt')
        
    #     # Collect all predictions for classification report at the end
    #     all_target_preds.extend(val_target_preds)
    #     all_severity_preds.extend(val_severity_preds)
    #     all_target_labels.extend(val_target_labels)
    #     all_severity_labels.extend(val_severity_labels)
    # print(f"Length of target labels: {len(all_target_labels)}")
    # print(f"Length of target predictions: {len(all_target_preds)}")
    # print(f"Length of severity labels: {len(all_severity_labels)}")
    # print(f"Length of severity predictions: {len(all_severity_preds)}")

    # Classification report after all epochs
    print("\nClassification Report (Target Detection):")
    print(classification_report(val_target_labels, val_target_preds))
    
    print("\nClassification Report (Severity Detection):")
    print(classification_report(val_severity_labels, val_severity_preds))

def main():
    # Load embeddings
    with open(r'C:\Users\Ankith Jain\Desktop\FAUX\HateBert\B_hate.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Initialize LabelEncoders
    target_encoder = LabelEncoder()
    severity_encoder = LabelEncoder()
    
    # Fit encoders and transform the labels
    data['Target'] = target_encoder.fit_transform(data['Target'])
    data['Severity'] = severity_encoder.fit_transform(data['Severity'])
    
    # Filter out rows with NaN values (if any)
    data = data[data['Target'].notna()]
    data = data[data['Severity'].notna()]
    
    print(f"Dataset shape: {data.shape}")

    # Extract embeddings and labels
    embeddings = np.array([item['Embeddings'] for _, item in data.iterrows()])
    target_labels = np.array(data['Target'])
    severity_labels = np.array(data['Severity'])

    # Split data
    X_train, X_val, y_train_target, y_val_target, y_train_severity, y_val_severity = train_test_split(
        embeddings, target_labels, severity_labels, test_size=0.2, random_state=42)
    
    print(X_train.shape)
    print(X_val.shape)
    print(y_train_target.shape)
    print(y_val_target.shape)

    # Create datasets
    train_dataset = FauxHateDataset(X_train, y_train_target, y_train_severity)
    val_dataset = FauxHateDataset(X_val, y_val_target, y_val_severity)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train model
    input_dim = embeddings.shape[1]  # Embedding dimension
    model = MultiTaskClassifier(input_dim)
    train_model(model, train_loader, val_loader, num_epochs=100)

if __name__ == '__main__':
    main()
