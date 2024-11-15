import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report

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

# Training function with early stopping
def train_model(model, train_loader, num_epochs=30, learning_rate=0.001, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_fake = 0
        correct_hate = 0
        total_samples = 0
        
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
            
            # Calculate accuracy
            _, fake_preds = torch.max(fake_output, dim=1)
            _, hate_preds = torch.max(hate_output, dim=1)
            correct_fake += (fake_preds == fake_labels).sum().item()
            correct_hate += (hate_preds == hate_labels).sum().item()
            total_samples += fake_labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        fake_accuracy = correct_fake / total_samples
        hate_accuracy = correct_hate / total_samples
        
        print(f'\nEpoch {epoch + 1}:')
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Fake News Accuracy: {fake_accuracy * 100:.2f}%')
        print(f'Hate Speech Accuracy: {hate_accuracy * 100:.2f}%')
        
        # Early stopping check
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            epochs_without_improvement = 0
            # Optionally save the model here if needed
            torch.save(model.state_dict(), 'best_multitask_model.pt')
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

# Evaluate function
def evaluate_model(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_fake_preds, all_hate_preds = [], []
    all_fake_labels, all_hate_labels = [], []

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=32, shuffle=False), desc="Evaluating"):
            embeddings = batch['embedding'].to(device)
            fake_output, hate_output = model(embeddings)
            
            all_fake_preds.extend(torch.argmax(fake_output, dim=1).cpu().numpy())
            all_hate_preds.extend(torch.argmax(hate_output, dim=1).cpu().numpy())
            all_fake_labels.extend(batch['fake_label'].numpy())
            all_hate_labels.extend(batch['hate_label'].numpy())
    
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
    
    # Use entire dataset
    dataset = FauxHateDataset(embeddings, fake_labels, hate_labels)
    
    # Create dataloader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train model
    input_dim = embeddings.shape[1]  # BERT embedding dimension
    model = MultiTaskClassifier(input_dim)
    
    # Train the model
    train_model(model, train_loader)
    
    # Evaluate the model
    evaluate_model(model, dataset)

if __name__ == "__main__":
    main()
