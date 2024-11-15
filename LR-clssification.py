import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to load data from pickle files
def load_data(train_file, val_file):
    # Load training data (BERT_whole_embeddings.pkl)
    with open(train_file, 'rb') as f:
        data_train = pickle.load(f)

    # Load validation data (val_embeddings.pkl)
    with open(val_file, 'rb') as f:
        data_val = pickle.load(f)

    # Prepare training data
    embeddings_train = np.array([item['embedding'] for item in data_train])
    fake_labels_train = np.array([item['Fake'] for item in data_train])
    hate_labels_train = np.array([item['Hate'] for item in data_train])

    # Prepare validation data
    embeddings_val = np.array([item['embedding'] for item in data_val])
    fake_labels_val = np.array([item['Fake'] for item in data_val])
    hate_labels_val = np.array([item['Hate'] for item in data_val])

    return embeddings_train, fake_labels_train, hate_labels_train, embeddings_val, fake_labels_val, hate_labels_val

# Function to train and evaluate logistic regression
def logistic_regression_classification(train_embeddings, train_fake_labels, train_hate_labels, val_embeddings, val_fake_labels, val_hate_labels):
    # Standardize the data (important for Logistic Regression)
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings = scaler.transform(val_embeddings)
    
    # Train Logistic Regression for Fake News Classification
    fake_model = LogisticRegression(max_iter=1000)
    fake_model.fit(train_embeddings, train_fake_labels)

    # Train Logistic Regression for Hate Speech Classification
    hate_model = LogisticRegression(max_iter=1000)
    hate_model.fit(train_embeddings, train_hate_labels)

    # Make predictions on validation set
    fake_preds_val = fake_model.predict(val_embeddings)
    hate_preds_val = hate_model.predict(val_embeddings)

    # Evaluate the Fake News classification
    print("\nClassification Report (Fake News Detection):")
    print(classification_report(val_fake_labels, fake_preds_val))

    # Evaluate the Hate Speech classification
    print("\nClassification Report (Hate Speech Detection):")
    print(classification_report(val_hate_labels, hate_preds_val))

# Main function to orchestrate loading data, training, and evaluation
def main():
    # Load data
    train_file = r'BERT_whole_embeddings.pkl'
    val_file = r'BERT_val_embeddings.pkl'
    embeddings_train, fake_labels_train, hate_labels_train, embeddings_val, fake_labels_val, hate_labels_val = load_data(train_file, val_file)
    
    # Perform logistic regression classification
    logistic_regression_classification(embeddings_train, fake_labels_train, hate_labels_train, embeddings_val, fake_labels_val, hate_labels_val)

if __name__ == "__main__":
    main()
