import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Load the embeddings data
with open('Test_mBERT_embeddings.pkl', 'rb') as f:
    embeddings_data = pickle.load(f)

# Prepare the dataset
X = np.array([item['embedding'] for item in embeddings_data])  # Extract embeddings
y = np.array([item['FAUX'] for item in embeddings_data])        # Extract labels

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numeric values

# Shift the data to ensure non-negativity for MultinomialNB
min_value = np.min(X)
if min_value < 0:
    X_shifted = X + abs(min_value)  # Shift the data to make all values non-negative
else:
    X_shifted = X

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_shifted, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Initialize a dictionary to store results
results = {}

# Define classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Train each classifier and evaluate its performance
for name, clf in tqdm(classifiers.items()):
    clf.fit(X_train, y_train)  # Fit the model
    y_pred = clf.predict(X_test)  # Make predictions

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    # Store the results
    results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report
    }

# Print results for each classifier
for name, metrics in results.items():
    print(f"Classifier: {name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("=" * 60)

# Find the best classifier based on F1 score
best_classifier = max(results, key=lambda k: results[k]['f1_score'])
print(f"The best classifier based on F1 Score is: {best_classifier} with F1 Score: {results[best_classifier]['f1_score']:.4f}")
