import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import pickle

# Load the training and testing datasets with ELMo embeddings
train_file_path = r'C:\Users\Ankith Jain\Desktop\faux hate\Embeddings\train_test_data\train_data_embeddings.pkl'
test_file_path = r'C:\Users\Ankith Jain\Desktop\faux hate\Embeddings\train_test_data\test_data_embeddings.pkl'

with open(train_file_path, 'rb') as f:
    train_df = pickle.load(f)

with open(test_file_path, 'rb') as f:
    test_df = pickle.load(f)

# Convert ELMo embeddings from lists to numpy arrays
X_train = np.array(train_df['ELMo_Embeddings'].tolist())
y_train = train_df['Label']  # Ensure this column exists

X_test = np.array(test_df['ELMo_Embeddings'].tolist())
y_test = test_df['Label']  # Ensure this column exists

# Define the SVM model
model = SVC()

# Set up the hyperparameter distribution
param_dist = {
    'C': np.logspace(-3, 3, 7),  # Regularization parameter from 0.001 to 1000
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel types
    'gamma': ['scale', 'auto'] + np.logspace(-3, 3, 7).tolist()  # Gamma values
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, 
                                   n_iter=100, scoring='accuracy', 
                                   cv=5, n_jobs=-1, verbose=1, random_state=42)

# Train the model with Randomized Search
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {accuracy:.4f}")

# Print classification report for more details
print(classification_report(y_test, y_pred))
