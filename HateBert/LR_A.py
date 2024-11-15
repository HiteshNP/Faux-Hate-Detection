import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the saved embeddings and labels
with open(r"C:\Users\Ankith Jain\Desktop\FAUX\HateBert\A_hate.pkl", "rb") as f:
    data = pickle.load(f)

# Separate embeddings
X = np.array(list(data['Embeddings']))

# Define a function to train and evaluate a model for each label
def train_and_evaluate(X, y, label_name):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classifiers
    logistic_regression_model = LogisticRegression(max_iter=1000)
    print(f"\n--- Results for {label_name} ---")

    # Train and evaluate Logistic Regression
    logistic_regression_model.fit(X_train, y_train)
    y_pred_lr = logistic_regression_model.predict(X_test)
    print(f"Logistic Regression Accuracy ({label_name}):", accuracy_score(y_test, y_pred_lr))
    print(f"Logistic Regression Classification Report ({label_name}):\n", classification_report(y_test, y_pred_lr))

# Train and evaluate separately for 'Target' and 'Severity'
train_and_evaluate(X, data['Fake'], 'Fake')
train_and_evaluate(X, data['Hate'], 'Hate')
