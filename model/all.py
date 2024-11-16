import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier

# Load the saved embeddings and labels
with open(r"C:\Users\Ankith Jain\Desktop\FAUX\HateBert\B_hate.pkl", "rb") as f:
    data = pickle.load(f)

# Replace NaN values in 'Target' and 'Severity' with 'N/A'
data['Target'] = data['Target'].fillna('N/A')
data['Severity'] = data['Severity'].fillna('N/A')

# Separate embeddings
X = np.array(list(data['Embeddings']))

# Define a function to train and evaluate a model for each label
def train_and_evaluate(X, y, label_name):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classifiers
    logistic_regression_model = LogisticRegression(max_iter=1000)
    svm_model = SVC()
    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    ensemble_model = VotingClassifier(estimators=[('lr', logistic_regression_model),
                                                  ('svm', svm_model),
                                                  ('rf', rf_model)],
                                      voting='hard')
    stacking_model = StackingClassifier(estimators=[('lr', logistic_regression_model),
                                                    ('svm', svm_model),
                                                    ('dt', dt_model)],
                                        final_estimator=RandomForestClassifier())

    models = {
        'Logistic Regression': logistic_regression_model,
        'SVM': svm_model,
        'Decision Tree': dt_model,
        'Random Forest': rf_model,
        'Ensemble': ensemble_model,
        'Stacking': stacking_model
    }

    print(f"\n--- Results for {label_name} ---")

    for model_name, model in models.items():
        # Train and evaluate each model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{model_name} Accuracy ({label_name}):", accuracy_score(y_test, y_pred))
        print(f"{model_name} Classification Report ({label_name}):\n", classification_report(y_test, y_pred))

# Train and evaluate separately for 'Target' and 'Severity'
train_and_evaluate(X, data['Target'], 'Target')
train_and_evaluate(X, data['Severity'], 'Severity')
