import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the saved embeddings and labels for training
with open(r"C:\Users\Ankith Jain\Desktop\FAUX\Task-B\B_flair.pkl", "rb") as f:
    train_data = pickle.load(f)

# Replace NaN values in 'Target' and 'Severity' with 'N/A'
train_data['Target'] = train_data['Target'].fillna('N/A')
train_data['Severity'] = train_data['Severity'].fillna('N/A')

# Separate embeddings and labels
X_train = np.array(list(train_data['Embeddings']))
y_target_train = train_data['Target']
y_severity_train = train_data['Severity']

# Train Logistic Regression models on the entire training data
target_model = LogisticRegression(max_iter=1000)
severity_model = LogisticRegression(max_iter=1000)

print("\n--- Training Models ---")
# Train Target model
target_model.fit(X_train, y_target_train)
print("Target model trained.")

# Train Severity model
severity_model.fit(X_train, y_severity_train)
print("Severity model trained.")

# Load the validation embeddings and IDs
with open(r"C:\Users\Ankith Jain\Desktop\FAUX\Task-B\B_flair_val.pkl", "rb") as f:
    validation_data = pickle.load(f)  # Assume it contains embeddings and IDs

# Extract validation embeddings and IDs
validation_embeddings = np.array(list(validation_data['Embeddings']))
validation_ids = validation_data['Id']  # Ensure Tweet_ID is present in the pickle file

# Generate predictions for validation data
print("\n--- Generating Predictions for Validation Data ---")
predicted_targets = target_model.predict(validation_embeddings)
predicted_severity = severity_model.predict(validation_embeddings)

# Save predictions with Tweet_ID
validation_data['Predicted_Target'] = predicted_targets
validation_data['Predicted_Severity'] = predicted_severity

# Save predictions for analysis
validation_data[['Id', 'Predicted_Target', 'Predicted_Severity']].to_csv(
    r"C:\Users\Ankith Jain\Desktop\FAUX\validation\flair_val_A_predictions.csv", index=False
)
print("Validation predictions saved.")

# Map validation predictions with the training data for accuracy checking
print("\n--- Mapping Predictions to Training Data ---")
train_df = pd.read_csv(r"C:\Users\Ankith Jain\Desktop\FAUX\validation\cleaned_val_B.csv")  # Original train file
train_df = train_df[['Id', 'Target', 'Severity']]  # Keep only relevant columns

## Perform merge
merged_df = pd.merge(
    validation_data[['Id', 'Predicted_Target', 'Predicted_Severity']],
    train_df,
    on='Id',
    how='inner'
)

# Check if the merge resulted in data
print(f"\nMerged DataFrame Size: {merged_df.shape}")
print(f"Merged DataFrame (first few rows): \n{merged_df.head()}")

# Map 'N/A' as a valid class
merged_df['Target'] = merged_df['Target'].fillna('N/A')
merged_df['Predicted_Target'] = merged_df['Predicted_Target'].fillna('N/A')
merged_df['Severity'] = merged_df['Severity'].fillna('N/A')
merged_df['Predicted_Severity'] = merged_df['Predicted_Severity'].fillna('N/A')

# Ensure there are valid rows for evaluation
if merged_df.shape[0] == 0:
    print("No valid rows found for evaluation.")
else:
    # Calculate accuracy for Target
    target_accuracy = accuracy_score(merged_df['Target'], merged_df['Predicted_Target'])
    print(f"\nAccuracy for Target: {target_accuracy:.2f}")

    # Calculate accuracy for Severity
    severity_accuracy = accuracy_score(merged_df['Severity'], merged_df['Predicted_Severity'])
    print(f"Accuracy for Severity: {severity_accuracy:.2f}")

    # Display classification reports for both labels
    print("\nClassification Report for Target:")
    print(classification_report(merged_df['Target'], merged_df['Predicted_Target']))

    print("\nClassification Report for Severity:")
    print(classification_report(merged_df['Severity'], merged_df['Predicted_Severity']))
