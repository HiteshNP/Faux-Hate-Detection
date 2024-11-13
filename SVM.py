import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load embeddings data
with open('Test_HingBERT_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract embeddings and labels
embeddings = [entry['embedding'] for entry in data]
labels = [entry['FAUX'] for entry in data]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, random_state=42)

# Create SVM classifier pipeline with standardization
svm_pipeline = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', C=1.0, random_state=42))

# Train the model
svm_pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_pipeline.predict(X_test)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
