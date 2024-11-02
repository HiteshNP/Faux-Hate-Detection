import pandas as pd

# Load the tokenized CSV file
file_path = r'C:\Users\Ankith Jain\Desktop\faux hate\Embeddings\tokenized data\tokenized_dataset.csv'
df = pd.read_csv(file_path)

class_counts = df['Label'].value_counts()
print("Class counts in the main dataset:")
print(class_counts)

# Split the DataFrame into training and testing sets
train_df = df.iloc[:6000]  # First  rows for training
test_df = df.iloc[6000:]    # Remaining rows for testing

# Save the training and testing DataFrames to new CSV files
train_file_path = r'C:\Users\Ankith Jain\Desktop\faux hate\Embeddings\train_test_data\train_data.csv'
test_file_path = r'C:\Users\Ankith Jain\Desktop\faux hate\Embeddings\train_test_data\test_data.csv'

train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)

print("Training and testing datasets have been created successfully!")
