import pandas as pd

# Load the dataset
df = pd.read_csv('Data.csv')

# Function to split FAUX into Hate and Fake
def split_faux(label):
    if label == 'Fake-Hate':
        return 1, 1
    elif label == 'Fake-NonHate':
        return 0, 1
    elif label == 'NonFake-Hate':
        return 1, 0
    else:  # 'NonFake-NonHate'
        return 0, 0

# Apply the function and create new columns
df[['Hate', 'Fake']] = df['FAUX'].apply(lambda x: pd.Series(split_faux(x)))

# Drop the original FAUX column if needed
df.drop(columns=['FAUX'], inplace=True)

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_whole_data.csv', index=False)

# Display the modified DataFrame
print(df)
