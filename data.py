import pandas as pd

# Load the CSV file
data = pd.read_csv('cleaned.csv')

# Check for missing or empty values in the 'Tweets' column
missing_tweets = data[data['Tweet'].isnull() | (data['Tweet'].str.strip() == '')]

# Print the IDs with missing Tweets
if not missing_tweets.empty:
    print("IDs with missing or empty Tweets:")
    print(missing_tweets['Id'].to_list())
else:
    print("No missing or empty Tweets found.")
