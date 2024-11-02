import pandas as pd

# Load the Excel file
excel_file = 'Train_Task_A.xlsx'
data = pd.read_excel(excel_file)

# Save the data as a CSV file
csv_file = 'Train_Task_A.csv'
data.to_csv(csv_file, index=False)

print("Conversion complete! The file is saved as train.csv.")
