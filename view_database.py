# import sqlite3
# import pandas as pd

# # Connect to the database
# conn = sqlite3.connect('data.db')

# # Query the database and load the results into a DataFrame
# df = pd.read_sql_query("SELECT * FROM action", conn)
# text = pd.read_sql_query("SELECT * FROM metrics WHERE Ticker = 'UNH'", conn)
# # Print the DataFrame
# print(df.head())
# # Close the connection
# conn.close()

import os
import pandas as pd
path = os.getcwd()
# Path to the metrics folder
metrics_path = './results'  # Adjust the path as needed

# Initialize an empty DataFrame to store appended data
appended_data = pd.DataFrame()

# Loop through all files in the directory
for file in os.listdir(metrics_path):
    # print(file)
    # Check if the file is a CSV and contains '-action'
    if file.endswith('.csv') and '-action' in file:
        file_path = os.path.join(metrics_path, file)
        print(f"Processing: {file}")  # Debug message
        # Read the CSV and append to the DataFrame
        df = pd.read_csv(file_path)
        appended_data = pd.concat([appended_data, df], ignore_index=True)

# Resulting DataFrame
print("Final DataFrame shape:", appended_data.shape)
appended_data.to_csv(os.path.join(path, 'allstocks.csv'), index=False)



