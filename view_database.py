import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('data.db')

# Query the database and load the results into a DataFrame
df = pd.read_sql_query("SELECT * FROM action", conn)
text = pd.read_sql_query("SELECT * FROM metrics WHERE Ticker = 'UNH'", conn)
# Print the DataFrame
print(df.head())
# Close the connection
conn.close()
