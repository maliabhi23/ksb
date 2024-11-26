import sqlite3

# Connect to the database
conn = sqlite3.connect("users.db")  # Replace with the path to your database file
cursor = conn.cursor()

# Execute a query to fetch all data from the "users" table
cursor.execute("SELECT * FROM users")

# Fetch and print all rows
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the connection
conn.close()
