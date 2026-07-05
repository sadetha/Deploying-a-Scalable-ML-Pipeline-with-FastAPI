import pandas as pd

df = pd.read_csv('data/census.csv')
print(df.head()) #top rows
print(df.tail()) #bottom rows
print(df.describe()) #breif overview of the data
df.info() #view counts by column