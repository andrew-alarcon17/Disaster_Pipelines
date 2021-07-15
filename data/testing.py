import sys
import pandas as pd

#if len(sys.argv) == 4:

messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

print(messages_filepath)

#messages_filepath = pd.read_csv('disaster_messages.csv')
#categories_filepath = pd.read_csv('disaster_categories.csv')
# ('DisasterResponse.db')

#print(messages_filepath.head(5))