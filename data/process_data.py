import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    # Merge datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    """
    Here we clean the categories of the dataframe being passed.
    
    Takes in:
        df: DataFrame containing the combined messages and categories
    Outputs:
        df: DataFrame containing messages and cleaned up categories.
    
    """
    
    # Split categories into separate category columns
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Using this row to extract a list of new col names for categories
    category_colnames = list(map(lambda x: str(x)[:-2], row))
    
    #rename the columns of 'categories'
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: str(x)[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    
    #Remove duplicates
    df = df.drop_duplicates()   
    
    return df
    

def save_data(df, database_filename):
    """
    Saves data to SQLite Database
    
    Takes In:
        df: DataFrame of messages and cleaned categories
        database_filename: SQLite path
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    #engine = create_engine(f'sqlite:///{database_filename}')
    #table = database_filename.replace(".db","") + "_table"
    df.to_sql('df', engine, index=False, if_exists='replace')


def main():
    print(sys.argv) #sys.argv will always count the name of the file as the first element.
    if len(sys.argv) == 4:
        
        #passing in the location of the file when we run this file.
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] 
        #^skips index 0, ie the first element which is the name of this python file

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()