import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function load the messages and the categories from a file path determined by the user.

    Args:
        messages_filepath: a string
        categories_filepath: a string       

    Returns:
        df : Pandas DataFrame contains all the merged data  
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    messages= messages.set_index('id')
    categories= categories.set_index('id')
    df = messages.join(categories, on='id')
    return df

def clean_data(df):
    """
    This function takes a messy data and clean it.

    Args:
        df: Pandas DataFrame contains all the merged data 
     

    Returns:
        df : Pandas DataFrame contains  the cleaned data
    """        
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[1,:]
    row_splitted= row.str.split('-', expand=True)
    columns= row_splitted.iloc[:,0]
    categories.columns = columns

    for name in columns:
        categories[name] = categories[name].str[-1:]

    categories = categories.astype(int)
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df = df[df.related != 2]
    df.drop_duplicates(inplace =True)

    return df

def save_data(df, database_filename):
    """
    This function takes a clean data and store in in SQLite database with name sepecified by the user.

    Args:
        df: Pandas DataFrame contains  the cleaned data 
     

    Returns:
    """  


    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('T_messeges', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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
