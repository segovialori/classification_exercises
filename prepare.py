import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import acquire

def clean_iris(df):
    '''
    prep_iris will take a dataframe acquired as df and remove columns that are:
    duplicates,
    drop unessential columns,
    rename columns,
    and create dummies
    
    return: clean dataframe
    '''
    df.drop_duplicates(inplace=True)
    df.drop(['species_id'], axis=1)
    df.rename(columns={"species_name": "species"}, inplace=True)
    dummies = pd.get_dummies(df[['species']], drop_first=True)
    return pd.concat([df, dummies], axis=1)



def prep_iris(df): #when we are translating to its own script, call it as an argument
    '''
    prep_iris will take one argument df, a df, anticipated to be iris df
    and will remove species id,
    rename species name to species,
    and encode species into two columns
    perform a train, validate, test split
    
    return: threee pandas dataframes: train validate, test
    '''
    df = clean_iris(df)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate, train_size=0.3, random_state=123, stratify=train_validate.species)
    return train, validate, test
