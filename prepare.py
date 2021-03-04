import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import acquire
from sklearn.preprocessing import LabelEncoder

def clean_iris(df):
    '''
    prep_iris will take a dataframe acquired as df and remove columns that are:
    duplicates,
    drop unessential columns,
    rename columns,
    and create dummies
    
    return: clean dataframe
    '''
    df.drop(['species_id'], axis=1, inplace=True)
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
###################### Train, Validate, Test, Split ######################
def train_validate_test_split(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.species
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.species,
    )
    return train, validate, test


def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.3, random_state=123)
        train, validate = train_test_split(df, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(df, test_size=.3, random_state=123, stratify=train[stratify_by])
    
    return train, validate, test

def handle_missing_values(df):
    return df.assign(
        embark_town=df.embark_town.fillna('Other'),
        embarked=df.embarked.fillna('O'),
    )

def remove_columns(df):
    return df.drop(columns=['deck'])

def encode_embarked(df):
    encoder = LabelEncoder()
    encoder.fit(df.embarked)
    return df.assign(embarked_encode = encoder.transform(df.embarked))

def prep_titanic_data(df):
    df = df\
        .pipe(handle_missing_values)\
        .pipe(remove_columns)\
        .pipe(encode_embarked)
    return df