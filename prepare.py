def prep_iris():
    '''
    prep_iris will take a dataframe acquired as df and remove columns that are:
    duplicates,
    drop unessential columns,
    rename columns,
    and create dummies
    
    return: clean dataframe
    '''
    df.drop_duplicates(inplace=True)
    df = df.drop(['species_id'], axis=1)
    df.rename(columns={"species_name": "species"}, inplace=True)
    dummies = pd.get_dummies(df[['species']], drop_first=False)
    return pd.concat([df, dummies], axis=1)
