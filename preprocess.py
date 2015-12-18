import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_data(train_file, test_file):
    df_train = pd.read_csv(train_file, header = 0, delimiter = ',')
    df_test = pd.read_csv(test_file, header = 0, delimiter = ',')

    # put the original column names in a python list
    original_headers = list(df_train.columns.values)

    # remove the non-numeric columns
    df = df_train._get_numeric_data()

    # put the numeric column names in a python list
    # numeric_headers = list(df.columns.values)

    # create a numpy array with the numeric values for input into scikit-learn
    # numpy_array = df.as_matrix()
    print "# training records: %s"%df_train.shape[0]
    print "# training attrs: %s"%df_train.shape[1]
    print "# training numerical attrs: %s"%df.shape[1]
    print "# training categorical attrs: %s"%(df_train.shape[1] - df.shape[1])

    return df_train, df_test

def label_encoder(train, test):
    """
    Label Encoder: categorical -> nemurical
    """
    # import copy
    for f in train.columns:
        if train[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))
    return train, test

def pca(train, test):
    """
    PCA
    """
    return train, test

def pca_irranking(train, y, test):
    """
    PCA + Individual Relevance Ranking
    """
    return train, test

def lda(train, y, test):
    """
    LDA
    """
    return train, test

def famd(train, test):
    """
    Factor Analysis for Mixed Data
    """
    return train, test

def clean(train, test):
    y = train.QuoteConversion_Flag.values
    train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    test = test.drop('QuoteNumber', axis=1)

    # Lets do some cleaning
    train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
    train = train.drop('Original_Quote_Date', axis=1)

    test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
    test = test.drop('Original_Quote_Date', axis=1)

    train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
    train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
    train['weekday'] = train['Date'].dt.dayofweek


    test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
    test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
    test['weekday'] = test['Date'].dt.dayofweek

    train = train.drop('Date', axis=1)
    test = test.drop('Date', axis=1)

    train = train.fillna(-1)
    test = test.fillna(-1)
    return train, y, test

def feature_extraction(train, y, test, method):
    if method == 'pca':
        train, test = label_encoder(train, test)
        train, test = pca(train, test)
    elif method == 'pca_irranking':
        train, test = label_encoder(train, test)
        train, test = pca_irranking(train, y, test)
    elif method == 'lda':
        train, test = label_encoder(train, test)
        train, test = lda(train, y, test)
    elif method == 'famd':
        train, test = famd(train, test)
    else:
        raise ValueError('ERROR: Unexpected argument: method = %s'%method);
    return train, test

def preprocess(train, test, method):
    """
    clean + feature extraction
    """
    train, y, test = clean(train, test)
    train, test = feature_extraction(train, y, test, method)
    return train, y, test

if __name__  == '__main__':
    train_file = "datasets/train.csv"
    test_file = "datasets/test.csv"
    train, test = load_data(train_file, test_file)

    feature_extraction_method = 'pca'
    train, y, test = preprocess(train, test, feature_extraction_method)

