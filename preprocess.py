import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

def load_data(train_file, test_file):
    df_train = pd.read_csv(train_file, header = 0, delimiter = ',')
    df_test = pd.read_csv(test_file, header = 0, delimiter = ',')

    # put the original column names in a python list
    # original_headers = list(df_train.columns.values)

    # remove the non-numeric columns
    # df = df_train._get_numeric_data()

    # put the numeric column names in a python list
    # numeric_headers = list(df.columns.values)

    # create a numpy array with the numeric values for input into scikit-learn
    # numpy_array = df.as_matrix()
    # print "# training records: %s"%df_train.shape[0]
    # y = df_train.QuoteConversion_Flag.values
    # print "# training positive records: %s"%y.sum()
    # print "# training negative records: %s"%(y.shape[0] - y.sum())
    # print "# training attrs: %s"%df_train.shape[1]
    # print "# training numerical attrs: %s"%df.shape[1]
    # print "# training categorical attrs: %s"%(df_train.shape[1] - df.shape[1])

    return df_train, df_test

def label_encoder(data):
    """
    Label Encoder: categorical -> nemurical
    """
    # data.copy()
    for f in data.columns:
        if data[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
    return data

def pca(train, test, n_comp=292):
    """
    PCA
    """
    _pca = PCA(n_components=n_comp) # user-tuned: 292
    _pca.fit(train)
    new_train = _pca.transform(train)
    new_test = _pca.transform(test)
    # import pdb;pdb.set_trace()
    return new_train, new_test

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

def clean(data):
    if 'QuoteConversion_Flag' in data.columns:
        y = data.QuoteConversion_Flag.values
        train = data.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)

        # Lets do some cleaning
        train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
        train = train.drop('Original_Quote_Date', axis=1)

        train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
        train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
        train['weekday'] = train['Date'].dt.dayofweek

        train = train.drop('Date', axis=1)

        train = train.fillna(-1)
        return train, y
    else:
        test = data.drop('QuoteNumber', axis=1)

        # Lets do some cleaning
        test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
        test = test.drop('Original_Quote_Date', axis=1)

        test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
        test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
        test['weekday'] = test['Date'].dt.dayofweek

        test = test.drop('Date', axis=1)

        test = test.fillna(-1)
        return test

def feature_extraction(train, y, test, method):
    if method == 'pca':
        train = label_encoder(train)
        test = label_encoder(test)
        train, test = pca(train, test)
    elif method == 'pca_irranking':
        train = label_encoder(train)
        test = label_encoder(test)
        train, test = pca_irranking(train, y, test)
    elif method == 'lda':
        train = label_encoder(train)
        test = label_encoder(test)
        train, test = lda(train, y, test)
    elif method == 'famd':
        train, test = famd(train, test)
    else:
        raise ValueError('ERROR: Unexpected argument: method = %s'%method)
    return train, test

def preprocess(train, test, method):
    """
    clean + feature extraction
    """
    train, y, test = clean(train, test)
    train, test = feature_extraction(train, y, test, method)
    return train, y, test
