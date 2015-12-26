import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def standardize(data):
    """
    standardize
    """
    centered_data = data - data.mean()
    # MaxAbs scaling
    std_data = preprocessing.maxabs_scale(centered_data)
    return std_data

def standardize_local(data):
    # standardize for specific features
    data['Field10'] = pd.Series(preprocessing.maxabs_scale(data['Field10'] - data['Field10'].mean()))
    data['Year'] = pd.Series(preprocessing.maxabs_scale(data['Year'] - data['Year'].mean()))
    return data

def one_hot_encoder(data):
    """
    Label Encoder: categorical -> nemurical (One hot coding)
    """
    for f in data.columns:
        if data[f].dtype == 'object':
            print f
            lbl = preprocessing.LabelEncoder()
            int_label = lbl.fit_transform(list(data[f].values))

            enc = preprocessing.OneHotEncoder()
            one_hot_label = enc.fit_transform(int_label.reshape(-1, 1)).toarray()
            data = data.drop(f, axis=1)
            for i in range(one_hot_label.shape[1]):
                data['%s_%s'%(f,i)] = one_hot_label[:, i]
    return data


def label_encoder(data):
    """
    Label Encoder: categorical -> nemurical
    """
    # data.copy()
    for f in data.columns:
        if data[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            data[f] = lbl.fit_transform(list(data[f].values))
    return data

def pca(train, test, n_comp=None):
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

def lda(train, y, test, n_comp=None):
    """
    LDA
    """
    _lda = LinearDiscriminantAnalysis(solver='eigen', n_components=n_comp)
    # import pdb;pdb.set_trace()
    _lda.fit(train, y)
    new_train  = _lda.transform(train)
    new_test = _lda.transform(test)
    return new_train, new_test

def famd(train, test):
    """
    Factor Analysis for Mixed Data
    """
    return train, test

def clean(data):
    # Lets do some cleaning

    # play with date
    data['Date'] = pd.to_datetime(pd.Series(data['Original_Quote_Date']))
    data = data.drop('Original_Quote_Date', axis=1)

    data['Year'] = data['Date'].apply(lambda x: int(str(x)[:4]))
    data['Month'] = data['Date'].apply(lambda x: int(str(x)[5:7]))
    data['weekday'] = data['Date'].dt.dayofweek

    data = data.drop('Date', axis=1)

    # convert Field10 manully
    data['Field10'] = data['Field10'].apply(lambda x: int(x.replace(',','')))

    # # numerical features
    # ndf = data._get_numeric_data()
    # ncol = ndf.columns
    # nindex = ndf.index
    # # import pdb;pdb.set_trace()
    # # est = preprocessing.Imputer(missing_values='NaN', strategy='mean')
    # # ndf = est.fit_transform(ndf)
    # # ndf = pd.DataFrame(data=ndf, index=nindex, columns=ncol)


    # # numerical fillnan: -1
    # ndf = ndf.fillna(-1)
    # data[ncol] = ndf

    # # categorical fillnan: most_frequent
    # ccol = ['PersonalField7', 'Field6', 'PropertyField28', 'PropertyField5', 'PropertyField4', 'PropertyField7', 'PropertyField3', 'PersonalField18', 'PersonalField19', 'Field12', 'SalesField7', 'PersonalField16', 'PersonalField17', 'CoverageField8', 'CoverageField9', 'PropertyField32', 'GeographicField63', 'GeographicField64', 'PropertyField38', 'PropertyField37', 'PropertyField36', 'PropertyField34', 'PropertyField33', 'PropertyField14', 'PropertyField31', 'PropertyField30']
    # for col in ccol:
    #     if data[col].isnull().any():
    #         _list = data[col].tolist()
    #         _set = set(_list)
    #         most_freq_elem = None
    #         most_freq = 0
    #         for each in _set:
    #             freq = _list.count(each)
    #             if freq > most_freq:
    #                 most_freq = freq
    #                 most_freq_elem = each
    #         data[col] = data[col].fillna(each)
    # import pdb;pdb.set_trace()

    data = data.fillna(-1)

    if 'QuoteConversion_Flag' in data.columns:
        y = data.QuoteConversion_Flag.values
        train = data.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
        return train, y
    else:
        test = data.drop('QuoteNumber', axis=1)
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
