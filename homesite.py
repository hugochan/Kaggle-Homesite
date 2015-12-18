import preprocess as preproc
import classifier as clf
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd



if __name__  == '__main__':
    from time import time
    t0 = time()
    # train_file = "../datasets/train.csv"
    # test_file = "../datasets/test.csv"
    # train, test = load_data(train_file, test_file)

    # stratified k-fold cross-validation
    k = 3
    # random_state = np.random.RandomState(0)
    random_state = 0
    train_file = "../datasets/train.csv"
    df_train = pd.read_csv(train_file, header = 0, delimiter = ',')

    experiment = 2
    if experiment == 1:
        # experiment 1: LabelEncoder + xgboost

        data, y = preproc.clean(df_train)
        data = preproc.label_encoder(data)
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)
        avg_auc = 0.0
        for each_train, each_test in skf:
            pred = clf.boosted_trees(data.iloc[each_train], y[each_train], data.iloc[each_test])
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)

    elif experiment == 2:
        # experiment 2: LabelEncoder + PCA + xgboost

        data, y = preproc.clean(df_train)
        data = preproc.label_encoder(data)
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)
        avg_auc = 0.0
        for each_train, each_test in skf:
            # pca
            new_train, new_test = preproc.pca(data.iloc[each_train], data.iloc[each_test])
            pred = clf.boosted_trees(new_train, y[each_train], new_test)
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)

    else:
        raise ValueError('ERROR: Unexpected experiment: %s'%experiment)

    print "elapsed time: %ss"%int(time() - t0)

