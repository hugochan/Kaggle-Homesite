import preprocess as preproc
import classifier as clf
# from sklearn.cross_validation import StratifiedKFold
# from sklearn.metrics import roc_auc_score
# from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd




if __name__  == '__main__':
    from time import time
    t0 = time()
    train_file = "../datasets/train.csv"
    test_file = "../datasets/test.csv"
    sample_submission_file = "../datasets/sample_submission.csv"
    train, test = preproc.load_data(train_file, test_file)

    experiment_dict = {1: 'xgboost'}
    experiment = 1
    if experiment == 1:
        # experiment 1: OneHotEncoder + xgboost
        print "experiment 1: OneHotEncoder + xgboost"
        train, y = preproc.clean(train)
        test = preproc.clean(test)

        data = pd.concat([train, test])
        data = preproc.one_hot_encoder(data)

        data = preproc.standardize(data)

        train = data.iloc[: train.shape[0]]
        test = data.iloc[train.shape[0]:]
        pred = clf.boosted_trees(train, y, test)

    else:
        raise ValueError('ERROR: Unexpected experiment: %s'%experiment_dict[experiment])

    print "elapsed time: %ss"%int(time() - t0)

    sample = pd.read_csv(sample_submission_file, header=0, delimiter=',')
    sample.QuoteConversion_Flag = pred
    sample.to_csv('submission_%s.csv'%experiment_dict[experiment], index=False)
