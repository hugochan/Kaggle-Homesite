import preprocess as preproc
import classifier as clf
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
import xgboost as xgb

import numpy as np
import pandas as pd


# import pca2
import bi_lda


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

    experiment = 10
    if experiment == 1:
        # experiment 1: LabelEncoder + xgboost
        print "experiment 1: LabelEncoder + xgboost"
        data, y = preproc.clean(df_train)
        data = preproc.label_encoder(data)
        # local standardization
        # data = preproc.standardize_local(data)
        # data = preproc.standardize(data)
        # import pdb;pdb.set_trace()
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)
        avg_auc = 0.0
        for each_train, each_test in skf:
            pred = clf.boosted_trees(data.iloc[each_train], y[each_train], data.iloc[each_test])
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)

    elif experiment == 2:
        # experiment 2: LabelEncoder + PCA + xgboost
        print "experiment 2: LabelEncoder + PCA + xgboost"
        data, y = preproc.clean(df_train)
        data = preproc.label_encoder(data)
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)
        avg_auc = 0.0
        for each_train, each_test in skf:
            # pca
            # _pca = pca2.PCA()
            # w = _pca.train(data.iloc[each_train].as_matrix(), 1)
            # new_train = _pca.project(w, data.iloc[each_train].as_matrix().transpose()).transpose()
            # new_test = _pca.project(w, data.iloc[each_test].as_matrix().transpose()).transpose()

            new_train, new_test = preproc.pca(data.iloc[each_train], data.iloc[each_test], n_comp=280)
            pred = clf.boosted_trees(new_train, y[each_train], new_test)
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)

    elif experiment == 3:
        # experiment 3: LabelEncoder + standardized PCA + xgboost
        print "experiment 3: LabelEncoder + standardized PCA + xgboost"
        data, y = preproc.clean(df_train)
        data = preproc.label_encoder(data)
        data = preproc.standardize(data) # standardization
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)
        avg_auc = 0.0
        for each_train, each_test in skf:
            new_train, new_test = preproc.pca(data[each_train, :], data[each_test, :], n_comp=None)
            pred = clf.boosted_trees(new_train, y[each_train], new_test)
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)

    elif experiment == 4:
        # experiment 4: LabelEncoder + standardized LDA + xgboost
        print "experiment 4: LabelEncoder + standardized LDA + xgboost"
        data, y = preproc.clean(df_train)
        data = preproc.label_encoder(data)
        # data = preproc.standardize(data) # standardization

        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)
        avg_auc = 0.0
        for each_train, each_test in skf:
            # new_train, new_test = preproc.lda(data[each_train, :], y[each_train], data[each_test, :], n_comp=None)

            lda = bi_lda()
            train_label = np.hstack((data.iloc[each_train].as_matrix(), y[each_train]))
            w = lda.train(train_label)
            new_train = lda.project(w, data.iloc[each_train].as_matrix().transpose()).transpose()
            new_test = lda.project(w, data.iloc[each_test].as_matrix().transpose()).transpose()

            pred = clf.boosted_trees(new_train, y[each_train], new_test)
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)

    elif experiment == 5:
        # experiment 5: OneHotEncoder + xgboost
        print "experiment 5: OneHotEncoder + xgboost"
        data, y = preproc.clean(df_train)
        # import pdb;pdb.set_trace()
        data = preproc.one_hot_encoder(data)
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)
        avg_auc = 0.0
        for each_train, each_test in skf:
            pred = clf.boosted_trees(data.iloc[each_train], y[each_train], data.iloc[each_test])
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
            print "%s"%auc
        print "avg auc: %s"%(avg_auc/k)

    elif experiment == 6:
        # experiment 6: Exhaustive search over specified parameter values for xgboost
        print "experiment 6: Exhaustive search over specified parameter values for xgboost"
        data, y = preproc.clean(df_train)
        data = preproc.one_hot_encoder(data)
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)

        xgb_model = xgb.XGBClassifier()
        clf = GridSearchCV(estimator=xgb_model,
                       param_grid={'max_depth': [10], # range(8, 16, 2),
                        'n_estimators': [50, 100, 200, 500],
                        'learning_rate': [0.02],
                        # 'gamma': [0, 0.1],
                        'min_child_weight': [1], # range(1, 6, 2),
                        'subsample': [0.8],
                        'colsample_bytree': [0.8],

                        },
                        scoring='roc_auc',
                        cv=skf,
                        error_score=1,
                        verbose=1)
        clf.fit(data, y)

        avg_auc = 0.0
        for each_train, each_test in skf:
            pred = clf.predict_proba(data.iloc[each_test])[:,1]
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)
        import pdb;pdb.set_trace()

    elif experiment == 7:
        # experiment 7: OneHotEncoder + Logistic regression
        print "experiment 7: OneHotEncoder + Logistic regression"
        data, y = preproc.clean(df_train)
        data = preproc.one_hot_encoder(data)
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)

        avg_auc = 0.0
        for each_train, each_test in skf:
            pred = clf.logistic_regression(data.iloc[each_train], y[each_train], data.iloc[each_test])
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)
        # 0.8181932195

    elif experiment == 8:
        # experiment 8: OneHotEncoder + Naive bayes
        print "experiment 8: OneHotEncoder + Naive bayes"
        data, y = preproc.clean(df_train)
        data = preproc.one_hot_encoder(data)
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)

        avg_auc = 0.0
        for each_train, each_test in skf:
            pred = clf.naive_bayes(data.iloc[each_train], y[each_train], data.iloc[each_test])
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)
        # 0.753509641813

    elif experiment == 9:
        # experiment 9: OneHotEncoder + l1_based_select + xgboost
        print "experiment 9: OneHotEncoder + l1_based_select + xgboost"
        data, y = preproc.clean(df_train)
        data = preproc.one_hot_encoder(data)
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)

        avg_auc = 0.0
        for each_train, each_test in skf:
            new_train, new_test = preproc.l1_based_select(data.iloc[each_train], y[each_train], data.iloc[each_test])
            pred = clf.boosted_trees(new_train, y[each_train], new_test)
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
        print "avg auc: %s"%(avg_auc/k)
        # 0.957124130844

    elif experiment == 10:
        # experiment 10: OneHotEncoder + neural network
        print "experiment 10: OneHotEncoder + neural network"
        data, y = preproc.clean(df_train)
        data = preproc.one_hot_encoder(data)
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=random_state)

        avg_auc = 0.0
        for each_train, each_test in skf:
            pred = clf.nn(data.iloc[each_train], y[each_train], data.iloc[each_test], y[each_test])
            auc = roc_auc_score(y[each_test], pred)
            avg_auc += auc
            print "%s"%auc
        print "avg auc: %s"%(avg_auc/k)
        #

    else:
        raise ValueError('ERROR: Unexpected experiment: %s'%experiment)

    print "elapsed time: %ss"%int(time() - t0)

