# import pandas as pd
# import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier

seed = 260681

def boosted_trees(train, y, test):
    clf = xgb.XGBClassifier(n_estimators=25,
                            nthread=-1,
                            max_depth=16, # 16
                            learning_rate=0.03, # 0.03
                            min_child_weight=2, # 2
                            silent=True,
                            subsample=0.8,
                            colsample_bytree=0.9)



    xgb_model = clf.fit(train, y, eval_metric="auc")

    preds = clf.predict_proba(test)[:, 1]
    # sample = pd.read_csv('datasets/sample_submission.csv')
    # sample.QuoteConversion_Flag = preds
    # sample.to_csv('xgb_benchmark.csv', index=False)
    return preds


# If you care only about the ranking order (AUC) of your prediction
# Balance the positive and negative weights, via scale_pos_weight
# Use AUC for evaluation


def logistic_regression(train, y, test):
    clf = LogisticRegression(penalty='l2',
                            dual=False,
                            tol=0.0001,
                            C=1.0,
                            fit_intercept=True,
                            intercept_scaling=1,
                            class_weight=None,
                            random_state=None,
                            solver='sag',
                            max_iter=10000,
                            multi_class='ovr',
                            verbose=0,
                            warm_start=False
        )
    # 0.8181932195
    logreg_model = clf.fit(train, y)
    preds = clf.predict_proba(test)[:, 1]
    return preds

def naive_bayes(train, y, test):
    clf = GaussianNB()
    # 0.753509641813
    gnb_model = clf.fit(train, y)
    preds = clf.predict_proba(test)[:, 1]
    return preds

def nn(train, y, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    # apply same transformation to test data
    test = scaler.transform(test)

    clf = MLPClassifier(activation='logistic',
                        algorithm='adam',
                        alpha=0.0001,
                        hidden_layer_sizes=(100, ),
                        batch_size=2000,
                        random_state=1)

    nn_model = clf.fit(train, y)
    preds = clf.predict_proba()
    return preds
