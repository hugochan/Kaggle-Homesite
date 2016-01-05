# import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import TrainSplit
# from lasagne.objectives import binary_crossentropy, binary_accuracy
import theano
from sklearn.metrics import roc_auc_score

seed = 260681

def boosted_trees(train, y, test, y2=None):
    """ defalut params:
    max_depth=3, learning_rate=0.1,
    n_estimators=100, silent=True,
    objective="binary:logistic",
    nthread=-1, gamma=0, min_child_weight=1,
    max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
    reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
    base_score=0.5, seed=0, missing=None
    """
    clf = xgb.XGBClassifier(n_estimators=400,
                            nthread=-1,
                            max_depth=16, # 16
                            learning_rate=0.03, # 0.03
                            min_child_weight=2, # 2
                            silent=True,
                            # gamma=0, # 0
                            # colsample_bylevel=1, # 1
                            # scale_pos_weight=1, # 1
                            subsample=0.8, # 0.8
                            colsample_bytree=0.81) # 0.81



    xgb_model = clf.fit(train, y, eval_metric="auc", eval_set=[(train, y)])

    # xgb_model = clf.fit(train, y, eval_metric="auc", eval_set=[(test, y2), (train, y)], early_stopping_rounds=3)
    # evals_result = clf.evals_result()
    # print evals_result

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

def nn(train, y, test, y2):
    # Get the data in shape for Lasagne
    # Prep the data for a neural net
    n_classes = 2
    n_features = train.shape[1]

    # Convert to np.array to make lasagne happy
    train = train.as_matrix().astype(np.float32)
    test = test.as_matrix().astype(np.float32)
    y = y.astype(np.int32)

    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    epochs = 5

    # Comment out second layer for run time.
    layers = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('dropout1', DropoutLayer),
               ('output', DenseLayer)
               ]

    net1 = NeuralNet(layers=layers,
                    input_shape=(None, n_features),
                    dense0_num_units=512, # 512, - reduce num units to make faster
                    dropout0_p=0.1,
                    dense1_num_units=512,
                    dropout1_p=0.1,
                    output_num_units=n_classes,
                    output_nonlinearity=softmax,
                    update=adagrad,
                    update_learning_rate=0.02,
                    train_split=TrainSplit(eval_size=0.0),

                    # objective_loss_function = binary_accuracy,
                    verbose=1,
                    max_epochs=1)

    for i in range(epochs):
        net1.fit(train, y)
        preds = net1.predict_proba(test)[:, 1]
        auc = roc_auc_score(y2, preds)
        print "auc %s: %s"%(i, auc)
    return preds

def nn2(train, y, test):
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
    preds = clf.predict_proba(test)[:, 1]
    return preds
