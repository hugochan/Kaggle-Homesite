# import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# neural nets
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne import objective
from nolearn.lasagne import TrainSplit
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad, momentum, rmsprop, nesterov_momentum
from lasagne.nonlinearities import softmax, rectify # fails to import softplus
# from theano.tensor.nnet import softplus
from lasagne.init import Normal, Constant, GlorotUniform
from lasagne.objectives import categorical_crossentropy
import theano

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
    clf = xgb.XGBClassifier(n_estimators=500,
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


    # xgb_model = clf.fit(train, y, eval_metric="auc", eval_set=[(train, y), (val_X, val_y)], early_stopping_rounds=3)

    xgb_model = clf.fit(train, y, eval_metric="auc", eval_set=[(test, y2), (train, y)], early_stopping_rounds=3)
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

def nn(train, y, test, y2=None):
    # Get the data in shape for Lasagne
    # Prep the data for a neural net
    n_features = train.shape[1]

    # import pdb;pdb.set_trace()
    # Convert to np.array to make lasagne happy
    train = train.as_matrix().astype(np.float32)
    test = test.as_matrix().astype(np.float32)
    y = y.astype(np.int32)
    scaler = StandardScaler() # try uniformization ?
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    # epochs = 5

    # notes:
    # reference: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    # a dropout net should typically use 10-100 times the learning rate that was optimal for a standard neural net.
    # While momentum values of 0.9 are common for standard nets, with dropout we found that values around 0.95 to 0.99 work quite a lot better.
    # max-norm of weight usualy ranges from 3 to 4
    # layers = [('input', InputLayer),
    #            ('dropout0', DropoutLayer),
    #            ('hidden0', DenseLayer),
    #            ('dropout1', DropoutLayer),
    #            ('hidden1', DenseLayer),
    #            ('dropout2', DropoutLayer),
    #            ('output', DenseLayer)
    #            ]
    # # init weights
    # dropout0_p = 0.2 # as for input layer, usually close to 0
    # dropout1_p = 0.5 # as for hidden layer, usually close to 0.5
    # dropout2_p = 0.5 # as for hidden layer, usually close to 0.5
    # hidden0_num_units = 512
    # hidden1_num_units = 512
    # net1 = NeuralNet(layers=layers,
    #                 input_shape=(None, n_features),
    #                 dropout0_p=dropout0_p, #

    #                 hidden0_num_units=hidden0_num_units, #
    #                 hidden0_W=Normal(std=(1.0/n_features)**0.5), # small weights in case of saturation
    #                 hidden0_b=Normal(std=(1.0/n_features)**0.5),
    #                 hidden0_nonlinearity=softplus, # log(1+e**x)

    #                 dropout1_p=dropout1_p,

    #                 hidden1_num_units=hidden1_num_units,
    #                 hidden1_W=Normal(std=(1.0/(hidden0_num_units*(1-dropout1_p)))**0.5),
    #                 hidden1_b=Normal(std=(1.0/(hidden0_num_units*(1-dropout1_p)))**0.5),
    #                 hidden1_nonlinearity=softplus,

    #                 dropout2_p=dropout2_p,

    #                 output_num_units=2, # binary classification
    #                 output_nonlinearity=softmax,
    #                 # max-norm regularization, large decaying learning rates and high momentum
    #                 update=momentum, # try momentum
    #                 update_learning_rate=0.04,
    #                 update_momentum = 0.95, # 0.95 to 0.99 usually works well with dropout [0.5, 0.9, 0.95, 0.99]
    #                 objective=objective,
    #                 objective_loss_function=categorical_crossentropy,
    #                 # objective_deterministic=False,
    #                 objective_l2=1,
    #                 # batch_iterator_train=BatchIterator(batch_size=100),
    #                 train_split=TrainSplit(eval_size=0.2),
    #                 verbose=1,
    #                 max_epochs=500)


    # net1.fit(train, y, 50) # num of epochs
    # preds = net1.predict_proba(test)[:, 1]
    # return preds

    # for i in range(epochs):
    #     net1.fit(train, y)
    #     preds = net1.predict_proba(test)[:, 1]
    #     auc = roc_auc_score(y2, preds)
    #     print "auc %s: %s"%(i, auc)
    # return preds

    # Comment out second layer for run time.
    layers = [('input', InputLayer),
           # ('dropout0', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout1', DropoutLayer),
           # ('dense1', DenseLayer),
           # ('dropout2', DropoutLayer),
           # ('dense2', DenseLayer),
           # ('dropout3', DropoutLayer),
           ('output', DenseLayer)
           ]

    net1 = NeuralNet(layers=layers,
                 input_shape=(None, n_features),
                 # dropout0_p=0.1, # no is better
                 dense0_num_units=800, # 512, - reduce num units to make faster
                 # dense0_W=GlorotUniform(), # not very helpful
                 # dense0_W=Normal(std=(1.0/n_features)**0.5), # small weights in case of saturation
                 dense0_nonlinearity=rectify, #
                 dropout1_p=0.7, # 0.7

                 # dense1_num_units=512,
                 # dense1_nonlinearity=rectify,
                 # dropout2_p=0.7, # 0.7

                 # dense2_num_units=200,
                 # dense2_nonlinearity=softplus,
                 # dropout3_p=0.5, #

                 output_num_units=2,
                 output_nonlinearity=softmax,
                 update=nesterov_momentum,
                 # update=momentum,
                 update_momentum = 0.6, # 0.7
                 update_learning_rate=0.05, # 0.02

                 # objective_l1=0.01, # not helpful
                 batch_iterator_train=BatchIterator(batch_size=512),
                 batch_iterator_test=BatchIterator(batch_size=512),
                 eval_size=0.0,
                 verbose=1,
                 max_epochs=1)
    for i in range(1000):
        net1.fit(train, y)
        pred = net1.predict_proba(test)[:,1]
        print roc_auc_score(y2,pred)
    return pred

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


def voting_classifier(train, y, test):
    clf1 = xgb.XGBClassifier(n_estimators=25,
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

    clf2 = xgb.XGBClassifier(n_estimators=25,
                            nthread=-1,
                            max_depth=16, # 16
                            learning_rate=0.03, # 0.03
                            min_child_weight=2, # 2
                            silent=True,
                            # gamma=0, # 0
                            # colsample_bylevel=1, # 1
                            # scale_pos_weight=1, # 1
                            subsample=0.83, # 0.83
                            colsample_bytree=0.83) # 0.83

    clf3 = xgb.XGBClassifier(n_estimators=25,
                            nthread=-1,
                            max_depth=16, # 16
                            learning_rate=0.03, # 0.03
                            min_child_weight=2, # 2
                            silent=True,
                            # gamma=0, # 0
                            # colsample_bylevel=1, # 1
                            # scale_pos_weight=1, # 1
                            subsample=0.83, # 0.83
                            colsample_bytree=0.81) # 0.81

    eclf = VotingClassifier(estimators=[('xgboost1', clf1),
                            ('xgboost2', clf2),
                            ('xgboost3', clf3)],
                            voting='soft',
                            weights=[1, 1, 1])

    eclf.fit(train, y)
    preds = eclf.predict_proba(test)[:, 1]

    return preds

def ada_boost(train, y, test, estimator):
    if estimator == 'GaussianNB':
        base_est = GaussianNB()
    elif estimator == 'logistic_regression':
        base_est = LogisticRegression(penalty='l2',
                            dual=False,
                            tol=0.0001,
                            C=1.0,
                            fit_intercept=True,
                            intercept_scaling=1,
                            class_weight=None,
                            random_state=None,
                            solver='sag', # 'sag' fast convergence is only guaranteed on features with approximately the same scale.
                            max_iter=1000,
                            multi_class='ovr',
                            verbose=1,
                            # warm_start=False,
                    )
    else:
        raise ValueError('ERROR: Unexpected argument estimator: %s'%estimator)

    bdt = AdaBoostClassifier(base_estimator=base_est,
                            n_estimators=50,
                            # algorithm="SAMME.R",
                            learning_rate=0.5,
                            # random_state=None,
                            )
    bdt.fit(train, y)
    preds = bdt.predict_proba(test)[:, 1]
    return preds

def bagging(train, y, test, estimator):
    if estimator == 'NN':
        n_features = train.shape[1]

        # Convert to np.array to make lasagne happy
        train = train.as_matrix().astype(np.float32)
        test = test.as_matrix().astype(np.float32)
        y = y.astype(np.int32)

        scaler = StandardScaler()
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

        layers = [('input', InputLayer),
           # ('dropout0', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout1', DropoutLayer),
           # ('dense1', DenseLayer),
           # ('dropout2', DropoutLayer),
           # ('dense2', DenseLayer),
           # ('dropout3', DropoutLayer),
           ('output', DenseLayer)
           ]

        net1 = NeuralNet(layers=layers,
                 input_shape=(None, n_features),
                 # dropout0_p=0.2, # no is better
                 dense0_num_units=200, # 512, - reduce num units to make faster
                 # dense0_W=GlorotUniform(gain='relu'),
                 # dense0_W=Normal(std=(1.0/n_features)**0.5), # small weights in case of saturation
                 # dense0_b=Normal(std=(1.0/n_features)**0.5),
                 dense0_nonlinearity=softplus,
                 dropout1_p=0.7, # 0.7

                 # dense1_num_units=200,
                 # dense1_nonlinearity=softplus,
                 # dropout2_p=0.7, # 0.0

                 # dense2_num_units=200,
                 # dense2_nonlinearity=softplus,
                 # dropout3_p=0.7, #

                 output_num_units=2,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 # update=momentum,
                 # update_momentum = 0.7, # 0.7
                 update_learning_rate=0.04, # 0.02

                 # objective_l2=0.001, # not helpful

                 eval_size=0.0,

                 # objective_loss_function = binary_accuracy,
                 verbose=1,
                 max_epochs=110)
        base_est = net1
        bc = BaggingClassifier(base_estimator=base_est,
                    n_estimators=10,
                    max_samples=0.9,
                    max_features=0.9,
                    bootstrap=True,
                    bootstrap_features=True,
                    oob_score=True,
                    # warm_start=False,
                    # random_state=None,
                    verbose=1)
        # import pdb;pdb.set_trace()
        bc.fit(train, y)

    else:
        raise ValueError('ERROR: Unexpected argument estimator: %s'%estimator)


    preds = bc.predict_proba(test)[:, 1]
    return preds
