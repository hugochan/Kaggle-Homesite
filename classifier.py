# import pandas as pd
# import numpy as np
import xgboost as xgb

seed = 260681

def boosted_trees(train, y, test):
    clf = xgb.XGBClassifier(n_estimators=25,
                            nthread=-1,
                            max_depth=10,
                            learning_rate=0.025,
                            silent=True,
                            subsample=0.8,
                            colsample_bytree=1)

    xgb_model = clf.fit(train, y, eval_metric="auc")

    preds = clf.predict_proba(test)[:,1]
    # sample = pd.read_csv('datasets/sample_submission.csv')
    # sample.QuoteConversion_Flag = preds
    # sample.to_csv('xgb_benchmark.csv', index=False)
    return preds


# If you care only about the ranking order (AUC) of your prediction
# Balance the positive and negative weights, via scale_pos_weight
# Use AUC for evaluation
