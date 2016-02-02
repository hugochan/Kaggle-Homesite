import pandas as pd

pred_1=pd.read_csv('bagging_xgb_stop_1800_25.csv')
pred_2=pd.read_csv('bagging_xgb_stop_1800_26.csv')
pred_3=pd.read_csv('bagging_xgb_stop2_1800_1.csv')
pred_4=pd.read_csv('bagging_xgb_stop2_1800_2.csv')
pred_5=pd.read_csv('bagging_xgb_stop2_1800_3.csv')
pred_6=pd.read_csv('bagging_xgb_stop2_1800_4.csv')
pred_7=pd.read_csv('bagging_xgb_stop2_1800_5.csv')
pred_8=pd.read_csv('bagging_xgb_stop2_1800_6.csv')
pred_9=pd.read_csv('bagging_xgb_stop2_1800_7.csv')

pred1=pd.read_csv('bagging_xgb_stop_3000_1.csv')
pred2=pd.read_csv('bagging_xgb_stop_3000_2.csv')
pred3=pd.read_csv('bagging_xgb_stop_3000_3.csv')
pred4=pd.read_csv('bagging_xgb_stop_3000_4.csv')
pred5=pd.read_csv('bagging_xgb_stop_3000_5.csv')

pred6=pd.read_csv('bagging_xgb_stop_1.csv')
pred7=pd.read_csv('bagging_xgb_stop_2.csv')
pred8=pd.read_csv('bagging_xgb_stop_3.csv')
pred9=pd.read_csv('bagging_xgb_stop_4.csv')
pred10=pd.read_csv('bagging_xgb_stop_5.csv')
pred11=pd.read_csv('bagging_xgb_stop_6.csv')
pred12=pd.read_csv('bagging_xgb_stop_7.csv')
pred13=pd.read_csv('bagging_xgb_stop_8.csv')
pred14=pd.read_csv('bagging_xgb_stop_9.csv')
pred15=pd.read_csv('bagging_xgb_stop_10.csv')

pred16=pd.read_csv('submissions/submission_xgb_stop_R (0.96809).csv')


# p_weighted=(0.02*pred1.pred1+0.02*pred2.pred1+0.02*pred3.pred1+0.02*pred4.pred1+0.02*pred5.pred1+\
#     0.005*pred6.pred1+0.005*pred7.pred1+0.005*pred8.pred1+0.005*pred9.pred1+\
#     0.005*pred10.pred1+0.005*pred11.pred1+0.005*pred12.pred1+0.005*pred13.pred1+\
#     0.005*pred14.pred1+0.005*pred15.pred1+0.805*pred16.QuoteConversion_Flag+0.005*pred_1.pred1+\
#     0.005*pred_2.pred1+0.005*pred_3.pred1+0.005*pred_4.pred1+0.005*pred_5.pred1+\
#     0.005*pred_6.pred1+0.005*pred_7.pred1+0.005*pred_8.pred1+0.005*pred_9.pred1)

# 96.811%

# p_weighted=(
#     0.01*pred6.pred1+0.01*pred7.pred1+0.01*pred8.pred1+0.01*pred9.pred1+\
#     0.01*pred10.pred1+0.01*pred11.pred1+0.01*pred12.pred1+0.01*pred13.pred1+\
#     0.01*pred14.pred1+0.01*pred15.pred1+0.81*pred16.QuoteConversion_Flag+0.01*pred_1.pred1+\
#     0.01*pred_2.pred1+0.01*pred_3.pred1+0.01*pred_4.pred1+0.01*pred_5.pred1+\
#     0.01*pred_6.pred1+0.01*pred_7.pred1+0.01*pred_8.pred1+0.01*pred_9.pred1)

# 96.809%

p_weighted=(0.02*pred1.pred1+0.02*pred2.pred1+0.02*pred3.pred1+0.02*pred4.pred1+0.02*pred5.pred1+\
    0.015*pred6.pred1+0.015*pred7.pred1+0.015*pred8.pred1+0.015*pred9.pred1+\
    0.015*pred10.pred1+0.015*pred11.pred1+0.015*pred12.pred1+0.015*pred13.pred1+\
    0.015*pred14.pred1+0.015*pred15.pred1+0.615*pred16.QuoteConversion_Flag+0.015*pred_1.pred1+\
    0.015*pred_2.pred1+0.015*pred_3.pred1+0.015*pred_4.pred1+0.015*pred_5.pred1+\
    0.015*pred_6.pred1+0.015*pred_7.pred1+0.015*pred_8.pred1+0.015*pred_9.pred1)




sample = pd.read_csv("../datasets/sample_submission.csv", header=0, delimiter=',')
sample.QuoteConversion_Flag = p_weighted
sample.to_csv('bagging_xgb_weighted25_2.csv', index=False)


