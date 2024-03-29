# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example

library(readr)
library(xgboost)
library(foreach)
library(pROC)

#my favorite seed^^
set.seed(1718)

cat("reading the train and test data\n")
train <- read_csv("../datasets/train.csv")
test  <- read_csv("../datasets/test.csv")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- -1
test[is.na(test)]   <- -1

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)


# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Original_Quote_Date, "%m"))
train$year <- as.integer(format(train$Original_Quote_Date, "%y"))
train$day <- weekdays(as.Date(train$Original_Quote_Date))

# removing the date column
train <- train[,-c(2)]

# seperating out the elements of the date column for the train set
test$month <- as.integer(format(test$Original_Quote_Date, "%m"))
test$year <- as.integer(format(test$Original_Quote_Date, "%y"))
test$day <- weekdays(as.Date(test$Original_Quote_Date))

# removing the date column
test <- test[,-c(2)]


feature.names <- names(train)[c(3:301)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)

#training-validation split
split_ratio <- 0.7
training_rows <- sample(nrow(train), size=nrow(train)*split_ratio)
training_set <- train[training_rows,]
val_set <- train[-c(training_rows),]

#real training 
training_set <- train

n_iter <- 1
sample_ratio = 0.75
for (m in 1:n_iter) {
  sampleRows <- sample(nrow(training_set), size=sample_ratio*nrow(training_set))
  train <- training_set[sampleRows,]
  nrow(train)
  tra<-train[,feature.names]
  h<-sample(nrow(train),2000)
  
  dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$QuoteConversion_Flag[h])
  #dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$QuoteConversion_Flag[-h])
  dtrain<-xgb.DMatrix(data=data.matrix(tra),label=train$QuoteConversion_Flag)
  
  watchlist<-list(val=dval,train=dtrain)
  param <- list(  objective           = "binary:logistic", 
                  booster = "gbtree",
                  eval_metric = "auc",
                  eta                 = 0.023, # 0.06, #0.01,
                  max_depth           = 6, #changed from default of 8
                  subsample           = 0.83, # 0.7
                  colsample_bytree    = 0.77 # 0.7
                  #num_parallel_tree   = 2
                  # alpha = 0.0001, 
                  # lambda = 1
  )
  
  clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = 1, 
                      verbose             = 0,  #1
                      #early.stop.round    = 150,
                      #watchlist           = watchlist,
                      maximize            = FALSE
  )
  
  pred1 <- predict(clf, data.matrix(val_set[,feature.names]))
  
  #pred1 <- predict(clf, data.matrix(test[,feature.names]))
  #pred1 <- data.frame(pred1)
  #submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
  #cat("saving the submission file\n")
  #write_csv(submission, paste("bagging_xgb_stop_3000_",m,'.csv',sep=''))
  
}

n_iter <- 1
sample_ratio = 1
for (m in 1:n_iter) {
  sampleRows <- sample(nrow(training_set), size=sample_ratio*nrow(training_set))
  train <- training_set[sampleRows,]
  nrow(train)
  tra<-train[,feature.names]
  h<-sample(nrow(train),2000)
  
  dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$QuoteConversion_Flag[h])
  #dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$QuoteConversion_Flag[-h])
  dtrain<-xgb.DMatrix(data=data.matrix(tra),label=train$QuoteConversion_Flag)
  
  watchlist<-list(val=dval,train=dtrain)
  param <- list(  objective           = "binary:logistic", 
                  booster = "gbtree",
                  eval_metric = "auc",
                  eta                 = 0.023, # 0.06, #0.01,
                  max_depth           = 6, #changed from default of 8
                  subsample           = 0.83, # 0.7
                  colsample_bytree    = 0.77 # 0.7
                  #num_parallel_tree   = 2
                  # alpha = 0.0001, 
                  # lambda = 1
  )
  
  clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = 1, 
                      verbose             = 0,  #1
                      #early.stop.round    = 150,
                      #watchlist           = watchlist,
                      maximize            = FALSE
  )
  
  pred1_1 <- predict(clf, data.matrix(val_set[,feature.names]))
  
  #pred1 <- predict(clf, data.matrix(test[,feature.names]))
  #pred1 <- data.frame(pred1)
  #submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
  #cat("saving the submission file\n")
  #write_csv(submission, paste("bagging_xgb_stop_3000_",m,'.csv',sep=''))
  
}



#pred1 = rowMeans(data.frame(pred1)) # averaging

# calc auc
auc(val_set$QuoteConversion_Flag, pred1) # 0.75

auc(val_set$QuoteConversion_Flag, pred1_1) # 1

auc(val_set$QuoteConversion_Flag, (pred1+pred1_1)/2)

#submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
#cat("saving the submission file\n")
#write_csv(submission, "bagging_xgb_stop_date0124.csv")

