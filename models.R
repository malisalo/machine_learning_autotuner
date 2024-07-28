library(dplyr)
library(e1071)
library(randomForest)
library(gbm)
library(caret)
library(DALEX)

# Gradient Boosting
create_gbm <- function(train, test, predicting_var, models_amount) {
  set.seed(123)
  
  train_control <- trainControl(method = "cv", number = 5)
  
  gbm_model <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "gbm",
    metric = "Accuracy",
    tuneLength = models_amount,
    trControl = train_control,
    verbose = FALSE
  )
  
  print(gbm_model$bestTune)
  
  best_params <- gbm_model$bestTune
  predictions_gbm <- predict(gbm_model, newdata = test)
  cm_gbm <- confusionMatrix(predictions_gbm, test[[predicting_var]])
  
  variable_importance <- varImp(gbm_model)$importance
  
  list(
    accuracy = round(cm_gbm$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_gbm$table,
    best_params = best_params,
    variable_importance = variable_importance
  )
}

# Random Forest
create_rf <- function(train, test, predicting_var, models_amount) {
  set.seed(123)
  
  train_control <- trainControl(method = "cv", number = 5, search = "random")
  
  rf_model <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "rf",
    metric = "Accuracy",
    tuneLength = models_amount,
    trControl = train_control
  )
  
  print(rf_model$bestTune)
  
  predictions_rf <- predict(rf_model, newdata = test)
  cm_rf <- confusionMatrix(predictions_rf, test[[predicting_var]])
  
  variable_importance <- varImp(rf_model)$importance
  
  list(
    accuracy = round(cm_rf$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_rf$table,
    best_params = rf_model$bestTune,
    variable_importance = variable_importance
  )
}

# K Nearest Neighbors
create_knn <- function(train, test, predicting_var, models_amount) {
  set.seed(123)
  
  train_control <- trainControl(method = "cv", number = 5, search = "grid")
  
  knn_tuned <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "knn",
    metric = "Accuracy",
    tuneLength = models_amount,
    trControl = train_control
  )
  
  print(knn_tuned$bestTune)
  
  predictions_knn <- predict(knn_tuned, newdata = test)
  cm_knn <- confusionMatrix(predictions_knn, test[[predicting_var]])
  
  list(
    accuracy = round(cm_knn$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_knn$table,
    best_params = knn_tuned$bestTune
  )
}

# SVM
create_svm <- function(train, test, predicting_var, models_amount) {
  set.seed(123)
  
  train_control <- trainControl(method = "cv", number = 5, search = "grid")
  
  svm_model <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "svmRadial",
    metric = "Accuracy",
    tuneLength = models_amount,
    trControl = train_control
  )
  
  print(svm_model$bestTune)
  
  predictions_svm <- predict(svm_model, newdata = test)
  cm_svm <- confusionMatrix(predictions_svm, test[[predicting_var]])
  
  # Variable importance for SVM
  variable_importance <- varImp(svm_model, scale = FALSE)$importance
  
  list(
    accuracy = round(cm_svm$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_svm$table,
    best_params = svm_model$bestTune,
    variable_importance = variable_importance
  )
}

