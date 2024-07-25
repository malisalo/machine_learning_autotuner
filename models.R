library(dplyr)
library(e1071)
library(randomForest)
library(gbm)
library(caret)

# Gradient Boosting
create_gbm <- function(data, predicting_var, training_split, models_amount) {
  set.seed(123)
  trainIndex <- createDataPartition(data[[predicting_var]],
                                    p = training_split,
                                    list = FALSE,
                                    times = 1)
  train <- data[trainIndex, ]
  test <- data[-trainIndex, ]
  
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
  
  # Print the best parameters
  print(gbm_model$bestTune)
  
  # Save the best parameters
  best_params <- gbm_model$bestTune
  
  predictions_gbm <- predict(gbm_model, newdata = test)
  cm_gbm <- confusionMatrix(predictions_gbm, test[[predicting_var]])
  
  list(
    accuracy = round(cm_gbm$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_gbm$table,
    best_params = best_params
  )
}


# Random Forest
create_rf <- function(data, predicting_var, training_split, models_amount) {
  set.seed(123)
  
  # Split the data into training and test sets
  trainIndex <- createDataPartition(data[[predicting_var]], 
                                    p = training_split, 
                                    list = FALSE, 
                                    times = 1)
  train <- data[trainIndex, ]
  test <- data[-trainIndex, ]
  
  # Train control with cross-validation
  train_control <- trainControl(method = "cv", number = 5, search = "random")
  
  # Train the Random Forest model with hyperparameter tuning
  rf_model <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "rf",
    metric = "Accuracy",
    tuneLength = models_amount,
    trControl = train_control
  )
  
  # Print the best parameters
  print(rf_model$bestTune)
  
  # Use the best model to make predictions on the test set
  predictions_rf <- predict(rf_model, newdata = test)
  
  # Calculate confusion matrix
  cm_rf <- confusionMatrix(predictions_rf, test[[predicting_var]])
  
  # Return confusion matrix and overall accuracy
  list(
    accuracy = round(cm_rf$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_rf$table,
    best_params = rf_model$bestTune
  )
  
}

# K Nearest Neighbors
create_knn <- function(data, predicting_var, training_split, models_amount) {
  # Split the data into training and test sets
  set.seed(123)
  trainIndex <- createDataPartition(data[[predicting_var]],
                                    p = training_split,
                                    list = FALSE,
                                    times = 1)
  train <- data[trainIndex, ]
  test <- data[-trainIndex, ]
  
  # Train control with cross-validation
  train_control <- trainControl(method = "cv", number = 5, search = "grid")
  
  # Train the KNN model with hyperparameter tuning
  knn_tuned <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "knn",
    metric = "Accuracy",
    tuneLength = models_amount,
    trControl = train_control
  )
  
  # Print the best parameters
  print(knn_tuned$bestTune)
  
  # Use the best model to make predictions on the test set
  predictions_knn <- predict(knn_tuned, newdata = test)
  
  # Calculate confusion matrix
  cm_knn <- confusionMatrix(predictions_knn, test[[predicting_var]])
  
  # Return confusion matrix and overall accuracy
  list(
    accuracy = round(cm_knn$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_knn$table,
    best_params = knn_tuned$bestTune
  )
}

# SVM Classifier
create_svm <- function(data, predicting_var, training_split, models_amount) {
  data[[predicting_var]] <- as.factor(data[[predicting_var]])
  
  set.seed(123)
  trainIndex <- createDataPartition(data[[predicting_var]],
                                    p = training_split,
                                    list = FALSE,
                                    times = 1)
  train <- data[trainIndex, ]
  test <- data[-trainIndex, ]
  
  # Define cross-validation method
  train_control <- trainControl(method = "cv", number = 5)
  
  # Train the SVM model with hyperparameter tuning
  svm_model <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "svmRadial",
    trControl = train_control,
    tuneLength = models_amount
  )
  
  # Print the best parameters
  print(svm_model$bestTune)
  
  # Make predictions
  predictions <- predict(svm_model, newdata = test)
  
  # Evaluate model
  cm <- confusionMatrix(predictions, test[[predicting_var]])
  
  list(
    accuracy = round(cm$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm$table,
    best_params = svm_model$bestTune
  )
}
