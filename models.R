library(shiny)
library(bslib)
library(caret)
library(missForest)
library(dplyr)
library(e1071)
library(randomForest)
library(gbm)
library(mice)
library(Hmisc)

# Gradient Boosting
create_gbm <- function(data, predicting_var, training_split) {
  # Split the data into training and test sets
  set.seed(123)
  trainIndex <- createDataPartition(data[[predicting_var]],
                                    p = training_split,
                                    list = FALSE,
                                    times = 1)
  train <- data[trainIndex, ]
  test <- data[-trainIndex, ]
  
  # Set up the tuning grid for hyperparameters
  tune_grid <- expand.grid(
    n.trees = c(50, 100, 150),
    interaction.depth = c(1, 3, 5),
    shrinkage = c(0.01, 0.1, 0.3),
    n.minobsinnode = 10
  )
  
  # Train control with cross-validation
  train_control <- trainControl(method = "cv", number = 5)
  
  # Train the Gradient Boosting model with hyperparameter tuning
  gbm_model <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "gbm",
    metric = "Accuracy",
    tuneGrid = tune_grid,
    trControl = train_control,
    verbose = FALSE
  )
  
  # Print the best parameters
  print(gbm_model$bestTune)
  
  # Use the best model to make predictions on the test set
  predictions_gbm <- predict(gbm_model, newdata = test)
  
  # Calculate confusion matrix
  cm_gbm <- confusionMatrix(predictions_gbm, test[[predicting_var]])
  
  # Print confusion matrix and overall accuracy
  result <- paste("Gradient Boosting - Accuracy:",
                  round(cm_gbm$overall['Accuracy'] * 100, 2),
                  "%")
  return(result)
}

# Random Forest
create_rf <- function(data, predicting_var, training_split) {
  set.seed(123)
  train_index <- createDataPartition(data[[predicting_var]],
                                     p = training_split,
                                     list = FALSE,
                                     times = 1)
  train <- data[train_index, ]
  test <- data[-train_index, ]
  
  # Train a Random Forest model with hyperparameter tuning
  tune_grid <- expand.grid(
    mtry = c(1:5),
    ntree = c(155, 255, 355, 555, 755, 1005),
    nodesize = c(1, 5, 10),
    maxnodes = c(10, 20, 30)
  )
  
  train_control <- trainControl(method = "cv", number = 5)
  
  model_rf <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = 'rf',
    metric = "Accuracy",
    tuneGrid = tune_grid,
    trControl = train_control,
    importance = TRUE
  )
  
  predictions <- predict(model_rf, newdata = test)
  predictions <- factor(predictions, levels = levels(test[[predicting_var]]))
  cm_rf <- confusionMatrix(predictions, test[[predicting_var]])
  
  # Print confusion matrix and overall accuracy
  result <- paste("Random Forest - Accuracy:",
                  round(cm_rf$overall['Accuracy'] * 100, 2),
                  "%")
  return(result)
}

# KNN
create_knn <- function(data, predicting_var, training_split) {
  # Split the data into training and test sets
  set.seed(123)
  trainIndex <- createDataPartition(data[[predicting_var]],
                                    p = training_split,
                                    list = FALSE,
                                    times = 1)
  train <- data[trainIndex, ]
  test <- data[-trainIndex, ]
  
  # Set up the tuning grid for k
  tune_grid <- expand.grid(k = c(1:20))
  
  # Train control with cross-validation
  train_control <- trainControl(method = "cv",
                                number = 5,
                                search = "grid")
  
  # Train the KNN model with hyperparameter tuning
  knn_tuned <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "knn",
    metric = "Accuracy",
    tuneGrid = tune_grid,
    trControl = train_control
  )
  
  # Print the best parameters
  print(knn_tuned$bestTune)
  
  # Use the best model to make predictions on the test set
  predictions_knn <- predict(knn_tuned, newdata = test)
  
  # Calculate confusion matrix
  cm_knn <- confusionMatrix(predictions_knn, test[[predicting_var]])
  
  # Print confusion matrix and overall accuracy
  result <- paste("KNN - Accuracy:", round(cm_knn$overall['Accuracy'] * 100, 2), "%")
  return(result)
}

# SVM Classifier
create_svm <- function(data, predicting_var, training_split) {
  # Convert the target variable to a factor for classification
  data[[predicting_var]] <- as.factor(data[[predicting_var]])
  
  # Split the data into training and test sets
  set.seed(123)
  trainIndex <- createDataPartition(data[[predicting_var]],
                                    p = .8,
                                    list = FALSE,
                                    times = 1)
  train <- data[trainIndex, ]
  test <- data[-trainIndex, ]
  
  # Train control with cross-validation
  train_control <- trainControl(method = "cv", number = 5)
  
  # Train the SVM model
  svm_model <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "svmRadial",
    trControl = train_control
  )
  
  # Make predictions
  predictions <- predict(svm_model, newdata = test)
  
  # Calculate confusion matrix
  cm <- confusionMatrix(predictions, test[[predicting_var]])
  
  # Print confusion matrix and accuracy
  result <- paste("SVM - Accuracy:", round(cm$overall['Accuracy'] * 100, 2), "%")
  return(result)
}