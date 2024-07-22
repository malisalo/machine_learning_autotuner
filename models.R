library(dplyr)
library(e1071)
library(randomForest)
library(gbm)
library(caret)

create_gbm <- function(data, predicting_var, training_split) {
  set.seed(123)
  trainIndex <- createDataPartition(data[[predicting_var]],
                                    p = training_split,
                                    list = FALSE,
                                    times = 1)
  train <- data[trainIndex, ]
  test <- data[-trainIndex, ]
  
  tune_grid <- expand.grid(
    n.trees = c(50, 100, 150),
    interaction.depth = c(1, 3, 5),
    shrinkage = c(0.01, 0.1, 0.3),
    n.minobsinnode = 10
  )
  train_control <- trainControl(method = "cv", number = 5)
  
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
create_rf <- function(data, predicting_var, training_split) {
  # Split the data
  set.seed(123)
  trainIndex <- createDataPartition(data[[predicting_var]],
                                    p = training_split,
                                    list = FALSE,
                                    times = 1)
  train_data <- data[trainIndex, ]
  test_data <- data[-trainIndex, ]
  
  # Define the task
  task <- makeClassifTask(data = train_data, target = predicting_var)
  
  # Define the learner
  rf_learner <- makeLearner("classif.randomForest", predict.type = "response")
  
  # Define parameter space
  param_set <- makeParamSet(
    makeDiscreteParam("mtry", values = c(4)),
    makeDiscreteParam("ntree", values = c(755)),
    makeDiscreteParam("nodesize", values = c(10)),
    makeDiscreteParam("maxnodes", values = c(30))
  )
  
  # Define resampling strategy
  resampling <- makeResampleDesc("CV", iters = 5)
  
  # Define tuning method
  tuner <- makeTuneControlGrid()
  
  # Tune the parameters
  tuned_rf <- tuneParams(
    learner = rf_learner,
    task = task,
    resampling = resampling,
    par.set = param_set,
    control = tuner,
    measures = list(acc = acc)
  )
  
  # Extract the best parameters
  best_params <- tuned_rf$x
  
  # Print the best parameters
  print(best_params)
  
  # Train the model with the best parameters
  rf_model <- setHyperPars(rf_learner, par.vals = best_params)
  rf_model <- train(rf_model, task)
  
  # Predict on the test set
  test_task <- makeClassifTask(data = test_data, target = predicting_var)
  predictions <- predict(rf_model, newdata = test_task)
  
  # Evaluate performance
  cm_rf <- confusionMatrix(predictions$truth, predictions$predict)
  
  # Return the results
  list(
    accuracy = round(cm_rf$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_rf$table,
    best_params = best_params
  )
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
  tune_grid <- expand.grid(k = 1:100)
  
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
  
  # Return confusion matrix and overall accuracy
  list(
    accuracy = round(cm_knn$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_knn$table,
    best_params = knn_tuned$bestTune
  )
}

# SVM Classifier
create_svm <- function(data, predicting_var, training_split) {
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
  
  # Define hyperparameter grid for tuning
  svm_grid <- expand.grid(C = 2 ^ (-5:2), sigma = 2 ^ (-15:3)) # Radial basis function kernel parameters
  
  # Train the SVM model with hyperparameter tuning
  svm_model <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train,
    method = "svmRadial",
    trControl = train_control,
    tuneGrid = svm_grid
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
