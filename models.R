library(dplyr)
library(e1071)
library(randomForest)
library(gbm)
library(caret)
library(DALEX)
library(ranger)

# Gradient Boosting
create_gbm <- function(train, test, predicting_var, models_amount) {
  set.seed(123)
  
  train_control <- trainControl(method = "cv", number = 5, savePredictions = "final")
  
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
  
  predictions_gbm <- predict(gbm_model, newdata = test)
  cm_gbm <- confusionMatrix(predictions_gbm, test[[predicting_var]])
  
  variable_importance <- varImp(gbm_model)$importance
  
  list(
    accuracy = round(cm_gbm$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_gbm$table,
    best_params = gbm_model$bestTune,
    variable_importance = variable_importance,
    cv_results = gbm_model$resample$Accuracy
  )
}


create_rf <- function(train, test, predicting_var, models_amount) {
  library(caret)
  library(ranger)
  
  set.seed(123)
  
  train_control <- trainControl(
    method = "cv",
    number = 5,
    search = "random",
    savePredictions = "final"
  )
  
  # Define the grid of hyperparameters to search (excluding num.trees)
  rf_grid <- expand.grid(
    mtry = seq(2, ncol(train) - 1, by = 1),
    splitrule = c("gini", "extratrees"),
    min.node.size = seq(1, 10, by = 1)
  )
  
  # Define the range of num.trees values
  num_trees_values <- c(100, 500, 1000)
  
  # Generate all possible combinations of hyperparameters
  all_combinations <- expand.grid(
    mtry = rf_grid$mtry,
    splitrule = rf_grid$splitrule,
    min.node.size = rf_grid$min.node.size
  )
  
  # Check if models_amount is larger than the number of possible combinations
  total_combinations <- nrow(all_combinations) * length(num_trees_values)
  
  if (models_amount > total_combinations) {
    warning(paste("models_amount is greater than the number of possible combinations.",
                  "Setting models_amount to", total_combinations))
    models_amount <- total_combinations
  }
  
  # Randomly sample N combinations
  sampled_indices <- sample(nrow(all_combinations), size = models_amount, replace = TRUE)
  sampled_combinations <- all_combinations[sampled_indices, ]
  sampled_combinations$num.trees <- sample(num_trees_values, models_amount, replace = TRUE)
  
  best_accuracy <- -Inf  # Initialize best accuracy
  best_combination <- NULL
  best_model <- NULL
  
  for (i in 1:nrow(sampled_combinations)) {
    # Extract parameters for current model
    current_params <- sampled_combinations[i, ]
    num_trees <- current_params$num.trees
    
    cat("Training model with parameters: mtry =", current_params$mtry, 
        ", splitrule =", current_params$splitrule, 
        ", min.node.size =", current_params$min.node.size, 
        ", num.trees =", num_trees, "\n")
    
    rf_model <- train(
      as.formula(paste(predicting_var, "~ .")),
      data = train,
      method = "ranger",
      metric = "Accuracy",
      tuneGrid = current_params[, c("mtry", "splitrule", "min.node.size")],  # Ensure tuneGrid has the correct columns
      trControl = train_control,
      num.trees = num_trees,  # Specify num.trees
      importance = 'impurity'  # Request variable importance
    )
    
    # Get the best accuracy for the current model
    current_best_accuracy <- max(rf_model$results$Accuracy)
    
    # Print the best accuracy for the current model
    cat("Current best accuracy with num.trees =", num_trees, "is", current_best_accuracy, "\n")
    
    # Update best model and parameters if current model is better
    if (current_best_accuracy > best_accuracy) {
      best_accuracy <- current_best_accuracy
      best_combination <- current_params
      best_model <- rf_model
    }
  }
  
  # Print the best number of trees and best parameters
  cat("Best number of trees:", best_combination$num.trees, "\n")
  cat("Best model parameters:\n")
  print(best_combination[, c("mtry", "splitrule", "min.node.size")])
  
  # Make predictions and compute metrics
  predictions_rf <- predict(best_model, newdata = test)
  cm_rf <- confusionMatrix(predictions_rf, test[[predicting_var]])
  
  variable_importance <- varImp(best_model, scale = FALSE)$importance
  
  list(
    accuracy = round(cm_rf$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_rf$table,
    best_params = best_combination[, c("mtry", "splitrule", "min.node.size")],
    variable_importance = variable_importance,
    cv_results = best_model$resample$Accuracy
  )
}


# SVM
create_svm <- function(train, test, predicting_var, models_amount) {
  set.seed(123)
  
  train_control <- trainControl(method = "cv", number = 5, savePredictions = "final")
  
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
  
  variable_importance <- varImp(svm_model, scale = FALSE)$importance
  
  list(
    accuracy = round(cm_svm$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_svm$table,
    best_params = svm_model$bestTune,
    variable_importance = variable_importance,
    cv_results = svm_model$resample$Accuracy
  )
}
