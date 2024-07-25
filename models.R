library(shiny)
library(bslib)
library(caret)
library(randomForest)
library(e1071)
library(class)
library(ggplot2)
library(corrplot)
library(reshape2)
library(tidyverse)
library(doParallel)
library(parallel)  # Ensure parallel library is loaded

# Function to create and train Random Forest model
create_rf <- function(data, numerical_vars, categorical_vars, predicting_var, training_split) {
  set.seed(123)
  train_index <- createDataPartition(data[[predicting_var]], p = training_split, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # Convert categorical variables to factors
  for (var in categorical_vars) {
    train_data[[var]] <- as.factor(train_data[[var]])
    test_data[[var]] <- as.factor(test_data[[var]])
  }
  
  # Ensure levels in factors are consistent
  train_data[[predicting_var]] <- factor(train_data[[predicting_var]], levels = levels(data[[predicting_var]]))
  test_data[[predicting_var]] <- factor(test_data[[predicting_var]], levels = levels(data[[predicting_var]]))
  
  # Hyperparameter tuning grid
  tune_grid <- expand.grid(mtry = seq(1, length(numerical_vars) + length(categorical_vars), by = 1))
  control <- trainControl(method = "cv", number = 10, search = "grid")
  
  # Parallel processing
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  
  model_rf <- train(
    as.formula(paste(predicting_var, "~ .")),
    data = train_data,
    method = 'rf',
    trControl = control,
    tuneGrid = tune_grid,
    ntree = 500,
    importance = TRUE
  )
  
  stopCluster(cl)
  
  predictions <- predict(model_rf, test_data)
  predictions <- factor(predictions, levels = levels(test_data[[predicting_var]]))
  cm_rf <- confusionMatrix(predictions, test_data[[predicting_var]])
  
  var_imp <- varImp(model_rf)
  numeric_data <- data[, sapply(data, is.numeric)]
  correlation_matrix <- cor(numeric_data, use = "complete.obs")
  
  result <- list(
    accuracy = round(cm_rf$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_rf,
    model = model_rf,
    var_imp = var_imp,
    correlation_matrix = correlation_matrix
  )
  
  return(result)
}

# Function to create and train K-Nearest Neighbors model
create_knn <- function(data, numerical_vars, categorical_vars, predicting_var, training_split) {
  set.seed(123)
  train_index <- createDataPartition(data[[predicting_var]], p = training_split, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # Convert categorical variables to factors
  for (var in categorical_vars) {
    train_data[[var]] <- as.factor(train_data[[var]])
    test_data[[var]] <- as.factor(test_data[[var]])
  }
  
  # Ensure levels in factors are consistent
  train_data[[predicting_var]] <- factor(train_data[[predicting_var]], levels = levels(data[[predicting_var]]))
  test_data[[predicting_var]] <- factor(test_data[[predicting_var]], levels = levels(data[[predicting_var]]))
  
  # Scale numerical variables
  train_data[numerical_vars] <- scale(train_data[numerical_vars])
  test_data[numerical_vars] <- scale(test_data[numerical_vars])
  
  train_x <- train_data[, numerical_vars]
  train_y <- train_data[[predicting_var]]
  test_x <- test_data[, numerical_vars]
  test_y <- test_data[[predicting_var]]
  
  # Train the KNN model with k = 5 and use.all = FALSE
  knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = 5, prob = TRUE, use.all = FALSE)
  cm_knn <- confusionMatrix(knn_pred, test_y)
  
  result <- list(
    accuracy = round(cm_knn$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_knn,
    model = knn_pred
  )
  
  return(result)
}

# Function to create and train Support Vector Machine model
create_svm <- function(data, numerical_vars, categorical_vars, predicting_var, training_split) {
  set.seed(123)
  train_index <- createDataPartition(data[[predicting_var]], p = training_split, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # Convert categorical variables to factors
  for (var in categorical_vars) {
    train_data[[var]] <- as.factor(train_data[[var]])
    test_data[[var]] <- as.factor(test_data[[var]])
  }
  
  # Ensure levels in factors are consistent
  train_data[[predicting_var]] <- factor(train_data[[predicting_var]], levels = levels(data[[predicting_var]]))
  test_data[[predicting_var]] <- factor(test_data[[predicting_var]], levels = levels(data[[predicting_var]]))
  
  # Scale numerical variables
  train_data[numerical_vars] <- scale(train_data[numerical_vars])
  test_data[numerical_vars] <- scale(test_data[numerical_vars])
  
  # Train the SVM model
  model_svm <- svm(as.formula(paste(predicting_var, "~ .")), data = train_data, probability = TRUE)
  predictions <- predict(model_svm, test_data, probability = TRUE)
  predictions <- factor(predictions, levels = levels(test_data[[predicting_var]]))
  cm_svm <- confusionMatrix(predictions, test_data[[predicting_var]])
  
  result <- list(
    accuracy = round(cm_svm$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_svm,
    model = model_svm
  )
  
  return(result)
}

# Function to create and train Gradient Boosting model
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
  result <- list(
    accuracy = round(cm_gbm$overall['Accuracy'] * 100, 2),
    confusion_matrix = cm_gbm
  )
  
  return(result)
}

# Function to generate variable importance plot
plot_var_importance <- function(var_imp) {
  var_imp_plot <- ggplot(var_imp, aes(x = reorder(Overall, Overall), y = Overall)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_minimal() +
    labs(title = "RF Variable Importance", x = "Variables", y = "Importance")
  
  return(var_imp_plot)
}

# Function to generate correlation matrix plot
plot_correlation_matrix <- function(data) {
  numeric_data <- data[, sapply(data, is.numeric)]
  correlation_matrix <- cor(numeric_data, use = "complete.obs")
  correlation_df <- melt(correlation_matrix)
  correlation_plot_gg <- ggplot(correlation_df, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
    theme_minimal() +
    labs(title = "Correlation Matrix", x = "", y = "") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(correlation_plot_gg)
}

# Function to plot confusion matrix heatmap
plot_confusion_matrix <- function(conf_matrix) {
  cm_df <- as.data.frame(conf_matrix$table)
  cm_plot <- ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "red") +
    theme_minimal() +
    labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Actual")
  return(cm_plot)
}

# Function to plot K-Value Selection for KNN
plot_knn_k_value_selection <- function(data, numerical_vars, predicting_var) {
  set.seed(123)
  
  # Remove rows with missing values
  data <- na.omit(data)
  
  # Split data
  train_index <- createDataPartition(data[[predicting_var]], p = 0.8, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  train_x <- train_data[, numerical_vars]
  train_y <- train_data[[predicting_var]]
  test_x <- test_data[, numerical_vars]
  test_y <- test_data[[predicting_var]]
  
  # Try different k values, ensuring k is odd to minimize ties
  k_values <- seq(1, 19, by = 2)
  accuracy <- sapply(k_values, function(k) {
    knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = k, prob = TRUE)
    sum(knn_pred == test_y) / length(test_y)
  })
  
  # Plot accuracy vs. k
  plot(k_values, accuracy, type = "b", col = "blue", pch = 19, xlab = "K Value", ylab = "Accuracy", main = "K-Value Selection")
  grid()
}

# Function to plot Support Vectors for SVM
plot_support_vectors <- function(model, data, numerical_vars, predicting_var) {
  support_vectors <- data[model$index, ]
  
  # Plot only for 2D case (if there are more numerical variables, select two for plotting)
  if (length(numerical_vars) > 2) {
    x_var <- numerical_vars[1]
    y_var <- numerical_vars[2]
  } else {
    x_var <- numerical_vars[1]
    y_var <- numerical_vars[2]
  }
  
  sv_plot <- ggplot(data, aes_string(x = x_var, y = y_var, color = predicting_var)) +
    geom_point() +
    geom_point(data = support_vectors, aes_string(x = x_var, y = y_var), shape = 1, size = 3, color = "red") +
    theme_minimal() +
    ggtitle("Support Vectors")
  
  return(sv_plot)
}

# Function to plot Learning Curve for SVM
plot_learning_curve <- function(model, data, numerical_vars, predicting_var) {
  set.seed(123)
  
  # Define training sizes
  training_sizes <- seq(0.1, 0.9, by = 0.1)
  
  train_accuracies <- c()
  test_accuracies <- c()
  
  for (size in training_sizes) {
    train_index <- createDataPartition(data[[predicting_var]], p = size, list = FALSE)
    train_data <- data[train_index, ]
    test_data <- data[-train_index, ]
    
    # Train the SVM model
    model_svm <- svm(as.formula(paste(predicting_var, "~ .")), data = train_data, probability = TRUE)
    
    # Training accuracy
    train_predictions <- predict(model_svm, train_data)
    train_cm <- confusionMatrix(train_predictions, train_data[[predicting_var]])
    train_accuracies <- c(train_accuracies, train_cm$overall['Accuracy'])
    
    # Test accuracy
    test_predictions <- predict(model_svm, test_data)
    test_cm <- confusionMatrix(test_predictions, test_data[[predicting_var]])
    test_accuracies <- c(test_accuracies, test_cm$overall['Accuracy'])
  }
  
  # Plot learning curve
  learning_curve_plot <- data.frame(
    Training_Size = training_sizes * 100,
    Train_Accuracy = train_accuracies,
    Test_Accuracy = test_accuracies
  )
  
  p <- ggplot(learning_curve_plot, aes(x = Training_Size)) +
    geom_line(aes(y = Train_Accuracy, color = "Train Accuracy")) +
    geom_line(aes(y = Test_Accuracy, color = "Test Accuracy")) +
    labs(title = "Learning Curve", x = "Training Size (%)", y = "Accuracy") +
    theme_minimal()
  
  return(p)
}




