library(shiny)
library(bslib)
library(caret)
library(randomForest)
library(ggplot2)
library(corrplot)
library(reshape2)
library(tidyverse)
library(doParallel)

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

# Function to generate variable importance plot
plot_var_importance <- function(var_imp) {
  var_imp_plot <- ggplot(var_imp, aes(x = reorder(Overall, Overall), y = Overall)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Variable Importance", x = "Variables", y = "Importance")
  
  return(var_imp_plot)
}

# Function to generate correlation matrix plot
plot_correlation_matrix <- function(correlation_matrix) {
  correlation_df <- melt(correlation_matrix)
  correlation_plot_gg <- ggplot(correlation_df, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
    theme_minimal() +
    labs(title = "Correlation Matrix", x = "", y = "") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(correlation_plot_gg)
}
