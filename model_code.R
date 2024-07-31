library(glue)
library(dplyr)

create_code_svm <- function(predicting_var, training_split, best_params) {
  # Ensure best_params is a list with named elements
  if (!is.list(best_params) || !all(c("C", "sigma") %in% names(best_params))) {
    stop("best_params must be a list with elements 'C' and 'sigma'.")
  }
  
  # Create the SVM code template with the best parameters
  svm_code <- glue::glue(
    "
# Load necessary libraries
library(e1071)
library(caret)

# Load Dataset [CHANGE ME]
data <- read.csv('your_datapath')

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
index <- createDataPartition(data${predicting_var}, p = {training_split}, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]

# Train the SVM model
svm_model <- svm({predicting_var} ~ ., data = train_data, kernel = 'radial', cost = {best_params$C}, gamma = {best_params$sigma})

# Predict on the test set
predictions <- predict(svm_model, newdata = test_data)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data${predicting_var})
print(conf_matrix)
"
  )
  
  return(svm_code)
}


create_code_rf <- function(predicting_var,
                           training_split,
                           best_params) {
  rf_code <- glue::glue(
    "
# Load necessary libraries
library(randomForest)
library(caret)

# Load Dataset [CHANGE ME]
data <- read.csv('your_datapath')

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
index <- createDataPartition(data${predicting_var}, p = {training_split}, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]

# Train the Random Forest model
rf_model <- randomForest({predicting_var} ~ ., data = train_data, mtry = {best_params$mtry})

# Predict on the test set
predictions <- predict(rf_model, newdata = test_data)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data${predicting_var})
print(conf_matrix)
"
  )
  return(rf_code)
}

create_code_knn <- function(predicting_var,
                            training_split,
                            best_params) {
  knn_code <- glue::glue(
    "
# Load necessary libraries
library(caret)

# Load Dataset [CHANGE ME]
data <- read.csv('your_datapath')

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
index <- createDataPartition(data${predicting_var}, p = {training_split}, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]

# Train the KNN model
knn_model <- train({predicting_var} ~ ., data = train_data, method = 'knn', tuneGrid = expand.grid(k = {best_params$k}))

# Predict on the test set
predictions <- predict(knn_model, newdata = test_data)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data${predicting_var})
print(conf_matrix)
"
  )
  return(knn_code)
}

create_code_gbm <- function(predicting_var,
                            training_split,
                            best_params) {
  gbm_code <- glue::glue(
    "
# Load necessary libraries
library(gbm)
library(caret)

# Load Dataset [CHANGE ME]
data <- read.csv('your_datapath')

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
index <- createDataPartition(data${predicting_var}, p = {training_split}, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]

# Train the GBM model
gbm_model <- train({predicting_var} ~ ., data = train_data, method = 'gbm', tuneGrid = expand.grid(n.trees = {best_params$n.trees}, interaction.depth = {best_params$interaction.depth}, shrinkage = {best_params$shrinkage}, n.minobsinnode = {best_params$n.minobsinnode}), verbose = FALSE)

# Predict on the test set
predictions <- predict(gbm_model, newdata = test_data)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data${predicting_var})
print(conf_matrix)
"
  )
  return(gbm_code)
}
