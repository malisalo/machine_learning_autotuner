library(glue)
library(dplyr)

# Function to get imputation code based on the technique
get_imputation_code <- function(impute_technique) {
  switch(impute_technique,
         "median_impute" = "
# Perform Median imputation
numerical_columns <- names(train_data)[sapply(train_data, is.numeric)]
train_data <- train_data %>%
  mutate_at(vars(one_of(numerical_columns)), ~ ifelse(is.na(.), median(., na.rm = TRUE), .))
test_data <- test_data %>%
  mutate_at(vars(one_of(numerical_columns)), ~ ifelse(is.na(.), median(., na.rm = TRUE), .))
",
         "mice_impute" = "
# Perform MICE imputation
library(mice)
train_data_imputed <- mice(train_data, m = 5, maxit = 50, method = 'pmm', seed = 123)
train_data <- complete(train_data_imputed)
test_data_imputed <- mice(test_data, m = 5, maxit = 50, method = 'pmm', seed = 123)
test_data <- complete(test_data_imputed)
",
         "missForest_impute" = "
# Perform MissForest imputation
library(missForest)
train_data_imputed <- missForest(train_data)
train_data <- train_data_imputed$ximp
test_data_imputed <- missForest(test_data)
test_data <- test_data_imputed$ximp
",
         "delete_na" = "
# Delete rows with NA values
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)
")
}

# Function to select specific columns
select_columns_code <- function(columns_to_keep) {
  columns <- paste(columns_to_keep, collapse = ", ")
  glue::glue(
    "
# Keep only selected columns
data <- data %>% select({columns})
"
  )
}

# Function to generate SVM code
create_code_svm <- function(predicting_var, training_split, best_params, impute_technique, features) {
  if (!is.list(best_params) || !all(c("C", "sigma") %in% names(best_params))) {
    stop("best_params must be a list with elements 'C' and 'sigma'.")
  }
  
  imputation_code <- get_imputation_code(impute_technique)
  feature_selection_code <- select_columns_code(c(features, predicting_var))
  
  svm_code <- glue::glue(
    "
# Load necessary libraries
library(e1071)
library(caret)
library(dplyr)

# Load Dataset [CHANGE ME]
data <- read.csv('your_datapath')

{feature_selection_code}

data <- data %>% mutate_if(is.character, as.factor)

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
index <- createDataPartition(data${predicting_var}, p = {training_split}, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]
{imputation_code}

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

# Function to generate Random Forest code
create_code_rf <- function(predicting_var, training_split, best_params, impute_technique, features) {
  imputation_code <- get_imputation_code(impute_technique)
  feature_selection_code <- select_columns_code(c(features, predicting_var))
  
  rf_code <- glue::glue(
    "
# Load necessary libraries
library(randomForest)
library(caret)
library(dplyr)

# Load Dataset [CHANGE ME]
data <- read.csv('your_datapath')

{feature_selection_code}

data <- data %>% mutate_if(is.character, as.factor)

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
index <- createDataPartition(data${predicting_var}, p = {training_split}, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]
{imputation_code}

# Train the Random Forest model
rf_model <- randomForest({predicting_var} ~ ., data = train_data, mtry = {best_params$mtry}, ntree = {best_params$num.trees}, nodesize = {best_params$min.node.size})

# Predict on the test set
predictions <- predict(rf_model, newdata = test_data)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data${predicting_var})
print(conf_matrix)
"
  )
  
  return(rf_code)
}

# Function to generate KNN code
create_code_knn <- function(predicting_var, training_split, best_params, impute_technique, features) {
  imputation_code <- get_imputation_code(impute_technique)
  feature_selection_code <- select_columns_code(c(features, predicting_var))
  
  knn_code <- glue::glue(
    "
# Load necessary libraries
library(caret)
library(dplyr)

# Load Dataset [CHANGE ME]
data <- read.csv('your_datapath')

{feature_selection_code}

data <- data %>% mutate_if(is.character, as.factor)

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
index <- createDataPartition(data${predicting_var}, p = {training_split}, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]
{imputation_code}

# Train the KNN model
knn_model <- train({predicting_var} ~ ., data = train_data, method = 'knn', k = {best_params$k})

# Predict on the test set
predictions <- predict(knn_model, newdata = test_data)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data${predicting_var})
print(conf_matrix)
"
  )
  
  return(knn_code)
}

# Function to generate GBM code
create_code_gbm <- function(predicting_var, training_split, best_params, impute_technique, features) {
  imputation_code <- get_imputation_code(impute_technique)
  feature_selection_code <- select_columns_code(c(features, predicting_var))
  
  gbm_code <- glue::glue(
    "
# Load necessary libraries
library(gbm)
library(caret)
library(dplyr)

# Load Dataset [CHANGE ME]
data <- read.csv('your_datapath')

{feature_selection_code}

data <- data %>% mutate_if(is.character, as.factor)

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
index <- createDataPartition(data${predicting_var}, p = {training_split}, list = FALSE)
train_data <- data[index, ]
test_data <- data[-index, ]
{imputation_code}

# Train the GBM model
gbm_model <- train({predicting_var} ~ ., data = train_data, method = 'gbm', n.trees = {best_params$n.trees}, interaction.depth = {best_params$interaction.depth}, shrinkage = {best_params$shrinkage}, n.minobsinnode = {best_params$n.minobsinnode}, verbose = FALSE)

# Predict on the test set
predictions <- predict(gbm_model, newdata = test_data)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data${predicting_var})
print(conf_matrix)
"
  )
  
  return(gbm_code)
}
