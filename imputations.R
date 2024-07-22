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


# Data Imputation Methods
perform_imputation <- function(technique, data) {
  # Convert all character columns to factors
  data <- data %>% mutate_if(is.character, as.factor)
  func <- match.fun(technique)
  imputed_data <- func(data)
  return(imputed_data)
}

# Perform MICE imputation
mice_impute <- function(data) {
  imputed_data <- mice(
    data,
    m = 5,
    maxit = 50,
    method = 'pmm',
    seed = 123
  )
  imputed_data <- complete(imputed_data)
  return(imputed_data)
}

# Perform MissForest imputation
missForest_impute <- function(data) {
  imputed_data <- missForest(data)
  data <- imputed_data$ximp
  return(data)
}

# Convert all character columns to factors
delete_na <- function(data) {
  data <- na.omit(data)
  return(data)
}