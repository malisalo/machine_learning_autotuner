library(dplyr)
library(missForest)
library(mice)
library(Hmisc)


# Data Imputation Methods
perform_imputation <- function(technique, data) {
  func <- match.fun(technique)
  imputed_data <- func(data)
  return(imputed_data)
}

# Perform Median imputation
median_impute <- function(data) {
  numerical_columns <- names(data)[sapply(data, is.numeric)]
  data <- data %>%
    mutate_at(vars(one_of(numerical_columns)), ~ifelse(is.na(.), median(., na.rm = TRUE), .))
  return(data)
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
