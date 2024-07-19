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

# Define server logic ----
server <- function(input, output) {
  bs_themer()
  
  # Directory to save uploaded files
  save_dir <- "uploads"
  dir.create(save_dir, showWarnings = FALSE)
  
  # Reactive expression to read the uploaded file
  dataset <- reactive({
    req(input$file)  # Ensure a file is uploaded
    
    # File path for the uploaded file
    file <- input$file$datapath
    
    # Save the file to the server directory
    save_path <- file.path(save_dir, input$file$name)
    file.copy(file, save_path, overwrite = TRUE)
    
    # Read the file into a dataframe
    df <- read.csv(save_path, stringsAsFactors = FALSE)
    return(df)
  })
  
  # Data Imputation Methods
  perform_imputation <- function(technique, data) {
    func <- match.fun(technique)
    imputed_data <- func(data)
    return(imputed_data)
  }
  
  # Perform MICE imputation
  mice_impute <- function(data) {
    imputed_data <- mice(data, m = 5, maxit = 50, method = 'pmm', seed = 123)
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
  
  # Gradient Boosting
  create_gbm <- function(data, predicting_var, training_split) {
    # Split the data into training and test sets
    set.seed(123)
    trainIndex <- createDataPartition(data[[predicting_var]], p = training_split, list = FALSE, times = 1)
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
    gbm_model <- train(as.formula(paste(predicting_var, "~ .")), 
                       data = train, 
                       method = "gbm",
                       metric = "Accuracy",
                       tuneGrid = tune_grid,
                       trControl = train_control,
                       verbose = FALSE)
    
    # Print the best parameters
    print(gbm_model$bestTune)
    
    # Use the best model to make predictions on the test set
    predictions_gbm <- predict(gbm_model, newdata = test)
    
    # Calculate confusion matrix
    cm_gbm <- confusionMatrix(predictions_gbm, test[[predicting_var]])
    
    # Print confusion matrix and overall accuracy
    result <- paste("Gradient Boosting\nAccuracy:", round(cm_gbm$overall['Accuracy'] * 100, 2), "%")
    return(result)
  }
  
  # Random Forest
  create_rf <- function(data, predicting_var, training_split) {
    set.seed(123)
    train_index <- createDataPartition(data[[predicting_var]], p = training_split, list = FALSE, times = 1)
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
    
    model_rf <- train(as.formula(paste(predicting_var, "~ .")),
                      data = train, 
                      method = 'rf',
                      metric = "Accuracy",
                      tuneGrid = tune_grid, 
                      trControl = train_control,
                      importance = TRUE)
    
    predictions <- predict(model_rf, newdata = test)
    predictions <- factor(predictions, levels = levels(test[[predicting_var]]))
    cm_rf <- confusionMatrix(predictions, test[[predicting_var]])
    
    # Print confusion matrix and overall accuracy
    result <- paste("Random Forest\nAccuracy:", round(cm_rf$overall['Accuracy'] * 100, 2), "%")
    return(result)
  }
  
  # KNN
  create_knn <- function(data, predicting_var, training_split) {
    # Split the data into training and test sets
    set.seed(123)
    trainIndex <- createDataPartition(data[[predicting_var]], p = training_split, list = FALSE, times = 1)
    train <- data[trainIndex, ]
    test <- data[-trainIndex, ]
    
    # Set up the tuning grid for k
    tune_grid <- expand.grid(k = c(1:20))
    
    # Train control with cross-validation
    train_control <- trainControl(method = "cv", number = 5, search = "grid")
    
    # Train the KNN model with hyperparameter tuning
    knn_tuned <- train(as.formula(paste(predicting_var, "~ .")), 
                       data = train, 
                       method = "knn",
                       metric = "Accuracy",
                       tuneGrid = tune_grid,
                       trControl = train_control)
    
    # Print the best parameters
    print(knn_tuned$bestTune)
    
    # Use the best model to make predictions on the test set
    predictions_knn <- predict(knn_tuned, newdata = test)
    
    # Calculate confusion matrix
    cm_knn <- confusionMatrix(predictions_knn, test[[predicting_var]])
    
    # Print confusion matrix and overall accuracy
    result <- paste("KNN\nAccuracy:", round(cm_knn$overall['Accuracy'] * 100, 2), "%")
    return(result)
  }
  
  # SVM Classifier
  create_svm <- function(data, predicting_var, training_split) {
    # Convert the target variable to a factor for classification
    data[[predicting_var]] <- as.factor(data[[predicting_var]])
    
    # Split the data into training and test sets
    set.seed(123)
    trainIndex <- createDataPartition(data[[predicting_var]], p = .8, list = FALSE, times = 1)
    train <- data[trainIndex, ]
    test <- data[-trainIndex, ]
    
    # Train control with cross-validation
    train_control <- trainControl(method = "cv", number = 5)
    
    # Train the SVM model
    svm_model <- train(
      as.formula(paste(predicting_var, "~ .")),
      data = train,
      method = "svmRadial",
      trControl = train_control)
    
    # Make predictions
    predictions <- predict(svm_model, newdata = test)
    
    # Calculate confusion matrix
    cm <- confusionMatrix(predictions, test[[predicting_var]])
    
    # Print confusion matrix and accuracy
    result <- paste("SVM\nAccuracy:", round(cm$overall['Accuracy'] * 100, 2), "%")
    return(result)
  }
  
  # Output: Display a preview of the dataframe
  output$data_preview <- renderTable({
    req(dataset())  # Ensure that the dataset is available
    head(dataset(), 10)  # Show the first 10 rows of the dataframe
  })
  
  output$selected_Categorical <- renderText({
    paste("You have selected", paste(input$Categorical, collapse = ", "))
  })
  
  output$selected_Numerical <- renderText({
    paste("You have selected", paste(input$Numerical, collapse = ", "))
  })
  
  output$selected_Predictive <- renderText({
    paste("You have selected", input$Predictive)
  })
  
  output$selected_Trainset <- renderText({
    paste("You have selected", input$Train_Set, "%")
  })
  
  output$save_status <- renderText({
    req(input$file)
    paste("File saved as", input$file$name)
  })
  
  observeEvent(input$run_model, {
    # Get the dataset
    df <- dataset()
    
    # Get user inputs
    predicting_var <- input$Predictive
    imputation_technique <- input$Imputation
    training_split <- input$Train_Set / 100
    
    # Perform imputation
    imputed_data <- perform_imputation(imputation_technique, df)
    
    # Convert all character columns to factors
    imputed_data <- imputed_data %>% mutate_if(is.character, as.factor)
    
    # Run models and capture outputs
    knn_result <- create_knn(imputed_data, predicting_var, training_split)
    svm_result <- create_svm(imputed_data, predicting_var, training_split)
    gbm_result <- create_gbm(imputed_data, predicting_var, training_split)
    # rf_result <- create_rf(imputed_data, predicting_var, training_split)
    
    # Display results
    output$model_output <- renderText({
      paste(knn_result, svm_result, gbm_result, sep = "\n\n")
    })
  })
}
