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
server <- function(input, output, session) {
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
  
  # Observe the dataset and update UI inputs based on the dataset's column names
  observe({
    req(dataset())
    cols <- names(dataset())
    
    updateSelectInput(session, "Predictive", choices = cols)
    updateCheckboxGroupInput(session, "Features", choices = cols)
  })
  
  # Output: Display a preview of the dataframe
  output$data_preview <- renderTable({
    req(dataset())  # Ensure that the dataset is available
    head(dataset(), 10)  # Show the first 10 rows of the dataframe
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
