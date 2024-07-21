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

# Source the models file
source("models.R")
source("imputations.R")

# Define server logic
server <- function(input, output, session) {
  # bs_themer()
  
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
    selected_models <- input$model_selection
    
    # Perform imputation
    imputed_data <- perform_imputation(imputation_technique, df)
    
    # Initialize a list to store model results
    model_results <- list()
    
    # Run selected models and capture outputs
    for (model in selected_models) {
      model_func <- get(model)
      result <- model_func(imputed_data, predicting_var, training_split)
      model_results[[model]] <- result
    }
    
    # Render the model results
    output$model_results_ui <- renderUI({
      output_list <- lapply(names(model_results), function(model) {
        tagList(
          h3(model),
          verbatimTextOutput(paste0("result_", model))
        )
      })
      do.call(tagList, output_list)
    })
    
    for (model in names(model_results)) {
      local({
        model_name <- model
        output[[paste0("result_", model_name)]] <- renderPrint({
          model_results[[model_name]]
        })
      })
    }
  })
}
