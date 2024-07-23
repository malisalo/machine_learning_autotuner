library(shiny)
library(bslib)
library(dplyr)
library(DT)
library(shinycssloaders)

# Source the models and code generation files
source("models.R")
source("imputations.R")
source("model_code.R")

# Define user-friendly names for models
model_names <- c(
  "create_rf" = "Random Forest",
  "create_knn" = "K Nearest Neighbors",
  "create_gbm" = "Gradient Boosting",
  "create_svm" = "Support Vector Machine"
)

# Define server logic
server <- function(input, output, session) {
  # Directory to save uploaded files
  save_dir <- "uploads"
  dir.create(save_dir, showWarnings = FALSE)
  
  # Reactive expression to read the uploaded file
  dataset <- reactive({
    req(input$file)
    file <- input$file$datapath
    save_path <- file.path(save_dir, input$file$name)
    file.copy(file, save_path, overwrite = TRUE)
    read.csv(save_path, stringsAsFactors = FALSE)
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
    req(dataset())
    head(dataset(), 10)
  })
  
  observeEvent(input$run_model, {
    # Display the spinner while processing
    withProgress(message = 'Running models...', value = 0, {
      
      df <- dataset()
      predicting_var <- input$Predictive
      imputation_technique <- input$Imputation
      training_split <- input$Train_Set / 100
      selected_models <- input$model_selection
      
      imputed_data <- perform_imputation(imputation_technique, df)
      
      model_results <- list()
      model_params <- list()
      model_code <- list()
      
      for (model in selected_models) {
        incProgress(1 / length(selected_models), detail = paste("Processing", model))
        
        model_func <- get(model)
        result <- model_func(imputed_data, predicting_var, training_split)
        model_results[[model]] <- result
        model_params[[model]] <- result$best_params
        
        # Generate model setup code based on the best parameters
        if (model == "create_svm") {
          code <- create_code_svm(predicting_var, training_split, result$best_params)
        } else if (model == "create_rf") {
          code <- create_code_rf(predicting_var, training_split, result$best_params)
        } else if (model == "create_knn") {
          code <- create_code_knn(predicting_var, training_split, result$best_params)
        } else if (model == "create_gbm") {
          code <- create_code_gbm(predicting_var, training_split, result$best_params)
        }
        model_code[[model]] <- code
      }
      
      output$model_results_ui <- renderUI({
        output_list <- lapply(names(model_results), function(model) {
          result <- model_results[[model]]
          model_display_name <- model_names[model]  # Get the user-friendly name
          tagList(h3(model_display_name),
                  # Display the user-friendly name
                  h4(paste("Accuracy:", result$accuracy, "%")),
                  DTOutput(paste0("confusion_matrix_", model)),
                  hr())
        })
        do.call(tagList, output_list)
      })
      
      for (model in names(model_results)) {
        local({
          model_name <- model
          result <- model_results[[model_name]]
          output[[paste0("confusion_matrix_", model_name)]] <- renderDT({
            datatable(
              result$confusion_matrix,
              options = list(
                pageLength = 5,
                autoWidth = TRUE,
                searching = FALSE
              ),
              rownames=FALSE
            )
          })
        })
      }
      
      output$model_code_ui <- renderUI({
        code_list <- lapply(names(model_code), function(model) {
          code <- model_code[[model]]
          model_display_name <- model_names[model]
          tagList(h3(model_display_name), verbatimTextOutput(paste0("code_", model)))
        })
        do.call(tagList, code_list)
      })
      
      for (model in names(model_code)) {
        local({
          model_name <- model
          code <- model_code[[model_name]]
          output[[paste0("code_", model_name)]] <- renderPrint({
            code
          })
        })
      }
    })
  })
}
