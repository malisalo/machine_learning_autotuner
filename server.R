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
  
  # Update UI inputs based on dataset column names
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
  
  # Helper function to process models
  process_model <- function(model_name, data, predicting_var, train_split, model_amount) {
    model_func <- get(model_name)
    result <- model_func(data, predicting_var, train_split, model_amount)
    code_gen_func <- get(paste0("create_code_", sub("create_", "", model_name)))
    list(
      result = result,
      params = result$best_params,
      code = code_gen_func(predicting_var, train_split, result$best_params)
    )
  }
  
  observeEvent(input$run_model, {
    withProgress(message = 'Training models...', value = 0, {
      df <- dataset()
      predicting_var <- input$Predictive
      imputation_technique <- input$Imputation
      training_split <- input$Train_Set / 100
      selected_models <- input$model_selection
      models_amount <- list(
        "create_rf" = input$models_amount_rf,
        "create_knn" = input$models_amount_knn,
        "create_gbm" = input$models_amount_gbm,
        "create_svm" = input$models_amount_svm
      )
      
      # Filter dataset to keep only selected features
      selected_features <- input$Features
      if (length(selected_features) > 0) {
        df <- df %>%
          select(all_of(c(selected_features, predicting_var)))
      }
      
      imputed_data <- perform_imputation(imputation_technique, df)
      
      results <- lapply(selected_models, function(model) {
        incProgress(1 / length(selected_models), detail = paste(model_names[[model]], "is currently being trained"))
        process_model(model, imputed_data, predicting_var, training_split, models_amount[[model]])
      })
      
      names(results) <- selected_models
      model_results <- lapply(results, `[[`, "result")
      model_params <- lapply(results, `[[`, "params")
      model_code <- lapply(results, `[[`, "code")
      
      # Render model results
      output$model_results_ui <- renderUI({
        lapply(names(model_results), function(model) {
          result <- model_results[[model]]
          tagList(
            h3(model_names[model]),
            h4(paste("Accuracy:", result$accuracy, "%")),
            DTOutput(paste0("confusion_matrix_", model)),
            hr()
          )
        }) %>% do.call(tagList, .)
      })
      
      # Render confusion matrices
      lapply(names(model_results), function(model) {
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
              rownames = FALSE
            )
          })
        })
      })
      
      # Render model code
      output$model_code_ui <- renderUI({
        lapply(names(model_code), function(model) {
          tagList(
            h3(model_names[model]),
            verbatimTextOutput(paste0("code_", model))
          )
        }) %>% do.call(tagList, .)
      })
      
      lapply(names(model_code), function(model) {
        local({
          model_name <- model
          code <- model_code[[model_name]]
          output[[paste0("code_", model_name)]] <- renderPrint({
            cat(code)
          })
        })
      })
      
      # Dynamically manage tabs
      tab_titles <- setNames(c("RF Results", "KNN Results", "GBM Results", "SVM Results"),
                             c("create_rf", "create_knn", "create_gbm", "create_svm"))
      
      tabs_to_add <- tab_titles[selected_models]
      lapply(tabs_to_add, function(title) {
        appendTab("main_navset", nav_panel(title))
      })
    })
  })
}
