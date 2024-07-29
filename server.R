library(shiny)
library(bslib)
library(dplyr)
library(DT)
library(shinycssloaders)
library(ggplot2)
library(gridExtra)
library(plotly)
library(caret)

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
    
    # Identify categorical columns
    categorical_cols <- cols[sapply(dataset(), function(col) {
      is.factor(col) || is.character(col)
    })]
    
    updateSelectInput(session, "Predictive", choices = categorical_cols)
  })
  
  observe({
    req(dataset(), input$Predictive)
    cols <- names(dataset())
    updateCheckboxGroupInput(session, "Features", choices = setdiff(cols, input$Predictive))
  })
  
  observeEvent(input$select_all_features, {
    req(dataset())
    cols <- names(dataset())
    if (input$select_all_features) {
      updateCheckboxGroupInput(session, "Features", selected = setdiff(cols, input$Predictive))
    } else {
      updateCheckboxGroupInput(session, "Features", selected = character(0))
    }
  })
  
  # Output: Display a preview of the dataframe
  output$data_preview <- renderDT({
    req(dataset())
    datatable(
      dataset(),
      options = list(pageLength = 10, scrollX = TRUE),
      rownames = FALSE
    )
  })
  
  # Helper function to process models
  process_model <- function(model_name,
                            train,
                            test,
                            predicting_var,
                            model_amount) {
    model_func <- get(model_name)
    result <- model_func(train, test, predicting_var, model_amount)
    code_gen_func <- get(paste0("create_code_", sub("create_", "", model_name)))
    list(
      result = result,
      params = result$best_params,
      code = code_gen_func(predicting_var, 0.8, result$best_params),
      # Pass training_split here
      has_variable_importance = "variable_importance" %in% names(result)
    )
  }
  
  observeEvent(input$run_model, {
    withProgress(message = 'Training models...', value = 0, {
      # Remove existing model tabs
      existing_tabs <- names(model_names)
      lapply(existing_tabs, function(model) {
        tab_id <- paste0("tab_", model)
        removeTab(inputId = "main_navset", target = tab_id)
      })
      
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
          select(all_of(c(
            selected_features, predicting_var
          )))
      }
      
      # Handle imputation technique
      if (imputation_technique == "delete") {
        # Remove NAs first
        df <- df %>% na.omit()
        set.seed(123)
        trainIndex <- createDataPartition(df[[predicting_var]],
                                          p = training_split,
                                          list = FALSE,
                                          times = 1)
        train <- df[trainIndex, ]
        test <- df[-trainIndex, ]
      } else {
        set.seed(123)
        df <- df %>% mutate_if(is.character, as.factor)
        trainIndex <- createDataPartition(df[[predicting_var]],
                                          p = training_split,
                                          list = FALSE,
                                          times = 1)
        train <- df[trainIndex, ]
        test <- df[-trainIndex, ]
        # Perform separate imputations on the training and test sets
        train <- perform_imputation(imputation_technique, train)
        test <- perform_imputation(imputation_technique, test)
      }
      
      results <- lapply(selected_models, function(model) {
        incProgress(
          1 / length(selected_models),
          detail = paste(model_names[[model]], "is currently being trained")
        )
        process_model(model, train, test, predicting_var, models_amount[[model]])
      })
      
      names(results) <- selected_models
      model_results <- lapply(results, `[[`, "result")
      model_params <- lapply(results, `[[`, "params")
      model_code <- lapply(results, `[[`, "code")
      has_variable_importance <- lapply(results, `[[`, "has_variable_importance")
      
      # Dynamically add tabs for each model
      lapply(names(model_results), function(model) {
        tab_id <- paste0("tab_", model)
        insertTab(
          inputId = "main_navset",
          tabPanel(
            title = model_names[model],
            value = tab_id,
            div(
              class = "model-results",
              h3(model_names[model]),
              h4(paste(
                "Accuracy:", model_results[[model]]$accuracy, "%"
              )),
              plotlyOutput(paste0("confusion_matrix_", model), height = "400px"),
              hr(),
              h4("Variable Importance Chart"),
              if (has_variable_importance[[model]]) {
                plotOutput(paste0("var_importance_", model))
              } else {
                tags$p("Variable importance not available for this model.")
              },
              hr(),
              h4("Generated Model Code"),
              verbatimTextOutput(paste0("code_", model))
            )
          ),
          target = "Model Graph",
          # Insert the new tab before "Model Graph" tab
          position = "before"
        )
        
        # Render confusion matrices as heatmaps using plotly
        local({
          model_name <- model
          result <- model_results[[model_name]]
          output[[paste0("confusion_matrix_", model_name)]] <- renderPlotly({
            cm <- as.data.frame(result$confusion_matrix)
            colnames(cm) <- c("Prediction", "Reference", "Frequency")
            p <- ggplot(cm, aes(x = Reference, y = Prediction)) +
              geom_tile(aes(fill = Frequency), color = "white") +
              scale_fill_gradient(low = "white",
                                  high = "deepskyblue3") +
              geom_text(aes(label = Frequency), vjust = 1) +
              theme_minimal() +
              ggtitle(paste(model_names[model_name], "Confusion Matrix")) +
              xlab("Reference") +
              ylab("Prediction")
            ggplotly(p) %>% layout(autosize = TRUE)
          })
        })
        
        # Render variable importance if available
        if (has_variable_importance[[model]]) {
          local({
            model_name <- model
            result <- model_results[[model_name]]
            output[[paste0("var_importance_", model_name)]] <- renderPlot({
              var_importance <- result$variable_importance
              var_importance_df <- data.frame(Feature = rownames(var_importance),
                                              Importance = var_importance[, 1])
              var_importance_df <- var_importance_df[order(var_importance_df$Importance, decreasing = TRUE), ]
              ggplot(var_importance_df,
                     aes(
                       x = reorder(Feature, Importance),
                       y = Importance
                     )) +
                geom_bar(stat = "identity", fill = "steelblue") +
                coord_flip() +
                theme_minimal() +
                labs(
                  title = paste(model_names[model_name], "Variable Importance"),
                  x = "Features",
                  y = "Importance"
                )
            })
          })
        }
        
        # Render model code
        local({
          model_name <- model
          code <- model_code[[model_name]]
          output[[paste0("code_", model_name)]] <- renderPrint({
            cat(code)
          })
        })
      })
      
      # Render the bar graph of best parameter accuracies
      output$model_accuracy_plot <- renderPlot({
        accuracies <- sapply(selected_models, function(model) {
          result <- model_results[[model]]
          result$accuracy
        })
        
        model_labels <- model_names[selected_models]
        
        # Create data frame
        accuracy_df <- data.frame(Model = factor(model_labels, levels = model_labels),
                                  Accuracy = accuracies)
        
        # Plot accuracies
        ggplot(accuracy_df, aes(x = Model, y = Accuracy)) +
          geom_bar(stat = "identity", fill = "lightblue") +
          ggtitle("Model Accuracies") +
          xlab("Model") +
          ylab("Accuracy") +
          ylim(0, 100) +
          theme_classic() +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))
      })
    })
  })
  
  # Other server logic for model tabs and outputs
  observe({
    req(dataset())
    updateTabsetPanel(session, "main_navset", selected = "Dataset")
  })
}
