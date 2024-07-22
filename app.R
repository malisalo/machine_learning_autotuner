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
library(ggplot2)
library(corrplot)
library(reshape2)

# Source the models file
source("models.R")
source("imputations.R")

# Define UI ----
ui <- page_sidebar(
  title = "Maize Data ML Dashboard",
  sidebar = sidebar(
    class = "sidebar-container",
    div(
      class = "sidebar-content",
      # Input: Select a file ----
      fileInput(
        "file",
        "Choose CSV File",
        multiple = FALSE,
        accept = c("text/csv", "text/comma-separated-values,text/plain", ".csv")
      ),
      selectInput(
        "Predictive",
        "Select Predictive Variable",
        choices = NULL,
        selected = 1
      ),
      card(
        card_header("Categorical Variable"),
        checkboxGroupInput(
          inputId = "Categorical",
          label = NULL,
          choices = NULL
        )
      ),
      card(
        card_header("Numerical Variable"),
        checkboxGroupInput(
          inputId = "Numerical",
          label = NULL,
          choices = NULL
        )
      ),
      selectInput(
        inputId = "Imputation",
        "Select Imputation Technique",
        choices = list(
          "MICE" = "mice_impute",
          "missForest" = "missForest_impute",
          "Delete Missing Values" = "delete_na"
        ),
        selected = "mice_impute"
      ),
      card(
        card_header("Select Models to Train"),
        checkboxGroupInput(
          inputId = "model_selection",
          label = NULL,
          choices = list(
            "Random Forest" = "create_rf",
            "K Nearest Neighbors" = "create_knn",
            "Gradient Boosting" = "create_gbm", 
            "Support Vector Model" = "create_svm"
          )
        )
      ),
      card(
        card_header("Train Set"),
        sliderInput(
          "Train_Set",
          "Percentage %",
          min = 0,
          max = 100,
          value = 80
        )
      ),
      actionButton("run_model", "Run Model")
    ),
    div(class = "resize-handle")
  ),
  navset_card_underline(
    nav_panel("Dataset", tableOutput("data_preview")),
    nav_panel("Imputations"),
    nav_panel("Model Results", uiOutput("model_results_ui")),
    nav_panel("Feature Importance", plotOutput("var_imp_plot"), plotOutput("corr_plot"))
  ),
  tags$head(tags$style(
    HTML(
      "
      .sidebar-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        width: 250px;
        min-width: 200px;
        overflow-y: hidden;
        position: relative;
      }
      .sidebar-content {
        flex: 1;
        overflow-y: auto;
        padding: 10px;
      }
      .resize-handle {
        width: 5px;
        cursor: ew-resize;
        background-color: #ddd;
        position: absolute;
        top: 0,
        right: 0,
        bottom: 0;
      }
    "
    )
  ), tags$script(
    HTML(
      "
      $(document).on('shiny:connected', function() {
        var startX, startWidth;
        $('.resize-handle').on('mousedown', function(e) {
          startX = e.clientX;
          startWidth = $('.sidebar-container').width();
          $(document).on('mousemove', doDrag);
          $(document).on('mouseup', stopDrag);
        });

        function doDrag(e) {
          var newWidth = startWidth + (e.clientX - startX);
          if (newWidth >= 200) {
            $('.sidebar-container').css('width', newWidth);
          }
        }

        function stopDrag() {
          $(document).off('mousemove', doDrag);
          $(document).off('mouseup', stopDrag);
        }
      });
    "
    )
  ))
)

# Define server logic
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
    df <- dataset()
    
    numerical_vars <- names(df)[sapply(df, is.numeric)]
    categorical_vars <- setdiff(names(df), numerical_vars)
    
    updateSelectInput(session, "Predictive", choices = names(df))
    updateCheckboxGroupInput(session, "Numerical", choices = numerical_vars)
    updateCheckboxGroupInput(session, "Categorical", choices = categorical_vars)
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
    numerical_vars <- input$Numerical
    categorical_vars <- input$Categorical
    
    # Perform imputation
    imputed_data <- perform_imputation(imputation_technique, df)
    
    # Initialize a list to store model results
    model_results <- list()
    
    # Run selected models and capture outputs
    for (model in selected_models) {
      model_func <- get(model)
      result <- model_func(imputed_data, numerical_vars, categorical_vars, predicting_var, training_split)
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
          model_results[[model_name]]$accuracy
        })
      })
    }
    
    # Generate and render variable importance plot and correlation plot
    if ("create_rf" %in% selected_models) {
      rf_result <- model_results[["create_rf"]]
      output$var_imp_plot <- renderPlot({
        plot_var_importance(rf_result$var_imp)
      })
      output$corr_plot <- renderPlot({
        plot_correlation_matrix(rf_result$correlation_matrix)
      })
    }
  })
}

# Run the app ----
shinyApp(ui = ui, server = server)
