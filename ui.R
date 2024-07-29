library(shiny)
library(bslib)
library(shinycssloaders)

# Define UI ----
ui <- page_sidebar(
  title = "Machine Learning Autotunner",
  bg = "#1fa853",
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
        card_header(
          "Select Features",
          checkboxInput("select_all_features", "Select All", value = FALSE)
        ),
        checkboxGroupInput(
          inputId = "Features",
          label = NULL,
          choices = NULL
        )
      ),
      selectInput(
        inputId = "Imputation",
        "Select Imputation Technique",
        choices = list(
          "MICE" = "mice_impute",
          "Median" = "median_impute",
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
        card_header("Number of Models Trained"),
        conditionalPanel(
          condition = "input.model_selection.indexOf('create_rf') !== -1",
          sliderInput(
            "models_amount_rf",
            "# of Random Forests",
            min = 0,
            max = 200,
            value = 50
          )
        ),
        conditionalPanel(
          condition = "input.model_selection.indexOf('create_knn') !== -1",
          sliderInput(
            "models_amount_knn",
            "# of K Nearest Neighbors",
            min = 0,
            max = 200,
            value = 125
          )
        ),
        conditionalPanel(
          condition = "input.model_selection.indexOf('create_gbm') !== -1",
          sliderInput(
            "models_amount_gbm",
            "# of Gradient Boostings",
            min = 0,
            max = 200,
            value = 15
          )
        ),
        conditionalPanel(
          condition = "input.model_selection.indexOf('create_svm') !== -1",
          sliderInput(
            "models_amount_svm",
            "# of SVMs",
            min = 0,
            max = 200,
            value = 15
          )
        )
      ),
      card(
        card_header("Train Set"),
        sliderInput(
          "Train_Set",
          "Training Set %",
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
    id = "main_navset",
    nav_panel("Overview", fluidRow(column(
      12,
      img(
        src = "autotunner_logo.svg",
        height = "350px",
        alt = "Logo",
        style = "display: block; margin-left: auto; margin-right: auto;"
      ),
      h2("Welcome to Machine Learning Autotuner"),
      p(
        "This application helps you train and tune machine learning models with ease. Upload your dataset, select features, choose models, and let the application do the rest."
      ),
      p("Features include:"),
      tags$ul(
        tags$li("Data Imputation Techniques"),
        tags$li("Multiple Machine Learning Models"),
        tags$li("Hyperparameter Tuning"),
        tags$li("Model Evaluation Metrics")
      )
    ))),
    nav_panel("Dataset", DTOutput("data_preview")),
    nav_panel(
      "Model Graph",
      plotOutput("model_accuracy_plot", width = "100%")
    )
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
        top: 0;
        right: 0;
        bottom: 0;
      }
      .model-results {
        padding: 15px;
      }
      .model-results h3, .model-results h4 {
        margin-bottom: 20px;
      }
      .navbar.navbar-static-top {
        background-color: #05322e; /* Background color for the title section */
        color: #ffffff; /* Text color */
      }
      .bslib-page-title.navbar-brand {
        color: #ffffff; /* Font color for the h1 tag */
          font-size: 1rem; /* Adjust font size if needed */
          font-weight: bold; /* Make the font bold */
          margin: 0; /* Remove default margin if needed */
          padding: 10px 0; /* Add padding if needed */
      }
        .btn.btn-default.btn-file:hover,
        .btn.btn-default.action-button.shiny-bound-input:hover {
        background-color: #05322e; /* Background color on hover */
        color: #ffffff; /* Text color on hover */
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
    ),
  ))
)
