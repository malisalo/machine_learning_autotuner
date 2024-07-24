library(shiny)
library(bslib)
library(shinycssloaders)

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
        card_header("Select Features"),
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
    id = "main_navset",
    nav_panel("Dataset", tableOutput("data_preview")),
    nav_panel("Model Results",uiOutput("model_results_ui")),
    nav_panel("Model Code", uiOutput("model_code_ui"))
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
      }"
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
