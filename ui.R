library(shiny)
library(bslib)

# Define UI ----
ui <- page_sidebar(
  title = "Maize Data ML Dashboard",
  sidebar = sidebar(
    style = "width: 233px;",
    card(
      card_header("Upload Dataset"),
      fileInput("file", label = NULL)
    ),
    card(
      card_header("Categorical Variable"),
      checkboxGroupInput(
        inputId = "Categorical",
        label = "Select all that apply",
        choices = list("Leaf_No" = "Leaf_No",
                       "Genotype_ID" = "Genotype_ID",
                       "Treatment_ID" = "Treatment_ID"),
        selected = "Leaf_No"
      )
    ),
    card(
      card_header("Numerical Variable"),
      checkboxGroupInput(
        inputId = "Numerical",
        label = "Select all that apply",
        choices = list("Blade_Width" = "Blade_Width",
                       "Blade_Length" = "Blade_Length",
                       "Sheath_Length" = "Length",
                       "Surface_Area" = "Surface_Area"),
        selected = "Blade_Width"
      )
    ),
    card(
      card_header("Predictive Variable"),
      selectInput(
        inputId = "Predictive",
        label = "Select Predictive Variables",
        choices = list("Leaf_No" = "Leaf_No",
                       "Genotype_ID" = "Genotype_ID",
                       "Treatment_ID" = "Treatment_ID"),
        selected = 1
      )
    ),
    card(
      card_header("Imputation Technique"),
      selectInput(
        inputId = "Imputation",
        label = "Select Imputation Technique",
        choices = list("MICE" = "mice_impute",
                       "missForest" = "missForest_impute",
                       "Delete Missing Values" = "delete_na"),
        selected = "mice_impute"
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
  card(
    card_header("Results"),
    textOutput("save_status"),
    tableOutput("data_preview"),
    textOutput("model_output")
  )
)
