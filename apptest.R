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
        selected = "Leaf_No",
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
        selected = "Blade_Width",
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
      card_header("Train Set"),
      sliderInput(
        "Train_Set",
        "Percentage %",
        min = 0,
        max = 100,
        value = 80
      ),
    ),
  ),
  card(
    card_header(),
    textOutput("selected_Categorical"),
    textOutput("selected_Numerical"),
    textOutput("selected_Predictive"),
    textOutput("selected_Trainset")
  ),
)

# Define server logic ----
server <- function(input, output) {
  bs_themer()
  output$selected_Categorical <- renderText({
    paste("You have selected", input$Categorical)
  })
  output$selected_Numerical <- renderText({
    paste("You have selected", input$Numerical)
  })
  output$selected_Predictive <- renderText({
    paste("You have selected", input$Predictive)
  })
  output$selected_Trainset <- renderText({
    paste("You have selected", input$Train_Set)
  })
}

# Run the app ----
shinyApp(ui = ui, server = server)