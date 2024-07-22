library(shiny)
library(ggplot2)
library(dplyr)
library(DT)

# Function to create histograms for original and imputed data
create_histograms <- function(original_data, imputed_data) {
  plot_list <- list()
  
  for (feature in names(original_data)) {
    if (is.numeric(original_data[[feature]])) {
      p <- ggplot() +
        geom_histogram(data = original_data, aes_string(x = feature), fill = "blue", alpha = 0.5, bins = 30) +
        geom_histogram(data = imputed_data, aes_string(x = feature), fill = "red", alpha = 0.5, bins = 30) +
        labs(title = paste("Feature:", feature), x = feature, y = "Frequency") +
        theme_minimal()
      plot_list[[feature]] <- p
    }
  }
  
  return(plot_list)
}

# UI for the histogram module
histogramModuleUI <- function(id) {
  ns <- NS(id)
  fluidPage(
    textInput(ns("search"), "Search for a feature:", value = ""),
    DTOutput(ns("features_table")),
    plotOutput(ns("feature_plot"))
  )
}

# Server for the histogram module
histogramModuleServer <- function(id, original_data, imputed_data) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns
    
    # Create histograms
    plot_list <- create_histograms(original_data, imputed_data)
    
    # Create a reactive dataframe for features
    features_df <- reactive({
      data.frame(Feature = names(plot_list))
    })
    
    # Render the datatable
    output$features_table <- renderDT({
      datatable(features_df(), options = list(pageLength = 10, searchHighlight = TRUE))
    })
    
    # Render the plot
    output$feature_plot <- renderPlot({
      req(input$features_table_rows_selected)
      selected_feature <- features_df()$Feature[input$features_table_rows_selected]
      plot_list[[selected_feature]]
    })
  })
}
