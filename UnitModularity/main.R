#DAN: There are some good example of modularity here, but your answer is not correct (as you
#have observed - also see my code and results). Part of that is probably due to trying to do the filtering
#and regression steps at once. See comments below. 
#Grade: S

#DAN: Put a header at the top 
#DAN: The function should be defined in another script and then *used* in main. You did the reverse.
#The results disagrees with what I got. Check my code. 


library(ggplot2)
library(dplyr)
library(maps)
library(scales)

#MODULARITY EXAMPLE 1:Define modules and interfaces clearly..
analyze_climate_data <- function(rds_file, variable_name) {
  
  #MODULARITY EXAMPLE 2:Keep the function inputs flexible.
  #Load the data from any .rds file (I hope all .rds are structured like this)
  df <- readRDS(rds_file)
  
  #Reporting the names of the columns for the  usr
  cat("Column names in the dataset are:\n")
  print(names(df))
  
  #reporting the specific column being analyzed
  cat("Analyzing column:", variable_name, "\n")
  
  #MODULARITY EXAMPLE 3: Use variables instead of hardcoding constants.
  min_measurements <- 40  #I think this is using a variable, but I set it as a constant, could likely be passed as value later
  
  #MODULARITY Examaple 4:Write extensible code. Make the slope calculation a function
  calculate_slopes <- function(data, variable_name) {
    data %>% 
      group_by(name) %>%
      filter(n() >= min_measurements, !all(is.na(get(variable_name)))) %>%  #Make sure there is enough data. could add an output here to tell you that it's not enough
      summarize(
        slope = ifelse(length(unique(get(variable_name))) > 1,
                       coef(lm(get(variable_name) ~ year))[2],
                       NA),  #found ifelse online to deal with the NAs or only one unique value in data column
        lon = mean(lon, na.rm = TRUE),  # Remove NAs and get an average lat/lon
        lat = mean(lat, na.rm = TRUE)
      ) %>%
      filter(!is.na(slope))  #filter out locations with no slope
  }
  #DAN: The problem with the tidyverse is it is hard to test, because all steps are piled together with pipes. 
  #You result disagrees with mine (and with climate science). My result agrees with other students. Hard to say
  #because of use of tidyverse, what precisely the problem us without really delving into it. 
  #DAN: One problem seems to be you are trying to filter the sites (to use only those with 40) and also get
  #the slopes in one function. Would be better (clearer and less error prone) to break that up.
  
  #MODULARITY EXAMPLE 5: Avoid hardcoding "data"; pass column names as a variable.
  #Calculate slopes for the provided data
  slopes <- calculate_slopes(df, variable_name)
  
  #MODULARITYEXAMPLE 6: Write reusable code for plots. Function for creating histograms.
  #Function for creating histograms with mean and median
  plot_histogram <- function(slopes, title) {
    
    #mean and median of the slopes. get rid of NAs
    mean_slope <- mean(slopes$slope, na.rm = TRUE)
    median_slope <- median(slopes$slope, na.rm = TRUE)
    
    #create histogram
    p <- ggplot(slopes, aes(x = slope)) +
      geom_histogram(binwidth = 0.1, fill = "blue", color = "black") +
      
      #vert lines for mean and median. can't see these much
      geom_vline(aes(xintercept = mean_slope), color = "red", linetype = "dashed", size = 1) +
      geom_vline(aes(xintercept = median_slope), color = "green", linetype = "dotted", size = 1) +
      
      #Add text annotations to report mean and median
      annotate("text", x = Inf, y = Inf, label = paste("Mean:", round(mean_slope, 2)), 
               hjust = 1.1, vjust = 2, color = "red", size = 4) +
      annotate("text", x = Inf, y = Inf, label = paste("Median:", round(median_slope, 2)), 
               hjust = 1.1, vjust = 3.5, color = "green", size = 4) +
      
      labs(title = title, x = "Slope", y = "Frequency") +
      theme_minimal()
    
    print(p)
  }
  
  plot_histogram(slopes, "Histogram of Slopes")
  
  #Function to create a map of slopes
  plot_map <- function(slopes, title) {
    p <- ggplot(data = slopes, aes(x = lon, y = lat, color = slope)) +
      borders("world", fill = "gray80", color = "black") +  #Draw world borders
      borders("state", fill = NA, color = "black") +  #Draw US state borders
      geom_point(size = 0.5) + #needed to mess with this
      scale_color_gradientn(
        colors = c("blue", "lightblue", "white", "pink", "red"),
        values = scales::rescale(c(-5, -0.5, 0, 0.5, 5)),
        limits = c(-5, 5),
        oob = scales::squish,
        name = "Slope"
      ) +
      coord_fixed(xlim = c(-170, -65), ylim = c(20, 72)) +  # Include the USA and Alaska. sorry Hawaii
      labs(title = title) +
      theme_minimal() +
      theme(legend.position = "bottom")
    print(p)  #print the plot
  }
  
  #map of slopes
  plot_map(slopes, "Slopes")
}

#MODULARITY EXAMPLE 7:Write pseudocode or comments before actual coding to structure tasks.



