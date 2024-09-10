data <- read.csv( "data.csv")

#Removing columns I don't want: 
library(dplyr)

# Remove specific columns (e.g., column1 and column2)
data2 <- data %>% select(-matches("_se|_worst"), -id)

write.csv(data2, "cancer_data_cleaned.csv")


 
 
 