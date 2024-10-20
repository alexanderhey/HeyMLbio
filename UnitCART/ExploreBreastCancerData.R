data <- read.csv( "data.csv")

#Removing columns I don't want: 
library(dplyr)

# Remove specific columns (e.g., column1 and column2)
data2 <- data %>% select(-matches("_se|_worst"), -id)

write.csv(data2, "cancer_data_cleaned.csv")

#DAN: This is very succinct code, which is good, but it's pretty limited in terms
#of exploring the data at all. You could also look at the "head", check dimensions,
#check for NAs, etc. Perhaps you did that command line...
 
 
 