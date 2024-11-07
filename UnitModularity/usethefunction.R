#DAN: Typically the function goes in another script and it is source'd and then run in main.R. 
#You did the reverse.
#DAN: The code does run on the first try, which is great.

source("main.R")


#temperature data
analyze_climate_data("USAAnnualTemp1950_2008.rds", "data")

#precipitation data
analyze_climate_data("USAAnnualPcpn1950_2008.rds", "data")
#DAN: Great that you have one function, run twice.