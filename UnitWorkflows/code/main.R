#DAN: Overall, all the elements seem to be here, but I could not get various things to compile/run,
#so the code was not as portable as it could be. See my comments. 
#Grade: S

#DAN: I see you committed some .tex files and .png files to git. That means every time you
#run your code, git will prompt you to commit the new versions of those files, which will
#tend to fill up your github repo. Just commit your source code.

#DAN: You need a readme.md file, to tell the user how to reproduce your results (see the example repo
#I liked from class)

# main.R

if (!dir.exists("results")) dir.create("results")

# Load required libraries
library(rpart)
library(rpart.plot)
library(caret)
library(ipred)
library(randomForest)
library(adabag)
#library(xgboost) #DAN: xgboost is huge, and I am having disk space issues so I deleted it
#from my machine. It was not needed - I commented it here and everything worked. Best to 
#avoid unnecessary dependencies because them people will find themselves installing packages
#they don't need!
#
#Aside from that, though, this ran!

source("code/analysis.R")

print("Analysis complete. Check the 'results' folder for outputs.")
