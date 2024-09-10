
#loadin data 
data <- read.csv("Frogs_MFCCs.csv")

#I can't remove any columns, they all seem important

#checking for missing sata
summary(data)
sum(is.na(data))
# no missing data here

#inspecting histograms
hist(data$MFCCs_.8)

#they all look different, I can't say one is better than the others or one needs to be tossed

