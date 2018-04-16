#Randomforest for feature selection
set.seed(123)
traindata <- read.csv("features.csv", header = T, stringsAsFactors = F)
library(Boruta)
boruta.train <- Boruta(mfcc1~., data = traindata, doTrace = 2)
#Results
stats = attStats(boruta.train)
print(stats)
options(max.print = 100000)