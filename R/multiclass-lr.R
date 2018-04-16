library(nnet)

data = read.csv("pca_data.csv", header=T)
rownames(data) = data[,1]
data = data[,c(-1,-2)]
data$label <- relevel(data$label, ref = "normal")


# Prepare Training and Test Data
set.seed(100)
trainingRows <- sample(1:nrow(data), 0.8*nrow(data))
training <- data[trainingRows, ]
test <- data[-trainingRows, ]

#training$label <- relevel(training$label, ref = "normal")
fit <- multinom(label ~ ., data = training)
summary(fit)

#z <- summary(fit)$coefficients/summary(fit)$standard.errors
#z

predicted_class <- predict (fit, newdata = test)
cm = as.matrix(table(predicted_class, test$label))
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

accuracy = sum(diag) / n 
accuracy

