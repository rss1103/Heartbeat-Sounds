library (e1071)
library(caret)
#library(gglasso)
data = read.csv("F:/MS/ISTE-780 Data Driven Knowledge Discovery/KDD_Project/pca_data.csv", header=T)
rownames(data) = data[,1]
data = data[,c(-1,-2)]
data$label <- relevel(data$label, ref = "normal")
#pairs(data[,c(-1,-2)])
#PCA
#pcadata = data
#pcadata <- prcomp(pcadata[,-1], scale. = T)
#std_dev <- pcadata$sdev
#pr_var <- std_dev^2
#prop_varex <- pr_var/sum(pr_var)
#pr_var[1:10]
#plot(prop_varex, xlab = "Principal Component",
# ylab = "Proportion of Variance Explained",
#type = "b")

#plot(cumsum(prop_varex), xlab = "Principal Component",
#ylab = "Cumulative Proportion of Variance Explained",
#type = "b")

#add a training set with principal components
#reduced_data <- data.frame(label = data$label, pcadata$x)

#we are interested in first 30 PCAs
#reduced_data <- reduced_data[,1:45]
#reduced_data$label <- relevel(reduced_data$label, ref = "normal")
# Prepare Training and Test Data
set.seed(123)
trainingRows <- sample(1:nrow(data), 0.85*nrow(data))
training <- data[trainingRows, ]
test <- data[-trainingRows, ]

#fit <- multinom(label ~ ., data = training)
x <- subset(training, select=-label)
y <- training$label
dat=data.frame(x=x, y=as.factor (y))

#svm_tune <- tune(svm, train.x=x, train.y=y, kernel="sigmoid", ranges=list(cost=10^(-1:3), gamma=c(.5,1,2)))
#print(svm_tune)
#regfit.full=regsubsets(y~.,dat)
#summary (regfit.full)
#CV.out = cv.glmnet(x,y,alpha=1)
#lasso.mod =glmnet(x,y,alpha =1, lambda =grid)
#cv.gglasso(x,y, group=group, loss="logit",pred.loss="misclass", lambda.factor=0.05, nfolds=5)
#x = as.matrix(x)
#cov(x)
#cor(x)
#group <- rep(1:82,each=10)
#cv <- cv.gglasso(x=x, y=y, group=group, loss="logit",pred.loss="misclass", lambda.factor=0.05, nfolds=10)
#svmfit =svm(y~., data= dat,kernel ="radial",gamma=0.01, cost = 10,scale=TRUE)
svmfit =svm(y~., data= dat,kernel ="radial",gamma=0.01,cost = 100,scale=TRUE)
predict_train = predict(svmfit,newdata = dat)
table(predict_train,y)
summary (svmfit)
#tune.out=tune(svm ,y~.,data=dat ,kernel ="linear",ranges =list(cost=c(0.001 , 0.01, 0.1, 1,5,10,100)))
#summary(tune.out)
#bestmod = tune.out$best.model
#summary(bestmod)
xtest = subset(test,select=-label)
ytest = test$label
testdat =data.frame (x=xtest , y=as.factor(ytest))
predicted_class = predict (svmfit,newdata = testdat)
table(predicted_class,ytest)
cm = table(predicted_class, testdat$y)
n = sum(cm)
nc = nrow(cm)
diag = diag(cm)
rowsums = apply(cm, 1, sum)
colsums = apply(cm, 2, sum)
p = rowsums / n
q = colsums / n
cm
accuracy = sum(diag) / n 
accuracy
rm(list = ls())

