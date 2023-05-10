# Assignment Part two

# Authors: 
# Chun-Ting Wu r0915592
# Yu-Jie Qiu r0823712

#########################
library(dplyr)
library(car)
library(boot)
library(ISLR)
library(glmnet)
library(leaps)
library(splines)
library(gam)
library(corrplot)


## 1 Study and describe the predictor variables. Do you see any issues that are relevant for making predictions?

load("prostate.Rdata")
data <- prostate
attach(data)
head(data)
dim(data) # 97 8 ==> 8 variables for 97 data points
sum(is.na(data)) #0
data$svi<-as.factor(data$svi)
str(data)
View(data)
summary(data)
summarise(data, mean(Cscore), median(Cscore), n=n(), sd(Cscore))

# 1 svi is unbalance sample size
# 2 Cscore range is dramatically high

## histograms: shape of Cscore distribution
hist(data$Cscore, prob = TRUE,col = 'red',xlab='Cscore',main='Histogram of Cscore', ylim=c(0,0.02))
lines(density(data$Cscore))
shapiro.test(data$Cscore) # p<0.05 => not normal distribution

## boxplot
# by_svi: Cscore are significantly different between group 0 and 1.
by_svi <- group_by(data, svi)
summarise(by_svi, n=n(), mean(Cscore), sd(Cscore), var(Cscore))
boxplot(Cscore ~ svi, data = data,col = 'red',main='Comparison of Cscore level between svi group', ylab = "Cscore")

data.wilcox <- wilcox.test(data$Cscore ~ data$svi) 
data.wilcox # p<0.05 =>  two populations have different continuous distribution

res <- var.test(Cscore ~ svi, data = data)
res # p<0.05 => there is a significant difference between the two variances



## 2 Generate your best linear regression model using only linear effects. 
## Are there any indications that assumptions underlying inferences with the model are violated? 
## Evaluate the effect of any influential point, or outlier.

## correlation
par(mfrow=c(1,1))
data$svi<-as.numeric(data$svi)
data.cor <- cor(data)
data.cor # Pearson’s linear correlations
corrplot.mixed(data.cor, tl.col = "black",tl.cex=1)
data$svi<-as.factor(data$svi)
plot(data)

## collinearity check
data.lm.all <- lm(Cscore~.,data=data)
summary(data.lm.all)
#lpsa
#residual = 34 , r2 = 0.56 

vif(data.lm.all)
# Rule of thumb: 
# => all vif lower than 5 
# => no collinearity

## residual
par(mfrow=c(2,3))
plot(data.lm.all) 
# outlier: index 96 => low leverage , high residual 
# => biased
# leverage point: index 32 => high leverage , low residual
# => not impact the slope of the regression line

res <- resid(data.lm.all)
plot(density(res))

## remove outlier
sd(data$Cscore) #52
var(data$Cscore) #2779

omit_data <- data[-c(96),] # remove row when index 96
sd(omit_data$Cscore) # => after omit, the standard deviation decrease 52->40
var(omit_data$Cscore) #  => after omit, the variance decrease 2779 -> 1601

data.lm.all.omit = lm(Cscore ~ ., data = omit_data)
summary(data.lm.all.omit) 
#lcp,lpsa
#residual = 24 (decreased after omit), r2 = 0.63 




####log transformation####
attach(omit_data)
omit_data_log <- omit_data
omit_data_log$Cscore <- log(Cscore + 1 +abs(min(Cscore)));

par(mfrow=c(1,1))
hist(omit_data_log$Cscore)
plot(omit_data_log)

omit_data_log.lm <- lm(omit_data_log$Cscore ~., data=omit_data_log)
summary(omit_data_log.lm)
#lcp,lpsa
#residual = 0.57 (lower than not log transform), r2 = 0.49 (lower than not log transform)

#residual
par(mfrow=c(2,3))
plot(omit_data_log.lm) 
res <- resid(omit_data_log.lm)
plot(density(res))
shapiro.test(res) # p<0.05 => not normal distribution

############



## best Subset Selection 
best <- regsubsets(Cscore~.,data=omit_data)
bestSUM <- summary(best)

par(mfrow=c(1,2))
plot(bestSUM$bic,type="b",ylab="BIC",main ="Best Subset Selection")
points(2,bestSUM$bic[2], col ="red",cex =2, pch =20) 
plot(best, scale="bic",main ="Best Subset Selection")
which.min(bestSUM$bic)
co=coef(best,2)
names(co)
coef(best, 2) 
# 8.5  21.2  
# => lcp,lpsa

## forward selection
fwd <- regsubsets(Cscore~.,data=omit_data,method="forward")
fwdSUM <- summary(fwd)
plot(fwdSUM$bic,type="b",ylab="BIC",main ="Forward selection")
points(2,fwdSUM$bic[2], col ="red",cex =2, pch =20) 
plot(fwd, scale="bic",main ="Forward selection")
which.min(fwdSUM$bic)
co=coef(fwd,2)
names(co)
coef(fwd, 2) 
#8.5  21.2  
# => lcp,lpsa

## backward selection
bwd <- regsubsets(Cscore~.,data=omit_data,method="backward")
bwdSUM <- summary(bwd)
plot(bwdSUM$bic,type="b",ylab="BIC",main ="Backward selection")
points(2,bwdSUM$bic[2], col ="red",cex =2, pch =20) 
plot(bwd, scale="bic",main ="Backward selection")
which.min(bwdSUM$bic)
co=coef(bwd,2)
names(co)
coef(bwd, 2) 
# 8.5  21.2 
# => lcp,lpsa 


data.lm.omit = lm(Cscore ~ lcp + lpsa, data = omit_data)
summary(data.lm.omit) 
#8.5 21.2
#lcp,lpsa


# They all select the same variables: 
# lcp,lpsa


## 3 Make an appropriate LASSO model, with the appropriate link and error function, and
## evaluate the prediction performance. Do you see evidence that over-learning is an issue?

set.seed(1)
train=sample(1:nrow(omit_data), nrow(omit_data)*2/3)
test=(-train)
x=model.matrix(Cscore~.,omit_data)[,-1] #delete Cscore column
y=omit_data$Cscore
y.train=y[train]
y.test=y[test]
x.train=x[train,]
x.test=x[test,]
omit_data.train=omit_data[train,]
omit_data.test=omit_data[test,]
dim(omit_data.train) #64,8
dim(omit_data.test) #32,8

## Lasso regression
par(mfrow=c(1,2))
lasso.mod=glmnet(y=y.train,x=x.train,alpha=1)
plot(lasso.mod,label=TRUE)
lasso.cv=cv.glmnet(x.train,y.train,alpha=1) # 10-fold cross validation
plot(lasso.cv) 

########
# lable variables 1
library(plotmo)
plot_glmnet(lasso.mod,label=TRUE,xvar="lambda",s=lasso.cv$lambda.min)


# lable variables 2
lbs_fun <- function(fit, ...) {
  L <- length(fit$lambda)
  x <- log(fit$lambda[L])
  y <- fit$beta[, L]
  labs <- names(y)
  text(x, y, labels=labs, ...)
  #legend('topright', legend=labs, col=1:length(labs), lty=1) 
}
plot(lasso.mod, xvar="lambda", col=1:dim(coef(lasso.mod))[1])
lbs_fun(lasso.mod)

#######


bestlam<-lasso.cv$lambda.min
bestlam ## Select lamda that minimizes training MSE
min(lasso.cv$cvm)
# bestlam = 0.7 results in the smallest cross-validation error 643


# Prediction and evaluation on test data
lasso.pred=predict(lasso.mod,s=bestlam,newx=x.test)
mean((lasso.pred-y.test)^2)
# the test MSE = 750

lasso.pred.train=predict(lasso.mod,s=bestlam,newx=x.train)
mean((lasso.pred.train-y.train)^2)
# training MSE = 468



##### Lasso regression with LOOCV ####
par(mfrow=c(1,2))
lasso.mod=glmnet(y=y.train,x=x.train,alpha=1)
plot(lasso.mod)
lasso.cv=cv.glmnet(x.train,y.train,alpha=1, nfolds=96) # 96-fold cross validation
plot(lasso.cv) 

bestlam<-lasso.cv$lambda.min
bestlam ## Select lamda that minimizes training MSE
min(lasso.cv$cvm)
# bestlam = 0.5 results in the smallest cross-validation error 651


# Prediction and evaluation on test data
lasso.pred=predict(lasso.mod,s=bestlam,newx=x.test)
mean((lasso.pred-y.test)^2)
# the test MSE = 757

lasso.pred.train=predict(lasso.mod,s=bestlam,newx=x.train)
mean((lasso.pred.train-y.train)^2)
# the training MSE = 465



###############not use this
## linear regression
linear.mod=lm(y.train~x.train)
summary(linear.mod)
# => lcavol, lcp, lpsa

mean(linear.mod$residuals^2)
# the training MSE = 460

# use fitted model to make predictions
linear.pred=predict(linear.mod,newx=x.test)
mean((y.test-linear.pred)^2)
# the test MSE = 2658

# the MSE of the linear model (2658) is larger than the lasso model (750)
# => the variance of lasso modle is smaller. 
# => there is no over-learning issue.
##############






## 4 Look at the coefficient for “lcavol” in your LASSO model. 
## Does this coefficient correspond to how well it can predict Cscore? Explain your observation.

coef(lasso.cv)
# lcavol, lweight, age, lbph coefficient = 0
# svi = 6
# lcp = 2
# lpsa = 15 => important variables





## 5 Fit your best model with appropriate non-linear effects. 
## Report a comparison of performance to LASSO and your model reported under question 2. 
## Explain what you find,and indicate relevant issues or limitations of your analysis.


