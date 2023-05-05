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

data.lm.omit = lm(Cscore ~ ., data = omit_data)
summary(data.lm.omit) #Residual standard error decreased


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

# They all select the same variables: 
# lcp,lpsa








## 3 Make an appropriate LASSO model, with the appropriate link and error function, and
## evaluate the prediction performance. Do you see evidence that over-learning is an issue?






## 4 Look at the coefficient for “lcavol” in your LASSO model. 
## Does this coefficient correspond to how well it can predict Cscore? Explain your observation.






## 5 Fit your best model with appropriate non-linear effects. 
## Report a comparison of performance to LASSO and your model reported under question 2. 
## Explain what you find,and indicate relevant issues or limitations of your analysis.


