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
library(plotmo)
library(ggplot2)

## 1 Study and describe the predictor variables. Do you see any issues that are relevant for making predictions?

load("prostate.Rdata")
data <- prostate
attach(data)
head(data)
dim(data) #n=97, predictor=7, Cscore is the response
sum(is.na(data)) #0
data$svi<-as.factor(data$svi)
str(data)

summary(data)
summarise(data, mean(Cscore), median(Cscore), n=n(), sd(Cscore))
# 1 Cscore distribution is right skewed
# 2 svi is unbalance sample size


## histograms: shape of Cscore distribution
par(mfrow=c(1,1))
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
vif(data.lm.all)
# Rule of thumb: 
# => all vif lower than 5 
# => no collinearity



## 2 Generate your best linear regression model using only linear effects. 
## Are there any indications that assumptions underlying inferences with the model are violated? 
## Evaluate the effect of any influential point, or outlier.

## Residual plot
#To evaluate the effect of any influential points or outliers, 
#we can examine influential plot and diagnostic plots of the model.
par(mfrow=c(2,3))
influencePlot(data.lm.all)
plot(data.lm.all) 
##This will generate a plot of the residuals against the fitted values, 
#a normal probability plot of the residuals,
#a plot of the residuals against the leverage values
#and a Cook's distance plot. 
#
# outlier: index 96 => low leverage , high residual 
# => biased
# leverage point: index 32 => high leverage , low residual
# => not impact the slope of the regression line

#resudial distribution
res <- resid(data.lm.all)
plot(density(res))
shapiro.test(res) # p<0.05 => not normal distribution

#added-variable plots
#Show the relationship between the dependent variable and the independent variable
#while holding all other variables constant
avPlots(data.lm.all)
#The index 96 indeed is an outlier

###
#If we identify any influential points or outliers, we may want to consider 
#removing them and refitting the model to see how the results are affected. 
#However, it's important to be cautious when removing data points, as this 
#can lead to biased or misleading results if not done carefully.
###

## remove outlier
data.lm.all <- lm(Cscore~.,data=data) #Fit a linear regression model with all predictors
summary(data.lm.all)
#lpsa***
#residual = 35 , Adjusted R-squared = 0.56 
sd(data$Cscore) #52
var(data$Cscore) #2779

omit_data <- data[-c(96),] # remove entire row when index is 96
sd(omit_data$Cscore) # => after omit, the standard deviation decrease 52 -> 40
var(omit_data$Cscore) #  => after omit, the variance decrease 2779 -> 1601

attach(omit_data)
data.lm.all.omit = lm(Cscore ~ ., data = omit_data)
summary(data.lm.all.omit)
#lcp**,lpsa***
#after omit, the residual decrease 35 -> 24, Adjusted R-squared increase 0.56 -> 0.63 


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



## validation set approach
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(omit_data ), rep=TRUE)
test <- (!train)

#apply regsubsets() to the training set in order to perform best subset selection
regfit.best <- regsubsets(Cscore~., data = omit_data [train,], nvmax=7)

#compare the validation set error
test.mat=model.matrix (Cscore~., data=omit_data [test ,])
#compute the test MSE
val.errors <- rep(NA,7)
for(i in 1:7){
  coefi = coef(regfit.best, id=i)
  pred=test.mat[,names(coefi)]%*%coefi
  val.errors[i]=mean((omit_data $Cscore[test]-pred)^2)
}
val.errors
par(mfrow =c(1,1))
plot(val.errors, type="b", xlab="Number of Predictors", ylab="Validation Set Error",main="Validation Set Errors")
points(which.min(val.errors), val.errors[which.min(val.errors)], col = "red", cex = 2, pch = 20)
which.min(val.errors)
coef(regfit.best, 1)

#use full model
regfit.best <- regsubsets(Cscore~., data =omit_data , nvmax=7)
coef(regfit.best,1)
#26.88
#lpsa



## 10-fold cross validation
predict.regsubsets =function (object ,newdata ,id ,...){
  form=as.formula (object$call [[2]])
  mat=model.matrix (form ,newdata )
  coefi =coef(object ,id=id)
  xvars =names (coefi )
  mat[,xvars ]%*% coefi
}

set.seed(1)
k=10
folds=sample(1:k, nrow(omit_data), replace=TRUE)
cv.errors=matrix(NA, k, 7, dimnames=list(NULL, paste(1:7)))
#perform cross-validation
for(j in 1:k){
  best.fit=regsubsets(Cscore~., data=omit_data[folds!=j,], nvmax=7)
  for(i in 1:7){
    pred=predict(best.fit, omit_data[folds==j,], id=i)
    cv.errors[j,i]=mean((omit_data$Cscore[folds==j]-pred)^2)
  }
}
#obtain cross-validation error
mean.cv.errors =apply(cv.errors ,2, mean)
mean.cv.errors
which.min(mean.cv.errors)
#2

par(mfrow =c(1,1))
plot(mean.cv.errors,xlab="Number of Predictors", ylab="10-fold Cross-Validation Error", main="Cross-Validation Errors", type="b")
points(which.min(mean.cv.errors), mean.cv.errors[which.min(mean.cv.errors)], col = "red", cex = 2, pch = 20)

#use full model
regfit.best <- regsubsets(Cscore~., data=omit_data, nvmax=7)
coef(regfit.best, 2)
#8.52  21.15
#lcp  lpsa

# => Cscore = -17.46 + 8.5*lcp + 21.2*lpsa + irreducible error






## 3 Make an appropriate LASSO model, with the appropriate link and error function, and
## evaluate the prediction performance. Do you see evidence that over-learning is an issue?
#evaluate the prediction performance of the LASSO model using
#the test set and calculate the mean squared error (MSE). 
set.seed(1)
train_index <- sample(nrow(omit_data), nrow(omit_data) * 2/3) 
train <- omit_data[train_index, ]
test <- omit_data[-train_index, ]

grid =10^seq(10,-2, length =100)
x_train <- model.matrix(Cscore ~ ., train)[,-1]
y_train <- train$Cscore
x_test <- model.matrix(Cscore ~ ., test)[,-1]
y_test <- test$Cscore

### Lasso regression with 10-fold Cross Validation ###
lasso_mod <- glmnet(x_train, y_train, alpha =1, lambda =grid)
lasso_fit <- cv.glmnet(x_train, y_train, alpha = 1, nfolds=10)
plot(lasso_fit,main="10 fold Cross Validation")

## lambda.min
lambda_best <- lasso_fit$lambda.min
lambda_best
min(lasso_fit$cvm)
# lambda_best = 0.7 results in the smallest cross-validation error 643


## Get the index of the lambda with the minimum CVM (cross-validation error mean)
lambda_idx <- which.min(lasso_fit$cvm) 
lambda_idx
# Get the CVM for the lambda with the minimum CVM
cvm <- lasso_fit$cvm[lambda_idx] 
cvm  #643.2
# Get the CVSD (standard deviation) for the lambda with the minimum CVM
cvsd <- lasso_fit$cvsd[lambda_idx] 
cvsd #155.03

## Calculate the MSE on the training set
y_train_pred <- predict(lasso_mod,s=lambda_best, newx = x_train)
mse_train <- mean((y_train_pred - y_train)^2)
mse_train 
#468

# Calculate the MSE on the test set
y_test_pred <- predict(lasso_mod,s=lambda_best, newx = x_test)
mse_test <- mean((y_test_pred - y_test)^2)
mse_test
#750

# => the MSE on the test set is much higher than the MSE on the training set, 
#   which may suggest that the LASSO model is overfitting to the training data.

#Generate standardized LASSO coefficients
out=glmnet(omit_data[,-1],omit_data$Cscore, alpha =1, lambda =grid)
lasso.coef.standardized <- coef(out, s = lambda_best, 
                                x = x_train, y = y_train, 
                                standardize = TRUE)[1:8,]
lasso.coef.standardized 
#==> Cscore = -20.13 -3.09lcavol -2.79lweight -1.66lbph +11.53svi +7.03lcp +22.38lpsa +irreduciable error




### Lasso regression with LOOCV ###
set.seed(1)
train_index <- sample(nrow(omit_data), nrow(omit_data) * 2/3) 
train <- omit_data[train_index, ]
test <- omit_data[-train_index, ]

grid =10^seq(10,-2, length =100)
x_train <- model.matrix(Cscore ~ ., train)[,-1]
y_train <- train$Cscore
x_test <- model.matrix(Cscore ~ ., test)[,-1]
y_test <- test$Cscore

lasso_mod_loocv <- glmnet(x_train, y_train, alpha =1, lambda =grid)
lasso.cv=cv.glmnet(x_train, y_train,alpha=1, nfolds=96) # 96-fold cross validation
plot(lasso.cv, ylim = c(500, 2000), main="Leave One Out Cross Validation") 

bestlam<-lasso.cv$lambda.min
bestlam ## Select lamda that minimizes training MSE
#0.55
min(lasso.cv$cvm)
# bestlam = 0.56 results in the smallest cross-validation error 651

## Get the index of the lambda with the minimum CVM (cross-validation error mean)
lambda_idx <- which.min(lasso.cv$cvm) 
lambda_idx
# Get the CVM for the lambda with the minimum CVM
cvm <- lasso.cv$cvm[lambda_idx] 
cvm  #651
# Get the CVSD (standard deviation) for the lambda with the minimum CVM
cvsd <- lasso.cv$cvsd[lambda_idx] 
cvsd #146

## Calculate the MSE on the training set
lasso.pred.train=predict(lasso_mod_loocv,s=bestlam,newx=x_train)
mean((lasso.pred.train-y_train)^2)
#465.29

#Calculate the MSE on the test set
lasso.pred=predict(lasso_mod_loocv,s=bestlam,newx=x_test)
mean((lasso.pred-y_test)^2)
#757.24

# => the MSE on the test set is much higher than the MSE on the training set, 
#   which may suggest that the LASSO model is overfitting to the training data.

# Generate standardized LASSO coefficients
out.loocv=glmnet(omit_data[,-1],omit_data$Cscore, alpha =1, lambda =grid)
lasso.coef.loocv <- coef(out.loocv, s = bestlam, 
                         x = x_train, y = y_train, 
                         standardize = TRUE)[1:8,]
lasso.coef.loocv


#### lambda_best_se (10-fold CV) ####
lambda_1se <- lasso_fit$lambda.1se
lambda_1se
min(lasso_fit$cvm)
# lambda_best_se = 9.1 results in the smallest cross-validation error 643
# =>much higher than lambda_best, means that we expect the coefficients under 1se
#   to be much smaller or exactly zero

## Get the index of lambda_1se in the lambda sequence
lambda_1se_index <- which(lasso_fit$lambda == lambda_1se)
lambda_1se_index
# Get the cvm and cvd values for lambda_1se
cvm_lambda_1se <- lasso_fit$cvm[lambda_1se_index]
cvm_lambda_1se
#787
cvd_lambda_1se <- lasso_fit$cvsd[lambda_1se_index]
cvd_lambda_1se
#277

# Calculate the MSE on the training set
y_train_pred <- predict(lasso_fit,s=lambda_1se, newx = x_train)
mse_train_se <- mean((y_train_pred - y_train)^2)
mse_train_se 
#655

# Calculate the MSE on the test set
y_test_pred <- predict(lasso_mod,s=lambda_1se, newx = x_test)
mse_test_se <- mean((y_test_pred - y_test)^2)
mse_test_se
#844

# the difference between training and test MSE:
# lambda.min : 282
# lambda.1se : 189
# => lambda.1se (higher penalty) get smaller difference between training and test MSE.


## Generate 1se LASSO coefficients
out=glmnet(omit_data[,-1],omit_data$Cscore, alpha =1, lambda =grid)
lasso.coef.standardized <- predict(out, s = lambda_1se, type = "coefficients",
                                   standardize = TRUE)[1:8,]
lasso.coef.standardized 
#==> Cscore = -10.30 +5.00svi +3.48lcp +15.36lpsa +irreduciable error



#### lambda_best_1se (LOOCV) ####
lambda_1se <- lasso.cv$lambda.1se
lambda_1se
min(lasso.cv$cvm)
# lambda_best_se = 9.1 results in the smallest cross-validation error 651
# =>much higher than lambda_best, means that we expect the coefficients under 1se
#   to be much smaller or exactly zero

## Get the index of lambda_1se in the lambda sequence
lambda_1se_index <- which(lasso.cv$lambda == lambda_1se)
lambda_1se_index
# Get the cvm and cvd values for lambda_1se
cvm_lambda_1se <- lasso.cv$cvm[lambda_1se_index]
cvm_lambda_1se
#770
cvd_lambda_1se <- lasso.cv$cvsd[lambda_1se_index]
cvd_lambda_1se
#206

# Calculate the MSE on the training set
y_train_pred <- predict(lasso.cv,s=lambda_best_se, newx = x_train)
mse_train_se <- mean((y_train_pred - y_train)^2)
mse_train_se 
#655

# Calculate the MSE on the test set
y_test_pred <- predict(lasso_mod,s=lambda_best_se, newx = x_test)
mse_test_se <- mean((y_test_pred - y_test)^2)
mse_test_se
#844

# the difference between training and test MSE:
# lambda.min : 292
# lambda.1se : 189
# => lambda.1se (higher penalty) get smaller difference between training and test MSE.


## Generate 1se LASSO coefficients
out=glmnet(omit_data[,-1],omit_data$Cscore, alpha =1, lambda =grid)
lasso.coef.standardized <- predict(out, s = lambda_1se, type = "coefficients",
                                   standardize = TRUE)[1:8,]
lasso.coef.standardized 
#==> Cscore = -10.30 +5.00svi +3.48lcp +15.36lpsa +irreduciable error



##To further evaluate if overfitting is an issue, you can also plot the predicted 
#values versus the actual values for both the training and test sets.

#If the plot for the test set shows a more scattered pattern than the plot 
#for the training set, this may also indicate overfitting. 
par(mfrow = c(1,2))
plot(y_train, y_train_pred, main = "Training set")
abline(0,1)
plot(y_test, y_test_pred, main = "Test set")
abline(0,1)

#The first panel: the predicted values versus the actual values for the training set
#The second panel will show the same for the test set. 
#The abline(0,1) function adds a reference line to the plot with a slope of 1 and an intercept of 0,
#which represents perfect prediction. 
#=>If the points on the plot are close to this reference line, 
#  it suggests that the model is doing a good job of predicting the outcomes.



## 4 Look at the coefficient for “lcavol” in your LASSO model. 
## Does this coefficient correspond to how well it can predict Cscore? Explain your observation.

## Lasso regression with 10-fold Cross Validation ##
out=glmnet(omit_data[,-1],omit_data$Cscore, alpha =1, lambda =grid)

# Generate standardized LASSO coefficients
##lambda_best
lasso.coef.standardized <- coef(out, s = lambda_best, x = x_train, y = y_train, standardize = TRUE)[1:8,]
lasso.coef.standardized #lcavol:-3.09

plot_glmnet(out, label = TRUE, s = lambda_best, xlim = c(10, -5), main="10-fold Cross-Validation")
#The coefficient for "lcavol" in the LASSO model is -3. 
#This means that a one-unit increase in the natural log of the "lcavol" 
#(luminal volume) is associated with a -3 unit decrease in the Cscore,
#holding all other predictors constant.


##lambda_1se
lasso.se.coef.standardized <- predict(out, s = lambda_1se, type = "coefficients", standardize = TRUE)[1:8,]
lasso.se.coef.standardized #only 3 variables left (svi, lcp,lpsa)

plot_glmnet(out, label = TRUE, s = lambda_1se , xlim = c(10, -5), main="10-fold Cross-Validation",add = TRUE)
#The first panel will show the predicted values versus the actual values for 
#the training set, and the second panel will show the same for the test set. 
#The abline(0,1) function adds a reference line to the plot with a slope of 1
#and an intercept of 0, which represents perfect prediction. If the points on 
#the plot are close to this reference line, it suggests that the model is doing 
#a good job of predicting the outcomes. If the points are more scattered, it 
#suggests that the model is not doing as well.




### Lasso regression with LOOCV ###
out.loocv=glmnet(omit_data[,-1],omit_data$Cscore, alpha =1, lambda =grid)

# Generate standardized LASSO coefficients
##lambda_best
lasso.coef.loocv <- coef(out.loocv, s = bestlam, x = x_train, y = y_train, standardize = TRUE)[1:8,]
lasso.coef.loocv #lcavol:-3.97

plot_glmnet(out.loocv, label = TRUE, s = bestlam, xlim = c(10, -5), main="Leave One Out Cross Validation")

##lambda_1se
lasso.se.coef.loocv<- predict(out, s = lambda_1se, type = "coefficients", standardize = TRUE)[1:8,]
lasso.se.coef.loocv #only 3 variables left (svi, lcp,lpsa)

plot_glmnet(out, label = TRUE, s = lambda_1se, xlim = c(10, -5), main="Leave-One-Out-Cross-Validation", add = TRUE)


#scatterplot => how well "lcavol" predicts "Cscore" 
ggplot(data = omit_data, aes(x = lcavol, y = Cscore)) +
  geom_point() +
  geom_smooth(method = "lm") 


model <- lm(Cscore ~ lcavol, data = omit_data)
summary(model)
#0.3138 The R-squared value measures the proportion of variance in the 
#response variable that is explained by the predictor variable. 
#A higher R-squared value indicates a stronger relationship between the
#predictor and response variables.

cor(omit_data$lcavol, omit_data$Cscore)
#0.56 => may not be representative of its true relationship with "Cscore"




## 5 Fit your best model with appropriate non-linear effects. 
## Report a comparison of performance to LASSO and your model reported under question 2. 
## Explain what you find,and indicate relevant issues or limitations of your analysis.

set.seed(1)
dim(omit_data)
train_index <- sample(nrow(omit_data), nrow(omit_data) * 2/3) 
train <- omit_data[train_index, ]
test <- omit_data[-train_index, ]

##best subset selection
bestSub <- regsubsets(Cscore ~ ., train, nvmax = ncol(train)-1)
bestSub_summary <- summary(bestSub)
bestSub_summary
plot(bestSub_summary$bic,type="b", ylab="BIC", xlab="Number of Predictors", main="Best Subset Selection")
points(which.min(bestSub_summary$bic),bestSub_summary$bic[which.min(bestSub_summary$bic)], col ="red",cex =2, pch =20) 
plot(bestSub,scale ="bic", main="Best Subset Selection")
coef(bestSub, 4)
#lcavol lbph lcp lpsa

# Fit the linear regression model (2 variables)
lm_fit <- lm(Cscore ~lcp + lpsa, data = train)
lm_pred <- predict(lm_fit, newdata = test)
lm_mse <- mean((lm_pred - test$Cscore)^2)
lm_mse 
#684 < 750 (lasso)

## Forward stepwise selection with training data
regfit.fwd=regsubsets(Cscore~. , train ,method="forward")
summary(regfit.fwd)
plot(summary(regfit.fwd,)$bic,type="b",ylab="BIC", main="Forward selection")#we select 4 variables
which.min(summary(regfit.fwd)$bic)
points(which.min(summary(regfit.fwd)$bic),summary(regfit.fwd)$bic[which.min(summary(regfit.fwd)$bic)], col ="red",cex =2, pch =20) 
plot(regfit.fwd, scale="bic",main ="Forward selection")
coef(regfit.fwd, 4)
#lcavol lbph   lcp  lpsa

lm1=lm(Cscore~lcavol+lbph+lcp+lpsa,data=train)
predlm1 = predict(lm1, newdata=test) 
mselm1 = mean((predlm1-test$Cscore)^2)
mselm1
#745.76 > 684 (indicate 4 variables is worse than 2 variables)

# Fit a GAM, plot the results, evaluate the model 
gam1=gam(Cscore~ s(lcavol,4) +s(lbph,4)+ s(lcp,4)+s(lpsa,4),data=train) 
par(mfrow=c(2,2))
plot(gam1,se=TRUE,col="purple")
summary(gam1)#only lpsa seems to have non-linear effect, lbph is not significant
predgam = predict(gam1, newdata=test) 
msegam1 = mean((predgam-test$Cscore)^2)
msegam1
#379.97

# Remove lbph and remove smooth spline function
gam2=gam(Cscore~ lcavol + lcp + s(lpsa,4),data=train) 
plot(gam2,se=TRUE,col="purple")
summary(gam2)
predgam2 = predict(gam2, newdata=test) 
msegam2 = mean((predgam2-test$Cscore)^2)
msegam2
#350

# Reduce df
gam3=gam(Cscore~ lcavol + lcp +s(lpsa,3),data=train) 
plot(gam3,se=TRUE,col="purple")
summary(gam3)
predgam3 = predict(gam3, newdata=test) 
msegam3 = mean((predgam3-test$Cscore)^2)
msegam3
#332
anova(gam1, gam2, gam3) #choose gam3
# simplification justified as expected.


#fit same model with ns
par(mfrow=c(2,2))
lm1 = lm(Cscore~lcavol+ lcp +ns(lpsa,3) ,data=train)
summary(lm1)
plot(lm1)
predlm1 = predict(lm1, newdata=test) 
mselm1 = mean((predlm1-test$Cscore)^2)
mselm1
#310, better than smoothing

#remove lcavol
lm2 = lm(Cscore~ lcp + ns(lpsa,3) ,data=train)
summary(lm2)
predlm2 = predict(lm2, newdata=test) 
mselm2 = mean((predlm2-test$Cscore)^2)
mselm2
#267

#reduce df
lm3 = lm(Cscore~ lcp +ns(lpsa,2) ,data=train)
summary(lm3)
predlm3 = predict(lm3, newdata=test) 
mselm3 = mean((predlm3-test$Cscore)^2)
mselm3 #marginally worse than msegam2
#277

anova(lm1,lm2,lm3) #choose lm3


#remove lcp
lm4 = lm(Cscore~ns(lpsa,2) ,data=train)
summary(lm4)
predlm4 = predict(lm4, newdata=test) 
mselm4 = mean((predlm4-test$Cscore)^2)
mselm4
#324

anova(lm1,lm2,lm3, lm4) #choose lm4


## polynomial transformation with lpsa variable
poly.lm <- lm(Cscore~ lcp +poly(lpsa,2,raw=TRUE),data=omit_data)
summary(poly.lm)
predlm5 = predict(poly.lm, newdata=test) 
mselm5 = mean((predlm5-test$Cscore)^2)
mselm5
#257
summary(poly.lm)
#R-squared 0.8056
summary(poly.lm)$coef
# Cscore = 17.94 +6.18lcp -20.00lpsa +8.98lpsa^2 +irreduciable error

par(mfrow=c(2,3))
plot(poly.lm)
res <- resid(poly.lm)
plot(density(res))
shapiro.test(res) # p>0.05 
# => residual is normal distribution

## polynomial 3 degree 
poly.lm2 <- lm(Cscore~ lcp +poly(lpsa,3,raw=TRUE),data=omit_data)
summary(poly.lm2)
predlm6 = predict(poly.lm2, newdata=test) 
mselm6 = mean((predlm6-test$Cscore)^2)
mselm6
#246

anova(poly.lm,poly.lm2)
# poly.lm2 is not significant 
# => polynomial 2 degree is better (simpler is better)




## Visualizing of the linear model(Question2) and non-linear model (Polynomial) 
# Draw regression curve
par(mfrow=c(2,2))
#lcp
my_mod_lcp <- lm(Cscore ~ lcp, data = omit_data)
summary(my_mod_lcp) 
plot(Cscore ~ lcp, omit_data , main="Cscore vs lcp") 
lines(sort(omit_data$lcp),      
      fitted(my_mod_lcp)[order(omit_data$lcp)],
      col = "red",
      type = "l")

#lpsa
my_mod_lpsa <- lm(Cscore ~ lpsa, data = omit_data)
summary(my_mod_lpsa) 
plot(Cscore ~ lpsa, omit_data, main="Cscore vs lpsa") 
lines(sort(omit_data$lpsa),      
      fitted(my_mod_lpsa)[order(omit_data$lpsa)],
      col = "red",
      type = "l")


# Draw polynomial regression curve
#lcp
my_mod_lcp <- lm(Cscore ~ lcp, data = omit_data)
summary(my_mod_lcp) 
plot(Cscore ~ lcp, omit_data , main="Cscore vs lcp") 
lines(sort(omit_data$lcp),      
      fitted(my_mod_lcp)[order(omit_data$lcp)],
      col = "red",
      type = "l")

#lpsa
my_mod_lpsa <- lm(Cscore ~ poly(lpsa, 2, raw=TRUE), data = omit_data)
summary(my_mod_lpsa) 
plot(Cscore ~ lpsa, omit_data, main="Cscore vs lpsa (qudratic)") 
lines(sort(omit_data$lpsa),      
      fitted(my_mod_lpsa)[order(omit_data$lpsa)],
      col = "red",
      type = "l")

#From the graph above, we can see that the model is nearly perfect.
#It fits the data points appropriately. 
#Therefore, we can use the model to make other predictions.

