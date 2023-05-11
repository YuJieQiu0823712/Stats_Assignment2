####Project 2####
load("prostate.Rdata")
View(prostate)
dim(prostate)
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
library(MASS)



#n=97, predictor=7, Cscore is the response


rm(summary)
## 1 Study and describe the predictor variables. Do you see any issues that are relevant for making predictions?
class(prostate$svi)
prostate$svi <- as.factor(prostate$svi)

class(prostate$svi)

# Check the structure of the data
str(prostate)

# Fit a linear regression model with all predictors
lm_all <- lm(Cscore ~ ., data = prostate)

# Print the summary of the model
summary(lm_all)


##residual plot
#To evaluate the assumptions underlying inferences with the model, we can examine the residuals of the model.
# Plot the residuals against the predicted values
par(mfrow = c(1, 1))
plot(lm_all$fitted.values, lm_all$residuals)
abline(h = 0, lty = 2)
#This will generate a plot of the residuals against the predicted values, 
#with a horizontal line at zero to help identify any patterns in the residuals. 
#If the assumptions of linearity, constant variance, and normality of residuals 
#are satisfied, we should see a random scatter of points around the horizontal line. 
#If there are any patterns in the residuals (e.g., a curved or U-shaped pattern, 
#or increasing variability with increasing predicted values), then the assumptions may be violated.

#To evaluate the effect of any influential points or outliers, we can examine
#diagnostic plots of the model.
# Generate diagnostic plots of the model
par(mfrow = c(2, 2))
plot(lm_all)



#This will generate a 2x2 grid of diagnostic plots, including a plot of the 
#residuals against the fitted values, a normal probability plot of the 
#residuals, a plot of the residuals against the leverage values, and a 
#Cook's distance plot. We can use these plots to identify any influential 
#points or outliers that may be affecting the model fit.

#If we identify any influential points or outliers, we may want to consider 
#removing them and refitting the model to see how the results are affected. 
#However, it's important to be cautious when removing data points, as this 
#can lead to biased or misleading results if not done carefully.

# Calculate the leverage and Cook's distance values for each observation
leverage <- hatvalues(lm_all)
cooks_distance <- cooks.distance(lm_all)

# Identify observations with high leverage and/or large Cook's distance values
influential_obs <- which(leverage > 2 * mean(leverage) | cooks_distance > 4 / (nrow(prostate) - length(lm_all$coefficients)))
#This code identifies any observations with leverage values that are greater 
#than twice the mean leverage, or Cook's distance values that are greater 
#than four divided by the difference between the number of observations and 
#the number of model coefficients. These cutoff values are based on standard 
#guidelines for identifying influential observations


##influencial plot
par(mfrow = c(1, 1))
avPlots(lm_all)

influencePlot(lm_all)

# Show the values of the Cscore and predictor variables for the influential observations
prostate[influential_obs, c("Cscore", "lcavol", "lweight", "age", "lbph", "svi", "lcp", "lpsa")]

###linear regression
omit_data <- prostate[-c(96),] 
dim(omit_data)
# Fit a linear regression model with all predictors
lm_reduced <- lm(Cscore ~ ., data = omit_data)
# Print the summary of the model
summary(lm_reduced)


###best subset selection
bestSub=regsubsets(Cscore~., omit_data)
bestSub.summary =summary (bestSub)
bestSub.summary
names(bestSub.summary)
bestSub.summary$rsq
which.min (bestSub.summary$bic)
plot(bestSub.summary$bic,type="b", ylab="BIC", xlab="Number of Predictors", main="Best Subset Selection")
points (2, bestSub.summary$bic[2], col ="red" ,cex =2, pch =20)
plot(bestSub,scale ="bic", main="Best Subset Selection")
coef(bestSub, 2)
#8.52, 21.15

###Forward stepwise selection
regfit.fwd =regsubsets(Cscore~., omit_data, method ="forward")
fwd_sum <- summary(regfit.fwd)
par(mfrow =c(1,1))
plot(fwd_sum$bic,type="b",ylab="BIC", xlab="Number of Predictors", main="Forwrad Stepwise Selection")
which.min(fwd_sum$bic)
points(which.min(fwd_sum$bic),fwd_sum$bic[which.min(fwd_sum$bic)], col ="red",cex =2, pch =20) 
plot(regfit.fwd, scale="bic",main ="Forward selection")
coef(regfit.fwd, 2) 
#8.52, 21.15

###Backward stepwise selection
regfit.bwd=regsubsets(Cscore~., omit_data , method ="backward")
bwd_sum <- summary (regfit.bwd)
plot(bwd_sum$bic,type="b",ylab="BIC", xlab="Number of Predictors", main="Backward Stepwise Selection")
points(2,bwd_sum$bic[2], col ="red",cex =2, pch =20) 
plot(regfit.bwd, scale="bic",main ="Backward Selection")
which.min(bwd_sum$bic)
coef(regfit.bwd, 2) 
#8.52, 21.15
coef(bestSub, 2)
coef(regfit.fwd, 2)
coef(regfit.bwd, 2)


###validation set approach

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
# the best model is the one that contains 1 variable lpsa
coef(regfit.best, 1)


#use full model
regfit.best <- regsubsets(Cscore~., data =omit_data , nvmax=7)
coef(regfit.best,1)
#-33.14853    lpas 26.88481


###10-fold cross validation
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
#8.52,21.15


###LASSO 
#To create a LASSO model, we can use the glmnet package in R. First, 
#we need to split the data into training and testing sets:



#4. do you see the evidence of over-fitting when using lasso?

#evaluate the prediction performance of the LASSO model using the test set 
#and calculate the mean squared error (MSE). 
set.seed(1)
train_index <- sample(nrow(omit_data), nrow(omit_data) * 2/3) 
train <- omit_data[train_index, ]
test <- omit_data[-train_index, ]

library(glmnet)
grid =10^seq(10,-2, length =100)
x_train <- model.matrix(Cscore ~ ., train)[,-1]
y_train <- train$Cscore
x_test <- model.matrix(Cscore ~ ., test)[,-1]
y_test <- test$Cscore

lasso_mod <- glmnet(x_train, y_train, alpha =1, lambda =grid)

lasso_fit <- cv.glmnet(x_train, y_train, alpha = 1, nfolds=10)
plot(lasso_fit,main="10 fold Cross Validation")
lambda_best <- lasso_fit$lambda.min
lambda_best
#0.73
min(lasso_fit$cvm)
#643


# Calculate the MSE on the training set
y_train_pred <- predict(lasso_mod,s=lambda_best, newx = x_train)
mse_train <- mean((y_train_pred - y_train)^2)
mse_train 
#468
# Calculate the MSE on the test set
y_test_pred <- predict(lasso_mod,s=lambda_best, newx = x_test)
mse_test <- mean((y_test_pred - y_test)^2)
mse_test
#750

#the MSE on the test set is much higher than the MSE on the training set, which 
#may suggest that the LASSO model is overfitting to the training data.
#fit to the full data
out=glmnet(omit_data[,-1],omit_data$Cscore, alpha =1, lambda =grid)

# Generate standardized LASSO coefficients
lasso.coef.standardized <- coef(out, s = lambda_best, 
                                x = x_train, y = y_train, 
                                standardize = TRUE)[1:8,]
lasso.coef.standardized

library(plotmo)
plot_glmnet(out, label = TRUE, s = lambda_best, xlim = c(10, -5), main="10-fold Cross-Validation")

#The coefficient for "lcavol" in the LASSO model is -3.097977. 
#This means that a one-unit increase in the natural log of the "lcavol" 
#(luminal volume) is associated with a -3.097977 unit decrease in the Cscore,
#holding all other predictors constant.

#The coefficient does not directly correspond to how well it can predict
#Cscore. However, it indicates the strength and direction of the relationship 
#between the predictor and the response variable in the model. 
#A larger absolute value of the coefficient suggests a stronger
#association between the predictor and the response. In this case, 
#the negative coefficient suggests that as "lcavol" increases, "Cscore" 
#tends to decrease.

coef(lasso_fit)

#test sample is only 32 so might influence the MSE
#overlearning 

# predict Cscore on the training set
y_train_pred <- predict(lasso_mod, s = lambda_best, newx = x_train)
# calculate RMSE on the training set
rmse_train <- sqrt(mean((y_train_pred - y_train)^2))
# calculate R-squared on the training set
rsq_train <- 1 - sum((y_train - y_train_pred)^2) / sum((y_train - mean(y_train))^2)

# predict Cscore on the test set
y_test_pred <- predict(lasso_mod, s = lambda_best, newx = x_test)
# calculate RMSE on the test set
rmse_test <- sqrt(mean((y_test_pred - y_test)^2))
# calculate R-squared on the test set
rsq_test <- 1 - sum((y_test - y_test_pred)^2) / sum((y_test - mean(y_test))^2)

# print the results
cat("RMSE on training set:", rmse_train, "\n")
cat("RMSE on test set:", rmse_test, "\n")
cat("R-squared on training set:", rsq_train, "\n")
cat("R-squared on test set:", rsq_test, "\n")
#60.9% of the variation in the Cscore values in the test set can be explained by the predictors in the model.

#the MSE on the test set is much higher than the MSE on the training set, which 
#may suggest that the LASSO model is overfitting to the training data.

#To further evaluate if overfitting is an issue, you can also plot the predicted 
#values versus the actual values for both the training and test sets.
#If the plot for the test set shows a more scattered pattern than the plot 
#for the training set, this may also indicate overfitting. 
par(mfrow = c(1,2))
plot(y_train, y_train_pred, main = "Training set")
abline(0,1)
plot(y_test, y_test_pred, main = "Test set")
abline(0,1)

#The first panel will show the predicted values versus the actual values for 
#the training set, and the second panel will show the same for the test set. 
#The abline(0,1) function adds a reference line to the plot with a slope of 1
#and an intercept of 0, which represents perfect prediction. If the points on 
#the plot are close to this reference line, it suggests that the model is doing 
#a good job of predicting the outcomes. If the points are more scattered, it 
#suggests that the model is not doing as well.


##### Lasso regression with LOOCV ####
set.seed(1)
train_index <- sample(nrow(omit_data), nrow(omit_data) * 2/3) 
train <- omit_data[train_index, ]
test <- omit_data[-train_index, ]

library(glmnet)
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
# bestlam = 0.55 results in the smallest cross-validation error 651


# Prediction and evaluation on test data
lasso.pred=predict(lasso_mod_loocv,s=bestlam,newx=x_test)
mean((lasso.pred-y_test)^2)
# the test MSE = 757.24

lasso.pred.train=predict(lasso_mod_loocv,s=bestlam,newx=x_train)
mean((lasso.pred.train-y_train)^2)
# the training MSE = 465.29

out.loocv=glmnet(omit_data[,-1],omit_data$Cscore, alpha =1, lambda =grid)

# Generate standardized LASSO coefficients
lasso.coef.loocv <- coef(out.loocv, s = bestlam, 
                                x = x_train, y = y_train, 
                                standardize = TRUE)[1:8,]
lasso.coef.loocv

library(plotmo)
plot_glmnet(out, label = TRUE, s = bestlam, xlim = c(10, -5), main="Leave One Out Cross Validation")

#lcavol:-3.969890

library(ggplot2)
ggplot(data = omit_data, aes(x = lcavol, y = Cscore)) +
  geom_point() +
  geom_smooth(method = "lm") 
#This code will create a scatterplot with "lcavol" on the x-axis and "Cscore"
#on the y-axis, and add a linear regression line to show the overall trend in
#the data. You can use this plot to see how well "lcavol" predicts "Cscore" 
#visually. If the points on the plot are tightly clustered around the regression
#line, it indicates that "lcavol" is a good predictor of "Cscore".

model <- lm(Cscore ~ lcavol, data = omit_data)
summary(model)
#0.3138 The R-squared value measures the proportion of variance in the 
#response variable that is explained by the predictor variable. 
#A higher R-squared value indicates a stronger relationship between the
#predictor and response variables.

cor(omit_data$lcavol, omit_data$Cscore)


###5. fit best non-linear model


set.seed(1)
dim(omit_data)
train_index <- sample(nrow(omit_data), nrow(omit_data) * 2/3) 
train <- omit_data[train_index, ]
test <- omit_data[-train_index, ]
par(mfrow=c(1,1))
pairs(omit_data)

#A. Split the data, and forward stepwise selection

# Fit the linear regression model
lm_fit <- lm(Cscore ~lcp + lpsa, data = train)
lm_pred <- predict(lm_fit, newdata = test)
lm_mse <- mean((lm_pred - test$Cscore)^2)
lm_mse 
#684 < 750 (lasso)

regfit.fwd=regsubsets(Cscore~. , train ,method="forward")
summary(regfit.fwd)
plot(summary(regfit.fwd,)$bic,type="b",ylab="BIC", main="Forward selection")#we select 4 variables
which.min(summary(regfit.fwd)$bic)
points(which.min(summary(regfit.fwd)$bic),summary(regfit.fwd)$bic[which.min(summary(regfit.fwd)$bic)], col ="red",cex =2, pch =20) 
plot(regfit.fwd, scale="bic",main ="Forward selection")
coef(regfit.fwd, 4) 
lm1=lm(Cscore~lcavol+lbph+lcp+lpsa,data=train)
predlm1 = predict(lm1, newdata=test) 
mselm1 = mean((predlm1-test$Cscore)^2)
mselm1
#745.76 > 684(higher than 2 variables?)


#B-D. Fit a GAM, plot the results, evaluate the model. Are there non-linear effects? 
gam1=gam(Cscore~ s(lcavol,4) +s(lbph,4)+ s(lcp,4)+s(lpsa,4),data=train) 
#ignore the gam "non-list contrasts" warning; it's a (harmless) bug
par(mfrow=c(2,2))
plot(gam1,se=TRUE,col="purple")
summary(gam1)#only lpsa seems to have non-linear effect, bph not significant
predgam = predict(gam1, newdata=test) 
msegam1 = mean((predgam-test$Cscore)^2)
msegam1
#379.97

#remove bph and remove s function
gam2=gam(Cscore~ lcavol + lcp +s(lpsa,4),data=train) 
plot(gam2,se=TRUE,col="purple")
summary(gam2)
predgam2 = predict(gam2, newdata=test) 
msegam2 = mean((predgam2-test$Cscore)^2)
msegam2
#350

#reduce df
gam3=gam(Cscore~ lcavol + lcp +s(lpsa,3),data=train) 
plot(gam3,se=TRUE,col="purple")
summary(gam3)
predgam3 = predict(gam3, newdata=test) 
msegam3 = mean((predgam3-test$Cscore)^2)
msegam3
#332
anova(gam1,gam2, gam3)#simplification justified as expected.

#fit same model with ns
lm1 = lm(Cscore~lcavol+ lcp +ns(lpsa,3) ,data=train)
summary(lm1)
plot(lm1)
predlm1 = predict(lm1, newdata=test) 
mselm1 = mean((predlm1-test$Cscore)^2)
mselm1
#310, better than smoothing

#remove lcavol
lm2 = lm(Cscore~ lcp +ns(lpsa,3) ,data=train)
summary(lm2)
predlm2 = predict(lm2, newdata=test) 
mselm2 = mean((predlm3-test$Cscore)^2)
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

anova(lm1,lm2,lm3, lm4) #lm4 significant different, ns2 better

poly.lm <- lm(Cscore~ lcp +poly(lpsa,2,raw=TRUE),data=omit_data)
summary(poly.lm)
predlm5 = predict(poly.lm, newdata=test) 
mselm5 = mean((predlm5-test$Cscore)^2)
mselm5
#257
summary(poly.lm)$coef
plot(poly.lm)
######best model
#easy and simple, so no need to compare with ns

poly.lm2 <- lm(Cscore~ lcp +poly(lpsa,3,raw=TRUE),data=omit_data)
summary(poly.lm2)
predlm6 = predict(poly.lm2, newdata=test) 
mselm6 = mean((predlm6-test$Cscore)^2)
mselm6
#246

anova(poly.lm,poly.lm2)#poly.lm2 not significant, simpler model better


