#---------------------------------------------------------------------------
# 0. Initialization
#---------------------------------------------------------------------------

rm(list=ls())
#install.packages('AER')
#install.packages('Metrics')
#install.packages('jtools')
#install.packages('randomForest')
#install.packages('caret')
#install.packages('Hmisc')
#install.packages("stargazer")
library(stargazer)
library(AER)
library(Metrics)
library(ggplot2)
library(jtools)
library(Hmisc)
library(MASS)

#---------------------------------------------------------------------------
# 1. Load data
#---------------------------------------------------------------------------
df <- read.csv("C:/Users/Dell/Documents/ELJA/Master/Case studies/Code Coolblue/Final versions/broadcast_for_models.csv")

# set up response and candidate predictors
X <- c("ad_long","ad_mid","between_programs","business","commercial.broadcaster",
       "cooking","drama.crime","men","music","public.broadcaster","science","sports",
       "First.Position","Last.Position","Second.Position","televisies","laptops",
       "program_films","program_kids","program_news","program_series","program_sports"
       ,"night","morning","afternoon","prime","weekend")
y <- df$effect_prepost_window_3_capped_at_10
n <- length(y)
p <- length(X)

#---------------------------------------------------------------------------
# 2. Create necessary functions
#---------------------------------------------------------------------------
# Define R squared function
rsq <- function (x, y) cor(x, y) ^ 2

# Define function to do 10 fold Cross Validation on given data and independent variables
kfold_cv <- function(df, X){
  set.seed(123)
  #Randomly shuffle the data
  df_cv <-df[sample(nrow(df)),]
  
  #Create 10 equally size folds
  folds <- cut(seq(1,nrow(df)),breaks=10,labels=FALSE)
  
  rmse_cv <- c()
  R2_cv <- c()
  
  #Perform 10 fold cross validation
  for(i in 1:10){
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- df[testIndexes, ]
    trainData <- df[-testIndexes, ]
    
    #Fit the model
    fmla <- as.formula(paste("effect_prepost_window_3_capped_at_10 ~ ", paste(X, collapse= "+")))
    tob_fit <- tobit(fmla, data = trainData)
    
    # Make predictions
    mu <- predict(tob_fit,testData) # Latent variable mean
    sigma <- tob_fit$scale # Scale parameter
    p0 <- pnorm(mu/sigma) # Probability of non-zero observation
    lambda <- function(x) dnorm(x)/pnorm(x) # Inverse Mills ratio
    ey0 <- mu + sigma * lambda(mu/sigma) # Conditional expectation of censored y given that it is non-zero
    ey <- p0 * ey0 # Unconditional expectation 
    
    # Caluclate performance metrics
    rmse_cv[i] = rmse(testData$effect_prepost_window_3_capped_at_10, ey) # RMSE
    R2_cv[i] = rsq(testData$effect_prepost_window_3_capped_at_10, ey) # R squared
  }
  return(list(rmse = mean(rmse_cv), R2=mean(R2_cv)))
}

#---------------------------------------------------------------------------
# 3. Perform forward variable selection based on minimizing RMSE
#---------------------------------------------------------------------------

# Begin with most important one: gross_rating_point
x <- "gross_rating_point"
fit <- kfold_cv(df, x)
rmse <- c(fit$rmse, rep.int(NA_real_, p))
rsquared <- c(fit$R2, rep.int(NA_real_, p))

# Loop over candidate predictors
candidates <- X
selected <- c("gross_rating_point",rep.int(NA_character_, p))
for (j in seq_len(p)) {
  # find not yet active candidate that yields smallest RMSE
  values_rmse <- sapply(candidates, function(v) {
    x <- c(x, v)  # add current variable
    fit <- kfold_cv(df, x)                # perform estimation using k fold cv
    fit$rmse                              # return rmse
  })
  values_r2 <- sapply(candidates, function(v) {
    x <- c(x, v)  # add current variable
    fit <- kfold_cv(df, x)                # perform estimation using k fold cv
    fit$R2                                # return R squared
  })
  best <- which.min(values_rmse)
  # add that candidate to explanatory variables
  selected[j+1] <- candidates[best]
  rmse[j+1] <- values_rmse[best]
  rsquared[j+1] <- values_r2[best]
  x <- cbind(x, selected[j+1])
  candidates <- candidates[-best]
}

# Report: Figure 17
# find optimal step
sOpt <- which.min(rmse) - 1  # step count starts with 0

dfRMSE <- data.frame(Step = 0:p, RMSE = rmse)
ggplot() +
  geom_vline(aes(xintercept = sOpt), color = "darkgrey") +
  geom_line(aes(x = Step, y = RMSE), dfRMSE) +
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# construct formula for final model
variables <- head(selected, sOpt+1) # as gross_rating_point is the first selected
fmla <- as.formula(paste("effect_prepost_window_3_capped_at_10 ~ ", paste(variables, collapse= "+")))

tob_best_rmse <- tobit(fmla, data = df)
summary(tob_best_rmse)

forward_selection_rmse <- cbind(selected, rmse, rsquared)

# Report: Table 10
# Output to latex
stargazer(tob_best_rmse)
