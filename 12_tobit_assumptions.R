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
X <- c('gross_rating_point','commercial.broadcaster','weekend' ,'between_programs','prime', 
       'science','program_news', 'drama.crime','First.Position', 'public.broadcaster','Second.Position',
       'sports','program_series', 'ad_long','program_kids', 'ad_mid', 'laptops','televisies','cooking','morning')
y <- df$effect_prepost_window_3_capped_at_10
n <- length(y)
p <- length(X)

#---------------------------------------------------------------------------
# 2. Create necessary functions
#---------------------------------------------------------------------------
# Define R squared function
rsq <- function (x, y) cor(x, y) ^ 2


#Fit the model
fmla <- as.formula(paste("effect_prepost_window_3_capped_at_10 ~ ", paste(X, collapse= "+")))
tob_fit <- tobit(fmla, data = df)

# Make predictions
mu <- predict(tob_fit,df) # Latent variable mean
sigma <- tob_fit$scale # Scale parameter
p0 <- pnorm(mu/sigma) # Probability of non-zero observation
lambda <- function(x) dnorm(x)/pnorm(x) # Inverse Mills ratio
ey0 <- mu + sigma * lambda(mu/sigma) # Conditional expectation of censored y given that it is non-zero
ey <- p0 * ey0 # Unconditional expectation 

#---------------------------------------------------------------------------
# 3a. Heteroskedasticity
#---------------------------------------------------------------------------
# Report: Figure 18
res <- y-ey
vec <- df[['gross_rating_point']]
y <- df[['effect_prepost_window_3_capped_at_10']]
fig1 = plot(y, res, main="Heteroskedasticity eye-test",
            xlab="Dependent variable", ylab="Residuals", pch=19)

#---------------------------------------------------------------------------
# 3a. Normality
#---------------------------------------------------------------------------
# Report: Figure 19
hist(res, breaks=100)
