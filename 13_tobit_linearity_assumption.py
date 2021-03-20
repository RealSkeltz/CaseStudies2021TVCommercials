# Import packages
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import random
import statistics

# Import data
df = pd.read_csv("XXX/XXX/broadcast_for_models.csv")

# Define the cross validation function
def cross_val(df, folds, model, seed):
    # First step: shuffle data randomly
    n = len(df)
    random.seed(seed)
    var = random.sample(range(0, n), n)
    dff = df.iloc[var]
    
    # Second step: perform k-fold cross validation
    RMSE_list=[]
    
    cycle=0
    for i in range(int(n/folds), n, int(n/folds)):
        # Split data into test and train
        test = dff.iloc[cycle:i]
        train = dff.drop(range(cycle,i))
        test = test.reset_index(drop=True)
        train = train.reset_index(drop=True)
        
        # Define variables for X and y, based on output of Tobit
        Xtest = test[['gross_rating_point','commercial broadcaster','between_programs','prime', 
       'science', 'drama/crime','First Position', 'public broadcaster','Second Position',
       'sports', 'ad_long', 'ad_mid', 'laptops','televisies','cooking', 'weekend', 'program_news']]
        ytest = test['effect_prepost_window_3_capped_at_10']
        Xtrain = train[['gross_rating_point','commercial broadcaster','between_programs','prime', 
       'science', 'drama/crime','First Position', 'public broadcaster','Second Position',
       'sports', 'ad_long', 'ad_mid', 'laptops','televisies','cooking', 'weekend', 'program_news']]
        ytrain = train['effect_prepost_window_3_capped_at_10']

        if model == 'AB':
            model_ab = AdaBoostRegressor(n_estimators=100,
                                         base_estimator=DecisionTreeRegressor(),
                                         random_state=seed)
            model_ab.fit(Xtrain,ytrain)
            pred_ab = model_ab.predict(Xtest)
            rms_ab = mean_squared_error(ytest, pred_ab, squared=False)
            RMSE_list.append(rms_ab)
            
        if model == 'OLS':
            Xtrain=sm.add_constant(Xtrain)
            Xtest=sm.add_constant(Xtest)
            model_ols = sm.OLS(ytrain,Xtrain)
            results = model_ols.fit()
            pred_ols = results.predict(Xtest)
            rms_ols = mean_squared_error(ytest, pred_ols, squared=False)
            RMSE_list.append(rms_ols)
            
        cycle+=int(n/folds)
        
    mean_RMSE = statistics.mean(RMSE_list)
    return mean_RMSE

# Report: Table 11
print(cross_val(df, 10, 'OLS', 782))
print(cross_val(df, 10, 'AB', 782))