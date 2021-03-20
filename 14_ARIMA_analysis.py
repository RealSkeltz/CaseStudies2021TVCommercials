# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMAResults
import time
from tqdm import tqdm
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller

# Import data
data_ext = pd.read_csv("XXX/XXX/dff_with_features.csv")

# Set date time index
data_ext['index'] = pd.to_datetime(data_ext['index'], format="%Y-%m-%d %H:%M:%S") 
data_ext = data_ext.set_index('index')

###############################################################################
###############################################################################
####### Determine ARIMA parameters ############################################
###############################################################################
###############################################################################

# Create new data frame for determining parameters ARIMA
dff_params = data_ext[0:]

# Function to plot rolling mean/stdev and test for stationarity
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=181).mean()
    rolstd = pd.Series(timeseries).rolling(window=181).std()

    #Plot rolling statistics:
    plt.figure(figsize=(30, 8))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput) 

# Report: Figure 20    
# Test stationarity for current data set    
test_stationarity(dff_params.visits_index)

# Report: Figure 21
# Test stationarity for differenced data sets
dff_params['first_difference'] = dff_params.visits_index - dff_params.visits_index.shift(1)
test_stationarity(dff_params.first_difference.dropna(inplace=False))

# Report: Figure 22
# Plot autocorrelation and partial autocorrelation of visits_index
fig, ax = plt.subplots(1,2,figsize=(12, 4))
sm.graphics.tsa.plot_acf(dff_params.visits_index.dropna(inplace=False), lags=20, ax=ax[0])
sm.graphics.tsa.plot_pacf(dff_params.visits_index.dropna(inplace=False), lags=20, ax=ax[1])

# Report: Figure 23
# Plot autocorrelation and partial autocorrelation of first difference
fig, ax = plt.subplots(1,2,figsize=(12, 4))
sm.graphics.tsa.plot_acf(dff_params.first_difference.dropna(inplace=False), lags=10, ax=ax[0])
sm.graphics.tsa.plot_pacf(dff_params.first_difference.dropna(inplace=False), lags=10, ax=ax[1])

# Create new data frame to fit ARIMA model on (otherwise datetime index of data_ext is changed as well)
dff = data_ext[0:]
dff.index = dff.index.to_period('T') # Transform datetime index to period index

# Create model for initial training sample
endog = dff['visits_index'] # Define endogenous variable
n_obs = len(endog)
train_size = int(n_obs * 0.9)
train, test = endog.iloc[:train_size], endog.iloc[train_size:]

# Grid search
# Do auto search on parameters
# fitting a stepwise model:
    # For d=0
stepwise_fit = pm.auto_arima(train, start_p=1, start_q=1, max_p=8, max_q=3, d=0,
                             seasonal=False, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise

stepwise_fit.summary()

    # For d=1
stepwise_fit = pm.auto_arima(train, start_p=1, start_q=1, max_p=8, max_q=3, d=1,
                             seasonal=False, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise

stepwise_fit.summary()

optimal_params = (7,1,2) # Fill in optimal parameters from grid search

###############################################################################
###############################################################################
####### ARIMA baseline model (no exogenous) ###################################
###############################################################################
###############################################################################

# Fit model
mod = sm.tsa.SARIMAX(endog = train, order = optimal_params)
start=time.process_time()
model = mod.fit()
end=time.process_time()
print('Model fitting time: {} seconds'.format(end-start))

# Save model to disk
filename = 'model_new_' + str(optimal_params) + '.pkl'
model.save(filename)

# load the model from disk
filename = 'model_new_(7, 1, 2).pkl' # Insert here the desired file name
loaded_model = ARIMAResults.load(filename)

# Forecast
forecasts = []   # Create empty list to store forecast results
forecasts.append(loaded_model.forecast()[0])  # Save initial forecast

# Step through the rest of the sample
for t in tqdm(range(train_size, n_obs-1)):
    # Update the results by appending the next observation
    updated_endog = endog.iloc[t:t+1]
    loaded_model = loaded_model.extend(endog = updated_endog)
    obs = test[t-train_size]
    output = loaded_model.forecast()
    yhat = output[0]
    # Save the new set of forecasts
    forecasts.append(yhat)

# some data manipulation to get the output in desired format...
forecasts = pd.DataFrame(forecasts).set_index(test.index)

# Calculate MAPE - first some data manipulation to get data in right format
forecasts_vals = forecasts.reset_index()[0]
forecasts_array = np.array(forecasts_vals)

test_mape = test.replace(0,0.00001) # MAPE cannot deal with 0
test_vals = test_mape.reset_index()['visits_index']
test_array = np.array(test_vals)

mape = np.mean(np.abs(forecasts_array - test_array) / np.abs(test_array)) * 100
print('MAPE: %.3f' % mape)

mae = np.mean(np.abs(forecasts_array - test_array)) 
print('MAE: %.3f' % mae)

# Calculate R2
r2 = r2_score(test, forecasts)
print('R2: %.3f' %r2)

# Evaluate forecasts by means of root mean squared error
rmse = sqrt(mean_squared_error(test, forecasts))
print('Test RMSE: %.3f' % rmse)

# Evaluate only forecasts for which an advertisement took place in 1 minute pre window
test_df = dff.iloc[train_size:] # Obtain test set with extra variables
temp_ind = test_df[test_df['ad_1_ago']>0].index # Obtain indeces of test set for which the observations falls in post-ad period
test_ad = test[test.index.isin(temp_ind)] # Filter test results on post-ad period
fc_ad = forecasts[forecasts.index.isin(temp_ind)] # Filter forecast results on post-ad period
rmse_ad = sqrt(mean_squared_error(test_ad, fc_ad))
print('Test RMSE of post-ad (1 min) period: %.3f' % rmse_ad)

# And forecasts for which an advertisement took place in 3 minute pre window
test_df = dff.iloc[train_size:] # Obtain test set with extra variables
temp_ind = test_df[(test_df['ad_1_ago']>0) | (test_df['ad_2_ago']>0) | (test_df['ad_3_ago']>0)].index # Obtain indeces of test set for which the observations falls in post-ad period
test_ad = test[test.index.isin(temp_ind)] # Filter test results on post-ad period
fc_ad = forecasts[forecasts.index.isin(temp_ind)] # Filter forecast results on post-ad period
rmse_ad_3 = sqrt(mean_squared_error(test_ad, fc_ad))
print('Test RMSE of post-ad (3 min) period: %.3f' % rmse_ad_3)

# Report: Table 12 (column baseline)
loaded_model.params
loaded_model.pvalues

###############################################################################
###############################################################################
####### ARIMA forward feature selection of non-interaction features ###########
###############################################################################
###############################################################################

# Feature groups (no interactions)
exogenous_vars = {1:['gpr_ad_1_ago','gpr_ad_1_ago_squared'],
                  2:['first_position_ad_1_ago','second_position_ad_1_ago','last_position_ad_1_ago'],
                  3:['morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago','night_ad_1_ago'], #late_evening = default
                  4:['commercial_broadcaster_ad_1_ago', 'public_broadcaster_ad_1_ago','sports_ad_1_ago', 'cooking_ad_1_ago'],#default = other channel groups
                  5:['wasmachines_ad_1_ago','televisies_ad_1_ago'], #default = laptops
                  6:['ad_long_ad_1_ago', 'ad_mid_ad_1_ago'],#default = ad_short
                  7:['weekend_ad_1_ago'], #default = weekday
                  8:['program_news_ad_1_ago'], #default = other programs
                  9: ['between_programs_ad_1_ago'] #default = within 1 program
                  }


rmse_output = pd.DataFrame({'metric':['rmse_exog', 'rmse_exog_ad','rmse_exog_ad_3','MAE','MAPE', 'R2']})
rmse_output = rmse_output.set_index('metric')

model_output = pd.DataFrame({'vars': ['gpr_ad_1_ago','gpr_ad_1_ago_squared', 
                                      'morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago','night_ad_1_ago',
                                      'commercial_broadcaster_ad_1_ago', 'public_broadcaster_ad_1_ago','sports_ad_1_ago', 'cooking_ad_1_ago',
                                      'wasmachines_ad_1_ago','televisies_ad_1_ago',
                                      'ad_long_ad_1_ago', 'ad_mid_ad_1_ago',
                                      'weekend_ad_1_ago',
                                      'program_news_ad_1_ago',
                                      'between_programs_ad_1_ago',
                                      'first_position_ad_1_ago','second_position_ad_1_ago','last_position_ad_1_ago',
                                      'ar.L1', 'ar.L2', 'ar.L3', 'ar.L4', 'ar.L5','ar.L6', 'ar.L7', 
                                      'ma.L1', 'ma.L2', 
                                      'sigma2']})
model_output = model_output.set_index('vars')


exogenous_vars_left = exogenous_vars
vars_selected = []
new_rmse = 3
old_rmse = 5
round_nr = 1

while (new_rmse < old_rmse + 0.0003):
    rmse_scores = []
    round_sub_nr = 1
    for var_group in exogenous_vars_left.keys(): 
        print(round_nr,round_sub_nr)
        #print(var_group)
        #vars_temp = np.concatenate((vars_selected,exogenous_vars_left[var_group]),axis=0)
        vars_temp = vars_selected + exogenous_vars_left[var_group]
        print(vars_temp)
        
        exog = dff[vars_temp]
        # model fitten met exog

        train_exog, test_exog = exog.iloc[:train_size], exog.iloc[train_size:]

        # Fit model
        mod = sm.tsa.SARIMAX(endog = train, exog = train_exog, order = optimal_params) #NOTE: Max iter hier nu toegevoegd
        start=time.process_time()
        model = mod.fit()
        end=time.process_time()
        print('Model fitting time: {} seconds'.format(end-start))
        
        # Save model to disk
        #filename = 'model_exog_' + str(round_nr) + '.' + str(round_sub_nr) + '_'  + str(optimal_params) + '.pkl'
        #model.save(filename)
            
        # Use the next line when fitting the models in the for loop
        loaded_model_exog = model
        
        # Use the next lines when model fitting has already been performed and you want to load the models from disk
        #filename = 'model_exog_' + str(round_nr) + '_'  + str(optimal_params) + '.pkl' # Insert here the desired file name
        #loaded_model_exog = ARIMAResults.load(filename)
        
        # Forecast
        forecasts_exog = []   # Create empty list to store forecast results
        forecasts_exog.append(loaded_model_exog.forecast(exog=exog.iloc[train_size-1:train_size])[0]) 
          # Save initial forecast
        
        # Step through the rest of the sample
        for t in tqdm(range(train_size, n_obs-1)):
            # Update the results by appending the next observation
            updated_endog = endog.iloc[t:t+1]
            updated_exog = exog.iloc[t:t+1]
            loaded_model_exog = loaded_model_exog.extend(endog = updated_endog, exog = updated_exog)
            obs = test[t-train_size]
            output = loaded_model_exog.forecast(exog = updated_exog)
            yhat = output[0]
            # Save the new set of forecasts
            forecasts_exog.append(yhat)
            #print('predicted=%f, expected=%f' % (yhat, obs))
        
        # some data manipulation to get the output in desired format...
        forecasts_exog = pd.DataFrame(forecasts_exog).set_index(test.index)
        
        # Calculate MAPE - first some data manipulation to get data in right format
        forecasts_exog_vals = forecasts_exog.reset_index()[0]
        forecasts_exog_array = np.array(forecasts_exog_vals)
        
        test_mape = test.replace(0,0.00001)
        test_vals = test_mape.reset_index()['visits_index']
        test_array = np.array(test_vals)
    
        mape_exog = np.mean(np.abs(forecasts_exog_array - test_array) / np.abs(test_array)) * 100
        mae_exog = np.mean(np.abs(forecasts_exog_array - test_array)) 
        
        # Calculate R2
        r2_exog = r2_score(test, forecasts_exog)
        
        # Root mean squared error
        rmse_exog = sqrt(mean_squared_error(test, forecasts_exog))
        
        # 1 minute post window
        test_df = dff.iloc[train_size:] # Obtain test set with extra variables
        temp_ind = test_df[test_df['ad_1_ago']>0].index # Obtain indeces of test set for which the observations falls in post-ad period
        test_ad = test[test.index.isin(temp_ind)] # Filter test results on post-ad period
        fc_exog_ad = forecasts_exog[forecasts_exog.index.isin(temp_ind)] # Filter forecast results on post-ad period
        rmse_exog_ad = sqrt(mean_squared_error(test_ad, fc_exog_ad))
        
        # 3 minute post window
        temp_ind = test_df[(test_df['ad_1_ago']>0) | (test_df['ad_2_ago']>0) | (test_df['ad_3_ago']>0)].index # Obtain indeces of test set for which the observations falls in post-ad period
        test_ad = test[test.index.isin(temp_ind)] # Filter test results on post-ad period
        fc_exog_ad = forecasts_exog[forecasts_exog.index.isin(temp_ind)] # Filter forecast results on post-ad period
        rmse_exog_ad_3 = sqrt(mean_squared_error(test_ad, fc_exog_ad))
    
        # Save model metrics
        metrics_list = [rmse_exog, rmse_exog_ad, rmse_exog_ad_3,mae_exog,mape_exog,r2_exog]
        
        rmse_output[str(round_nr) + '.' + str(round_sub_nr)] = metrics_list
        
        # Save model coefficients
        coefs = pd.DataFrame(loaded_model_exog.params, columns=['Round'+str(round_nr)+'.'+str(round_sub_nr)+'_coef']) 
        pvals = pd.DataFrame(loaded_model_exog.pvalues, columns=['Round'+str(round_nr)+'.'+str(round_sub_nr)+'_pval']) 
        
        model_output = model_output.join(coefs)
        model_output = model_output.join(pvals)
        
        # Save rmse_exog_ad_3 to select the best from the round
        rmse_score = rmse_exog_ad_3
        rmse_scores.append(rmse_score)
        round_sub_nr = round_sub_nr + 1
        print('Test RMSE of (3 min) post-ad period: %.6f' % rmse_exog_ad_3)
        
    round_nr = round_nr + 1
    
    old_rmse = new_rmse
    new_rmse = min(rmse_scores)
    index_min_rmse = np.argmin(rmse_scores)
    best_var_nr = list(exogenous_vars_left.keys())[index_min_rmse]
    vars_selected = vars_selected + exogenous_vars_left[best_var_nr]
    del exogenous_vars_left[best_var_nr]

# Report: Table 22    
model_output.to_csv("XXX/XXX/ARIMA_feat_selec_model.csv")
rmse_output.to_csv("XXX/XXX/ARIMA_feat_selec_metrics.csv")


###############################################################################
###############################################################################
####### ARIMA forward feature selection of interaction features ###########
###############################################################################
###############################################################################

# Feature groups
interaction_vars = {1:['int_televisies_gpr_ad_1_ago', 'int_wasmachines_gpr_ad_1_ago'], #product * gpr
                  2:['int_televisies_between_programs_ad_1_ago','int_wasmachines_between_programs_ad_1_ago'], #product*between programs
                  3:['int_televisies_morning_ad_1_ago', 'int_wasmachines_morning_ad_1_ago',
                     'int_televisies_afternoon_ad_1_ago','int_wasmachines_afternoon_ad_1_ago', 
                     'int_televisies_prime_ad_1_ago','int_wasmachines_prime_ad_1_ago'], #product * part of day
                  4:['int_televisies_program_news_ad_1_ago','int_wasmachines_program_news_ad_1_ago'], #product * program_news
                  5:['int_news_morning_ad_1_ago','int_news_afternoon_ad_1_ago','int_news_prime_ad_1_ago'], #program_news * part of day
                  6:['int_between_programs_morning_ad_1_ago','int_between_programs_afternoon_ad_1_ago','int_between_programs_prime_ad_1_ago'] #between programs * part of day
                  }


rmse_output = pd.DataFrame({'metric':['rmse_exog', 'rmse_exog_ad','rmse_exog_ad_3','MAE','MAPE', 'R2']})
rmse_output = rmse_output.set_index('metric')

model_output = pd.DataFrame({'vars': [# Selected non-interaction vars
                                      'gpr_ad_1_ago','gpr_ad_1_ago_squared', 
                                      'morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago','night_ad_1_ago',
                                      'wasmachines_ad_1_ago','televisies_ad_1_ago',
                                      'program_news_ad_1_ago',
                                      'between_programs_ad_1_ago',
                                      # Potential interaction vars
                                      # 1
                                      'int_televisies_gpr_ad_1_ago', 'int_wasmachines_gpr_ad_1_ago',
                                      # 2
                                      'int_televisies_between_programs_ad_1_ago','int_wasmachines_between_programs_ad_1_ago',
                                      # 3
                                      'int_televisies_morning_ad_1_ago', 'int_wasmachines_morning_ad_1_ago',
                                      'int_televisies_afternoon_ad_1_ago','int_wasmachines_afternoon_ad_1_ago', 
                                      'int_televisies_prime_ad_1_ago','int_wasmachines_prime_ad_1_ago',
                                      # 4
                                      'int_televisies_program_news_ad_1_ago','int_wasmachines_program_news_ad_1_ago',
                                      # 5
                                      'int_news_morning_ad_1_ago','int_news_afternoon_ad_1_ago','int_news_prime_ad_1_ago', 
                                      # 6
                                      'int_between_programs_morning_ad_1_ago','int_between_programs_afternoon_ad_1_ago','int_between_programs_prime_ad_1_ago',
                                      # Baseline ARIMA vars
                                      'ar.L1', 'ar.L2', 'ar.L3', 'ar.L4', 'ar.L5','ar.L6', 'ar.L7', 
                                      'ma.L1', 'ma.L2', 
                                      'sigma2']})
model_output = model_output.set_index('vars')


exogenous_vars_left = interaction_vars
vars_selected = [# From non-interaction feature selection
                 'gpr_ad_1_ago','gpr_ad_1_ago_squared', 
                 'morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago','night_ad_1_ago',
                 'wasmachines_ad_1_ago','televisies_ad_1_ago',
                 'program_news_ad_1_ago',
                 'between_programs_ad_1_ago']

new_rmse = 3
old_rmse = 5
round_nr = 1

while (new_rmse < old_rmse + 0.0003):
    rmse_scores = []
    round_sub_nr = 1
    for var_group in exogenous_vars_left.keys(): 
        print(round_nr,round_sub_nr)
        #print(var_group)
        #vars_temp = np.concatenate((vars_selected,exogenous_vars_left[var_group]),axis=0)
        vars_temp = vars_selected + exogenous_vars_left[var_group]
        print(vars_temp)
        
        exog = dff[vars_temp]
        # model fitten met exog

        train_exog, test_exog = exog.iloc[:train_size], exog.iloc[train_size:]

        # Fit model
        mod = sm.tsa.SARIMAX(endog = train, exog = train_exog, order = optimal_params)
        start=time.process_time()
        model = mod.fit()
        end=time.process_time()
        print('Model fitting time: {} seconds'.format(end-start))
        
        # Save model to disk
        #filename = 'model_exog_' + str(round_nr) + '.' + str(round_sub_nr) + '_'  + str(optimal_params) + '.pkl'
        #model.save(filename)
            
        # Use the next line when fitting the models in the for loop
        loaded_model_exog = model
        
        # Use the next lines when model fitting has already been performed and you want to load the models from disk
        #filename = 'model_exog_' + str(round_nr) + '_'  + str(optimal_params) + '.pkl' # Insert here the desired file name
        #loaded_model_exog = ARIMAResults.load(filename)
        
        # Forecast
        forecasts_exog = []   # Create empty list to store forecast results
        forecasts_exog.append(loaded_model_exog.forecast(exog=exog.iloc[train_size-1:train_size])[0]) 
          # Save initial forecast
        
        # Step through the rest of the sample
        for t in tqdm(range(train_size, n_obs-1)):
            # Update the results by appending the next observation
            updated_endog = endog.iloc[t:t+1]
            updated_exog = exog.iloc[t:t+1]
            loaded_model_exog = loaded_model_exog.extend(endog = updated_endog, exog = updated_exog)
            obs = test[t-train_size]
            output = loaded_model_exog.forecast(exog = updated_exog)
            yhat = output[0]
            # Save the new set of forecasts
            forecasts_exog.append(yhat)
            #print('predicted=%f, expected=%f' % (yhat, obs))
        
        # some data manipulation to get the output in desired format...
        forecasts_exog = pd.DataFrame(forecasts_exog).set_index(test.index)
        
        # Calculate MAPE - first some data manipulation to get data in right format
        forecasts_exog_vals = forecasts_exog.reset_index()[0]
        forecasts_exog_array = np.array(forecasts_exog_vals)
        
        test_mape = test.replace(0,0.00001)
        test_vals = test_mape.reset_index()['visits_index']
        test_array = np.array(test_vals)
    
        mape_exog = np.mean(np.abs(forecasts_exog_array - test_array) / np.abs(test_array)) * 100
        
        mae_exog = np.mean(np.abs(forecasts_exog_array - test_array)) 
        
        # Calculate R2
        r2_exog = r2_score(test, forecasts_exog)
        
        # Root mean squared error
        rmse_exog = sqrt(mean_squared_error(test, forecasts_exog))
        
        # 1 minute post window
        test_df = dff.iloc[train_size:] # Obtain test set with extra variables
        temp_ind = test_df[test_df['ad_1_ago']>0].index # Obtain indeces of test set for which the observations falls in post-ad period
        test_ad = test[test.index.isin(temp_ind)] # Filter test results on post-ad period
        fc_exog_ad = forecasts_exog[forecasts_exog.index.isin(temp_ind)] # Filter forecast results on post-ad period
        rmse_exog_ad = sqrt(mean_squared_error(test_ad, fc_exog_ad))
        
        # 3 minute post window
        temp_ind = test_df[(test_df['ad_1_ago']>0) | (test_df['ad_2_ago']>0) | (test_df['ad_3_ago']>0)].index # Obtain indeces of test set for which the observations falls in post-ad period
        test_ad = test[test.index.isin(temp_ind)] # Filter test results on post-ad period
        fc_exog_ad = forecasts_exog[forecasts_exog.index.isin(temp_ind)] # Filter forecast results on post-ad period
        rmse_exog_ad_3 = sqrt(mean_squared_error(test_ad, fc_exog_ad))
    
        # Save model metrics
        metrics_list = [rmse_exog, rmse_exog_ad, rmse_exog_ad_3,mae_exog,mape_exog,r2_exog]
        
        rmse_output[str(round_nr) + '.' + str(round_sub_nr)] = metrics_list
        
        # Save model coefficients
        coefs = pd.DataFrame(loaded_model_exog.params, columns=['Round'+str(round_nr)+'.'+str(round_sub_nr)+'_coef']) 
        pvals = pd.DataFrame(loaded_model_exog.pvalues, columns=['Round'+str(round_nr)+'.'+str(round_sub_nr)+'_pval']) 
        
        model_output = model_output.join(coefs)
        model_output = model_output.join(pvals)
        
        # Save rmse_exog_ad_3 to select the best from the round
        rmse_score = rmse_exog_ad_3
        rmse_scores.append(rmse_score)
        round_sub_nr = round_sub_nr + 1
        print('Test RMSE of (3 min) post-ad period: %.6f' % rmse_exog_ad_3)
        
    round_nr = round_nr + 1
    
    old_rmse = new_rmse
    new_rmse = min(rmse_scores)
    index_min_rmse = np.argmin(rmse_scores)
    best_var_nr = list(exogenous_vars_left.keys())[index_min_rmse]
    vars_selected = vars_selected + exogenous_vars_left[best_var_nr]
    del exogenous_vars_left[best_var_nr]

# Report: Table 23       
model_output.to_csv("XXX/XXX/ARIMA_interaction_selec_model.csv")
rmse_output.to_csv("XXX/XXX/ARIMA_interaction_selec_metrics.csv")

###############################################################################
###############################################################################
####### ARIMA models shown in report ##########################################
###############################################################################
###############################################################################

exogenous_variables ={0: #model 1 in report
                         ['gpr_ad_1_ago', 'gpr_ad_1_ago_squared'], 
                      1: #model 2 in report
                          ['gpr_ad_1_ago', 'gpr_ad_1_ago_squared',
                          'morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago','night_ad_1_ago',
                          'wasmachines_ad_1_ago','televisies_ad_1_ago',
                          'program_news_ad_1_ago',
                          'between_programs_ad_1_ago'], 
                      2: #model 3 in report
                          ['gpr_ad_1_ago', 'gpr_ad_1_ago_squared', 
                          'morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago','night_ad_1_ago',
                          'wasmachines_ad_1_ago','televisies_ad_1_ago',
                          'program_news_ad_1_ago',
                          'between_programs_ad_1_ago',
                          'commercial_broadcaster_ad_1_ago', 'public_broadcaster_ad_1_ago','sports_ad_1_ago', 'cooking_ad_1_ago',
                          'ad_long_ad_1_ago', 'ad_mid_ad_1_ago',
                          'weekend_ad_1_ago',
                          'first_position_ad_1_ago','second_position_ad_1_ago','last_position_ad_1_ago'],
                      3: #model 4 in report 
                          ['gpr_ad_1_ago', 'gpr_ad_1_ago_squared',
                          'morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago','night_ad_1_ago',
                          'wasmachines_ad_1_ago','televisies_ad_1_ago',
                          'program_news_ad_1_ago',
                          'between_programs_ad_1_ago',
                          'int_televisies_gpr_ad_1_ago', 'int_wasmachines_gpr_ad_1_ago',
                          'int_televisies_between_programs_ad_1_ago','int_wasmachines_between_programs_ad_1_ago',
                          'int_televisies_morning_ad_1_ago', 'int_wasmachines_morning_ad_1_ago',
                          'int_televisies_afternoon_ad_1_ago','int_wasmachines_afternoon_ad_1_ago', 
                          'int_televisies_prime_ad_1_ago','int_wasmachines_prime_ad_1_ago',
                          'int_televisies_program_news_ad_1_ago','int_wasmachines_program_news_ad_1_ago',
                          'int_news_morning_ad_1_ago','int_news_afternoon_ad_1_ago','int_news_prime_ad_1_ago', 
                          'int_between_programs_morning_ad_1_ago','int_between_programs_afternoon_ad_1_ago','int_between_programs_prime_ad_1_ago']
                      } 

rmse_output = pd.DataFrame(columns=['model_nr', 'rmse_exog', 'rmse_exog_ad','rmse_exog_ad_3','MAE','MAPE', 'R2'], index=range(len(exogenous_variables)))

model_output = pd.DataFrame({'vars': ['gpr_ad_1_ago','gpr_ad_1_ago_squared', 
                                      'morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago','night_ad_1_ago',
                                      'commercial_broadcaster_ad_1_ago', 'public_broadcaster_ad_1_ago','sports_ad_1_ago', 'cooking_ad_1_ago',
                                      'wasmachines_ad_1_ago','televisies_ad_1_ago',
                                      'ad_long_ad_1_ago', 'ad_mid_ad_1_ago',
                                      'weekend_ad_1_ago',
                                      'program_news_ad_1_ago',
                                      'between_programs_ad_1_ago',
                                      'first_position_ad_1_ago','second_position_ad_1_ago','last_position_ad_1_ago',
                                      'int_televisies_gpr_ad_1_ago', 'int_wasmachines_gpr_ad_1_ago',
                                      'int_televisies_between_programs_ad_1_ago','int_wasmachines_between_programs_ad_1_ago',
                                      'int_televisies_morning_ad_1_ago', 'int_wasmachines_morning_ad_1_ago',
                                      'int_televisies_afternoon_ad_1_ago','int_wasmachines_afternoon_ad_1_ago', 
                                      'int_televisies_prime_ad_1_ago','int_wasmachines_prime_ad_1_ago',
                                      'int_televisies_program_news_ad_1_ago','int_wasmachines_program_news_ad_1_ago',
                                      'int_news_morning_ad_1_ago','int_news_afternoon_ad_1_ago','int_news_prime_ad_1_ago', 
                                      'int_between_programs_morning_ad_1_ago','int_between_programs_afternoon_ad_1_ago','int_between_programs_prime_ad_1_ago',
                                      'ar.L1', 'ar.L2', 'ar.L3', 'ar.L4', 'ar.L5','ar.L6', 'ar.L7', 
                                      'ma.L1', 'ma.L2', 
                                      'sigma2']})
model_output = model_output.set_index('vars')

for mod_nr in range(len(exogenous_variables)): 
    print(mod_nr)
    exog = dff[exogenous_variables[mod_nr]]
    train_exog, test_exog = exog.iloc[:train_size], exog.iloc[train_size:]

    # Fit model
    mod = sm.tsa.SARIMAX(endog = train, exog = train_exog, order = optimal_params)
    start=time.process_time()
    model = mod.fit()
    end=time.process_time()
    print('Model fitting time: {} seconds'.format(end-start))
    
    # Save model to disk
    #filename = 'model_exog_' + str(mod_nr) + '_'  + str(optimal_params) + '.pkl'
    #model.save(filename)
        
    # Use the next line when fitting the models in the for loop
    loaded_model_exog = model
    
    # Use the next lines when model fitting has already been performed and you want to load the models from disk
    #filename = 'model_exog_' + str(mod_nr) + '_'  + str(optimal_params) + '.pkl' # Insert here the desired file name
    #loaded_model_exog = ARIMAResults.load(filename)
    
    # Forecast
    forecasts_exog = []   # Create empty list to store forecast results
    forecasts_exog.append(loaded_model_exog.forecast(exog=exog.iloc[train_size-1:train_size])[0]) 
      # Save initial forecast
    
    # Step through the rest of the sample
    for t in tqdm(range(train_size, n_obs-1)):
        # Update the results by appending the next observation
        updated_endog = endog.iloc[t:t+1]
        updated_exog = exog.iloc[t:t+1]
        loaded_model_exog = loaded_model_exog.extend(endog = updated_endog, exog = updated_exog)
        obs = test[t-train_size]
        output = loaded_model_exog.forecast(exog = updated_exog)
        yhat = output[0]
        # Save the new set of forecasts
        forecasts_exog.append(yhat)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    
    # some data manipulation to get the output in desired format...
    forecasts_exog = pd.DataFrame(forecasts_exog).set_index(test.index)
    
    # Calculate MAPE - first some data manipulation to get data in right format
    forecasts_exog_vals = forecasts_exog.reset_index()[0]
    forecasts_exog_array = np.array(forecasts_exog_vals)

    test_mape = test.replace(0,0.00001)
    test_vals = test_mape.reset_index()['visits_index']
    test_array = np.array(test_vals)

    mape_exog = np.mean(np.abs(forecasts_exog_array - test_array) / np.abs(test_array)) * 100
    print('MAPE: %.3f' % mape_exog)
    
    mae_exog = np.mean(np.abs(forecasts_exog_array - test_array)) 
    print('MAE: %.3f' % mae_exog)

    # Calculate R2
    r2_exog = r2_score(test, forecasts_exog)
    print('R2: %.3f' % r2_exog)
    
    # Evaluate forecasts by means of root mean squared error
    rmse_exog = sqrt(mean_squared_error(test, forecasts_exog))
    print('Test RMSE: %.3f' % rmse_exog)
    
    # Evaluate only forecasts for which an advertisement took place in 1 minute pre window
    test_df = dff.iloc[train_size:] # Obtain test set with extra variables
    temp_ind = test_df[test_df['ad_1_ago']>0].index # Obtain indeces of test set for which the observations falls in post-ad period
    test_ad = test[test.index.isin(temp_ind)] # Filter test results on post-ad period
    fc_exog_ad = forecasts_exog[forecasts_exog.index.isin(temp_ind)] # Filter forecast results on post-ad period
    rmse_exog_ad = sqrt(mean_squared_error(test_ad, fc_exog_ad))
    print('Test RMSE of post-ad (1min) period: %.3f' % rmse_exog_ad)
    
    # And evaluate forecasts for which an advertisement took place in 3 minute pre window
    test_df = dff.iloc[train_size:] # Obtain test set with extra variables
    temp_ind = test_df[(test_df['ad_1_ago']>0) | (test_df['ad_2_ago']>0) | (test_df['ad_3_ago']>0)].index # Obtain indeces of test set for which the observations falls in post-ad period
    test_ad = test[test.index.isin(temp_ind)] # Filter test results on post-ad period
    fc_exog_ad = forecasts_exog[forecasts_exog.index.isin(temp_ind)] # Filter forecast results on post-ad period
    rmse_exog_ad_3 = sqrt(mean_squared_error(test_ad, fc_exog_ad))
    print('Test RMSE of (3 min) post-ad period: %.3f' % rmse_exog_ad_3)
    
    rmse_output['model_nr'][mod_nr] = mod_nr
    rmse_output['rmse_exog'][mod_nr] = rmse_exog
    rmse_output['rmse_exog_ad'][mod_nr] = rmse_exog_ad
    rmse_output['rmse_exog_ad_3'][mod_nr] = rmse_exog_ad_3
    rmse_output['MAPE'][mod_nr] = mape_exog
    rmse_output['MAE'][mod_nr] = mae_exog
    rmse_output['R2'][mod_nr] = r2_exog

    coefs = pd.DataFrame(loaded_model_exog.params, columns=['M'+str(mod_nr)+'_coef']) 
    pvals = pd.DataFrame(loaded_model_exog.pvalues, columns=['M'+str(mod_nr)+'_pval']) 
    
    model_output = model_output.join(coefs)
    model_output = model_output.join(pvals)
    
    # Save the forecasts to unique dataframes, in order to easily extract these values afterwards
    if mod_nr==0:
        forecasts_mod_0 = forecasts_exog
    elif mod_nr==1:
        forecasts_mod_1 = forecasts_exog
    elif mod_nr==2:
        forecasts_mod_2 = forecasts_exog    
    elif mod_nr==3:
        forecasts_mod_3 = forecasts_exog

# Report: Table 12 (columns model 1-model 4)
model_output.to_csv("XXX/XXX/model_output_selectedmodels.csv")
rmse_output.to_csv("XXX/XXX/rmse_output_selectedmodels.csv")


###############################################################################
###############################################################################
####### Plots #################################################################
###############################################################################
###############################################################################

# Change index to be able to plot
test_df.index = test_df.index.to_timestamp()
test.index = test.index.to_timestamp() # Convert period index back to datetime index

forecasts.index = forecasts.index.to_timestamp() # Convert period index back to datetime index
forecasts_mod_0.index = forecasts_mod_0.index.to_timestamp() # Convert period index back to datetime index
forecasts_mod_1.index = forecasts_mod_1.index.to_timestamp() # Convert period index back to datetime index
forecasts_mod_2.index = forecasts_mod_2.index.to_timestamp() # Convert period index back to datetime index
forecasts_mod_3.index = forecasts_mod_3.index.to_timestamp() # Convert period index back to datetime index

# Report: Figure 13
# Set figure size
plt.rcParams['figure.figsize'] = 35, 25

begin=18500 # set beginning index of plot 18000
end=18800 # set ending index of plot 19000

plt.subplot(6,1,1)
plt.plot(forecasts[begin:end], color='red')
plt.plot(test[begin:end])
plt.title("Baseline ARIMA(7,1,2)")

plt.subplot(6,1,2)
plt.plot(forecasts_mod_0[begin:end], color='red')
plt.plot(test[begin:end])
plt.title("ARIMAX(7,1,2) Model 1")

plt.subplot(6,1,3)
plt.plot(forecasts_mod_1[begin:end], color='red')
plt.plot(test[begin:end])
plt.title("ARIMAX(7,1,2) Model 2")

plt.subplot(6,1,4)
plt.plot(forecasts_mod_2[begin:end], color='red')
plt.plot(test[begin:end])
plt.title("ARIMAX(7,1,2) Model 3")

plt.subplot(6,1,5)
plt.plot(forecasts_mod_3[begin:end], color='red')
plt.plot(test[begin:end])
plt.title("ARIMAX(7,1,2) Model 4 (with interactions)")

plt.subplot(6,1,6)
plt.plot(test_df['gpr_ad_1_ago'][begin:end])
plt.ylim(0,16)
plt.title("Gross rating point of adds")

plt.show()

######## For feedback presentation

begin=18500 # set beginning index of plot
end=18800 # set ending index of plot

plt.subplot(2,1,1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.plot(forecasts_mod_3[begin:end], color='red',linewidth=0.5)
plt.plot(test[begin:end],linewidth=0.5)
plt.title("ARIMAX(7,1,2) with interaction variables")

plt.subplot(2,1,2)
plt.plot(test_df['gpr_ad_1_ago'][begin:end])
plt.ylim(0,16)
plt.title("Gross rating point of ads")

plt.show()


