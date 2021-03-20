# Import packages
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

# Import data
broadcast_original = pd.read_csv("XXX/XXX/broadcasting_data")
traffic_original = pd.read_csv("XXX/XXX/traffic_data")

####### Convert datetime objects ##################################################################################################################################
traffic_original['date_time'] = pd.to_datetime(traffic_original['date_time'], format="%Y-%m-%d %H:%M:%S") 

broadcast_original['date_time'] = broadcast_original['date'] +' '+ broadcast_original['time']
broadcast_original['date_time'] = pd.to_datetime(broadcast_original['date_time'])

broadcast_original['date'] = pd.to_datetime(broadcast_original['date']).dt.date
broadcast_original['hour'] = pd.to_datetime(broadcast_original['date_time']).dt.hour
broadcast_original['time'] = pd.to_datetime(broadcast_original['time']).dt.time
broadcast_original['day_of_week'] = pd.to_datetime(broadcast_original['date_time']).dt.dayofweek

traffic_original['date'] = pd.to_datetime(traffic_original['date_time']).dt.date
traffic_original['hour'] = pd.to_datetime(traffic_original['date_time']).dt.hour
traffic_original['time'] = pd.to_datetime(traffic_original['date_time']).dt.time

####### Exclude bounces and 'other'/'push' visit sources ##################################################################################################################################
traffic_relevant = traffic_original.loc[ (traffic_original['bounces'] != 1) &
                                 (traffic_original['visit_source'].isin(['search','paid search','direct']) &
                                 (traffic_original['medium'] == 'website'))
                                ]  
traffic_relevant = traffic_relevant.drop(columns=['bounces'])

####### Include only observations from the Netherlands ##################################################################################################################################
traffic = traffic_relevant.loc[traffic_relevant['country'] == 'Netherlands']
broadcast = broadcast_original.loc[broadcast_original['country'] == 'Netherlands']
broadcast.reset_index(drop=True, inplace=True)

####### Include features #############################################################################################################################

# Position in break
broadcast['position_in_break_new'] = 'Any other position'
broadcast.loc[(broadcast['position_in_break']=='1') | (broadcast['position_in_break']=='First Position'), 'position_in_break_new'] = 'First Position'
broadcast.loc[(broadcast['position_in_break']=='2') | (broadcast['position_in_break']=='Second Position'), 'position_in_break_new'] = 'Second Position'
broadcast.loc[(broadcast['position_in_break']=='99') | (broadcast['position_in_break']=='Last Position'), 'position_in_break_new'] = 'Last Position'
broadcast['position_in_break'] = broadcast['position_in_break_new']

broadcast = broadcast.join(pd.get_dummies(broadcast['position_in_break']))

# Channel groups
broadcast.loc[(broadcast['channel']=='24Kitchen'),'channel_group'] = 'cooking'
broadcast.loc[(broadcast['channel']=='RTL Z'),'channel_group'] = 'business'
broadcast.loc[(broadcast['channel']=='BBC First Holland')|(broadcast['channel']=='ID')|(broadcast['channel']=='RTL Crime'),'channel_group'] = 'drama/crime'
broadcast.loc[(broadcast['channel']=='TLC')|(broadcast['channel']=='RTL 8'),'channel_group'] = 'women'
broadcast.loc[(broadcast['channel']=='MTV')|(broadcast['channel']=='Slam!TV')|(broadcast['channel']=='TV538'),'channel_group'] = 'music'
broadcast.loc[(broadcast['channel']=='Comedy Central')|(broadcast['channel']=='Viceland'),'channel_group'] = 'youth'
broadcast.loc[(broadcast['channel']=='Eurosport')|(broadcast['channel']=='Fox Sports 1')|(broadcast['channel']=='Fox Sports 2')|(broadcast['channel']=='Fox Sports 3'),'channel_group'] = 'sports'
broadcast.loc[(broadcast['channel']=='Discovery Channel')|(broadcast['channel']=='National Geographic Channel'),'channel_group'] = 'science'
broadcast.loc[(broadcast['channel']=='Net5')|(broadcast['channel']=='RTL 4')|(broadcast['channel']=='RTL 5')|(broadcast['channel']=='SBS 6')|(broadcast['channel']=='SBS 9')|(broadcast['channel']=='Veronica'),'channel_group'] = 'commercial broadcaster'
broadcast.loc[(broadcast['channel']=='Fox')|(broadcast['channel']=='Spike')|(broadcast['channel']=='RTL 7'),'channel_group'] = 'men'
broadcast.loc[(broadcast['channel']=='NPO1')|(broadcast['channel']=='NPO2')|(broadcast['channel']=='NPO3'),'channel_group'] = 'public broadcaster'
broadcast = broadcast.join(pd.get_dummies(broadcast['channel_group']))

# Product category
broadcast = broadcast.join(pd.get_dummies(broadcast['product_category']))

broadcast['program_category'] = 'other'
broadcast.loc[(broadcast['program_category_before']=='nieuws')|
              (broadcast['program_category_after']=='nieuws')
              ,'program_category'] = 'news'
broadcast = broadcast.join(pd.get_dummies(broadcast['program_category'],prefix='program'))


# Dummy for the length of the commercial
broadcast['ad_long'] = np.where(broadcast['length_of_spot']=='30 + 10 + 5', 1, 0)
broadcast['ad_mid'] = np.where(broadcast['length_of_spot']=='30 + 10', 1, 0)
broadcast['ad_short'] = np.where(broadcast['length_of_spot']=='30', 1, 0)

# Dummy for whether the commercial is broadcasted in between to different programs
broadcast['between_programs'] = np.where(broadcast['program_before']==broadcast['program_after'], 0, 1)

# Dummy for part of day
broadcast['night'] = np.where((broadcast['hour']>=1) & (broadcast['hour']<7), 1, 0)
broadcast['morning'] = np.where((broadcast['hour']>=7) & (broadcast['hour']<12), 1, 0)
broadcast['afternoon'] = np.where((broadcast['hour']>=12) & (broadcast['hour']<18), 1, 0)
broadcast['prime'] = np.where((broadcast['hour']>=18) & (broadcast['hour']<23), 1, 0)
broadcast['late_evening'] = np.where((broadcast['hour']>=23) & (broadcast['hour']<1), 1, 0)

# Dummy for weekend
broadcast['weekend'] = np.where((broadcast['day_of_week']==6 )|(broadcast['day_of_week']==5), 1, 0)


############################################# Turn into data suitable for ARIMA ##################################################################################
traffic_relevant_NL = traffic
broadcast_NL = broadcast

# Aggregate traffic data on unique minutes
visits_per_minute = traffic_relevant_NL.groupby(traffic_relevant_NL['date_time'])['visits_index'].sum().reset_index()

# Choose the required data set
data = visits_per_minute

# Transform data to obtain a datetime index
data = data.set_index('date_time')

# Impute missing hours by reindexing and fill NA values with 0
data_temp = data.reindex(pd.date_range('2019-01-01', '2019-06-30 23:59:59', freq='T'))
dff = data_temp.fillna(0) # (decompose function cannot deal with NA's)
dff = pd.DataFrame(dff)

dff = dff.reset_index()
dff['day_of_week'] = pd.to_datetime(dff['index']).dt.dayofweek
dff['hour'] = pd.to_datetime(dff['index']).dt.hour
dff['date_time'] = dff['index']
dff['time'] = pd.to_datetime(dff['date_time']).dt.time

############################################# Treat outliers at 00:00h every night   ####################################################################################

# Save original visits_index in separate column
dff['visits_index_original'] = dff['visits_index']

# VI[00:00] = 0.5*(VI[23:59]+VI[00:01])
dff['visits_index'] = np.where(dff['time'] == datetime.time(0,0,0), 0.5*(dff['visits_index_original'].shift(1)+dff['visits_index_original'].shift(-1)),dff['visits_index_original'])

# Only the datetime: 2019-03-02 00:00:00 gets the visit index of 2019-03-01 23:59:00
# This is the only observation that has an advertisement around 00:00 with an effect that otherwise causes trouble
dff['visits_index'] = np.where(dff['date_time'] == datetime.datetime(2019,3,2,0,0,0), dff['visits_index_original'].shift(1),dff['visits_index'])



############################################# Include features of the advertisements at t-1 ################################################################

dff['ad_1_ago'] = np.nan
dff['gpr_ad_1_ago'] = np.nan
dff['gpr_ad_1_ago_squared'] = np.nan
dff['between_programs_ad_1_ago'] = np.nan
dff['laptops_ad_1_ago'] = np.nan
dff['televisies_ad_1_ago'] = np.nan
dff['wasmachines_ad_1_ago'] = np.nan
dff['morning_ad_1_ago'] = np.nan
dff['afternoon_ad_1_ago'] = np.nan
dff['prime_ad_1_ago'] = np.nan
dff['night_ad_1_ago'] = np.nan
dff['commercial_broadcaster_ad_1_ago'] = np.nan
dff['public_broadcaster_ad_1_ago'] = np.nan
dff['sports_ad_1_ago'] = np.nan
dff['cooking_ad_1_ago'] = np.nan
dff['ad_long_ad_1_ago'] = np.nan
dff['ad_mid_ad_1_ago'] = np.nan
dff['weekend_ad_1_ago'] = np.nan
dff['program_news_ad_1_ago'] = np.nan
dff['first_position_ad_1_ago'] = np.nan
dff['second_position_ad_1_ago'] = np.nan
dff['last_position_ad_1_ago'] = np.nan


dff['date_time_minus_1'] = dff['date_time'] - datetime.timedelta(minutes=1)

for i in tqdm(range(len(dff))):
    date_time_minus_1 = dff.iloc[i]['date_time_minus_1']
    rows = broadcast_NL.loc[broadcast_NL['date_time'] == date_time_minus_1]
    rows_sorted = rows.sort_values(by='gross_rating_point',ascending=False)
    
    if (len(rows_sorted) > 0):
        dff['ad_1_ago'][i] = 1
        row = rows_sorted.iloc[0,]
        dff['gpr_ad_1_ago'][i] = row['gross_rating_point']
        dff['gpr_ad_1_ago_squared'][i] = row['gross_rating_point']**2
        dff['between_programs_ad_1_ago'][i] = row['between_programs']
        dff['laptops_ad_1_ago'][i] = row['laptops']
        dff['televisies_ad_1_ago'][i] = row['televisies']
        dff['wasmachines_ad_1_ago'][i] = row['wasmachines']
        dff['morning_ad_1_ago'][i] = row['morning']
        dff['afternoon_ad_1_ago'][i] = row['afternoon']
        dff['prime_ad_1_ago'][i] = row['prime']
        dff['night_ad_1_ago'][i] = row['night']
        dff['commercial_broadcaster_ad_1_ago'][i] = row['commercial broadcaster']
        dff['public_broadcaster_ad_1_ago'][i] = row['public broadcaster']
        dff['sports_ad_1_ago'][i] = row['sports']
        dff['cooking_ad_1_ago'][i] = row['cooking']
        dff['ad_long_ad_1_ago'][i] = row['ad_long']
        dff['ad_mid_ad_1_ago'][i] = row['ad_mid']
        dff['weekend_ad_1_ago'][i] = row['weekend']
        dff['program_news_ad_1_ago'][i] = row['program_news']
        dff['first_position_ad_1_ago'][i] = row['First Position']
        dff['second_position_ad_1_ago'][i] = row['Second Position']
        dff['last_position_ad_1_ago'][i] = row['Last Position']
    else:
        pass

dff['ad_2_ago'] = dff['ad_1_ago'].shift(1)
dff['ad_3_ago'] = dff['ad_1_ago'].shift(2)
    
dff = dff.fillna(0)
dff = dff.iloc[1:] # Remove the very first row because this row is at 00:00

# Interaction features
# Interactions with product
variables1 = ['gpr_ad_1_ago', #not gpr squared
             'between_programs_ad_1_ago',
             'morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago', #not night
             'program_news_ad_1_ago',
             'between_programs_ad_1_ago']
for var in variables1:
    tv_interaction = 'int_televisies_' + str(var) 
    wm_interaction = 'int_wasmachines_' + str(var) 
    dff[tv_interaction] = dff['televisies_ad_1_ago'] * dff[var]
    dff[wm_interaction] = dff['wasmachines_ad_1_ago'] * dff[var]

# Interactions with part of day 
variables2 = ['morning_ad_1_ago','afternoon_ad_1_ago','prime_ad_1_ago'] #not night 
for var in variables2:
    news_interaction = 'int_news_' + str(var) 
    between_interaction = 'int_between_programs_' + str(var) 
    dff[news_interaction] = dff['program_news_ad_1_ago'] * dff[var]
    dff[between_interaction] = dff['between_programs_ad_1_ago'] * dff[var]


######################## OUTPUT DATA = INPUT FOR ARIMA ##########################
dff.to_csv("XXX/XXX/data_for_ARIMA.csv", index = False)
