####### Import packages ##################################################################################################################################
import pandas as pd
import numpy as np
import datetime

####### Import data  ##################################################################################################################################
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

####### Include only commercials which do not overlap ##################################################################################################################################
broadcast['date_time_minus_3'] = broadcast['date_time'] - datetime.timedelta(minutes=3)
broadcast['date_time_plus_3'] = broadcast['date_time'] + datetime.timedelta(minutes=3)

broadcast['date_time_minus_5'] = broadcast['date_time'] - datetime.timedelta(minutes=5)
broadcast['date_time_plus_5'] = broadcast['date_time'] + datetime.timedelta(minutes=5)

####### Aggregate the sum of visit indices per minute ##################################################################################################################################
visits_per_minute = traffic.groupby(traffic['date_time'])['visits_index'].sum().reset_index()

####### Initialize the variable that counts the visit index before and after the campaign ##################################################################################################################################
broadcast['visits_pre_window_3'] = np.nan
broadcast['visits_post_window_3'] = np.nan

broadcast['visits_pre_window_5'] = np.nan
broadcast['visits_post_window_5'] = np.nan

for i in range(len(broadcast)):
    visit_rows_pre = visits_per_minute.loc[(visits_per_minute['date_time'] >= broadcast.iloc[i]['date_time_minus_5']) & (visits_per_minute['date_time'] < broadcast.iloc[i]['date_time'])]
    broadcast['visits_pre_window_5'][i] = visit_rows_pre['visits_index'].sum()
    
    visit_rows_post = visits_per_minute.loc[(visits_per_minute['date_time'] > broadcast.iloc[i]['date_time']) & (visits_per_minute['date_time'] <= broadcast.iloc[i]['date_time_plus_5'])]
    broadcast['visits_post_window_5'][i] = visit_rows_post['visits_index'].sum()
    
broadcast['effect_prepost_window_5'] = broadcast['visits_post_window_5'] - broadcast['visits_pre_window_5']
broadcast['effect_prepost_window_5_perc'] = (broadcast['visits_post_window_5'] - broadcast['visits_pre_window_5'])/broadcast['visits_pre_window_5']


####### Create the variable that counts the visit index before and after the campaign ##################################################################################################################################
for i in range(len(broadcast)):
    visit_rows_pre = visits_per_minute.loc[(visits_per_minute['date_time'] >= broadcast.iloc[i]['date_time_minus_3']) & (visits_per_minute['date_time'] < broadcast.iloc[i]['date_time'])]
    broadcast['visits_pre_window_3'][i] = visit_rows_pre['visits_index'].sum()
    
    visit_rows_post = visits_per_minute.loc[(visits_per_minute['date_time'] > broadcast.iloc[i]['date_time']) & (visits_per_minute['date_time'] <= broadcast.iloc[i]['date_time_plus_3'])]
    broadcast['visits_post_window_3'][i] = visit_rows_post['visits_index'].sum()
    
####### Calculate the effect of a broadcasting campaign on the number of visits ##################################################################################################################################
broadcast['effect_prepost_window_3'] = broadcast['visits_post_window_3'] - broadcast['visits_pre_window_3']
broadcast['effect_prepost_window_3_perc'] = (broadcast['visits_post_window_3'] - broadcast['visits_pre_window_3'])/broadcast['visits_pre_window_3']


# Report: Table 6
broadcast['effect_prepost_window_3'].mean()
broadcast['effect_prepost_window_3'].std()

broadcast['effect_prepost_window_5'].mean()
broadcast['effect_prepost_window_5'].std()
