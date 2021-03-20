# Import packages
import pandas as pd
import numpy as np
import datetime

####### Import data ##################################################################################################################################
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

# Keep filtered data set up to the point that overlapping commercials are included
broadcast_relevant_NL = broadcast[0:]

####### Include only commercials which do not overlap ##################################################################################################################################
broadcast['date_time_minus_3'] = broadcast['date_time'] - datetime.timedelta(minutes=3)
broadcast['date_time_plus_3'] = broadcast['date_time'] + datetime.timedelta(minutes=3)

broadcast['date_time_minus_5'] = broadcast['date_time'] - datetime.timedelta(minutes=5)
broadcast['date_time_plus_5'] = broadcast['date_time'] + datetime.timedelta(minutes=5)


broadcast_check_overlap = broadcast.sort_values(by=['date_time'])
broadcast_check_overlap['date_time_prev_broadcast_plus_3'] = broadcast_check_overlap['date_time_plus_3'].shift(1) 
broadcast_check_overlap['date_time_next_broadcast_minus_3'] = broadcast_check_overlap['date_time_minus_3'].shift(-1)

broadcast_check_overlap['overlaps_with_other'] = np.where((broadcast_check_overlap['date_time_minus_3'] < broadcast_check_overlap['date_time_prev_broadcast_plus_3'])
                                                          | (broadcast_check_overlap['date_time_plus_3'] > broadcast_check_overlap['date_time_next_broadcast_minus_3']) ,1,0)

broadcast = broadcast_check_overlap.loc[broadcast_check_overlap['overlaps_with_other']==0]
broadcast = broadcast.reset_index()

##################################################################################################################################
####### PRE-POST WINDOW VARIABLE CREATION ##################################################################################################################################
##################################################################################################################################

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

broadcast['effect_prepost_window_3_capped_at_12'] = np.where((broadcast['effect_prepost_window_3_perc']>0.12), broadcast['effect_prepost_window_3'], 0)
broadcast['effect_prepost_window_3_capped_at_10'] = np.where((broadcast['effect_prepost_window_3_perc']>0.1), broadcast['effect_prepost_window_3'], 0)
broadcast['effect_prepost_window_3_capped_at_8'] = np.where((broadcast['effect_prepost_window_3_perc']>0.08), broadcast['effect_prepost_window_3'], 0)
broadcast['effect_prepost_window_3_capped_at_5'] = np.where((broadcast['effect_prepost_window_3_perc']>0.05), broadcast['effect_prepost_window_3'], 0)

broadcast['effect_classifier_12_perc'] = np.where((broadcast['effect_prepost_window_3_capped_at_12']>0),1,0)
broadcast['effect_classifier_10_perc'] = np.where((broadcast['effect_prepost_window_3_capped_at_10']>0),1,0)
broadcast['effect_classifier_8_perc'] = np.where((broadcast['effect_prepost_window_3_capped_at_8']>0),1,0)
broadcast['effect_classifier_5_perc'] = np.where((broadcast['effect_prepost_window_3_capped_at_5']>0),1,0)

##################################################################################################################################
####### COMPUTE DATASET FOR MODELS ##################################################################################################################################
##################################################################################################################################

####### Redefine position in break variable ##################################################################################################################################
broadcast['position_in_break_new'] = 'Any other position'
broadcast.loc[(broadcast['position_in_break']=='1') | (broadcast['position_in_break']=='First Position'), 'position_in_break_new'] = 'First Position'
broadcast.loc[(broadcast['position_in_break']=='2') | (broadcast['position_in_break']=='Second Position'), 'position_in_break_new'] = 'Second Position'
broadcast.loc[(broadcast['position_in_break']=='99') | (broadcast['position_in_break']=='Last Position'), 'position_in_break_new'] = 'Last Position'

broadcast['position_in_break'] = broadcast['position_in_break_new']

# For the data exploration of raw data
broadcast_original['position_in_break_new'] = 'Any other position'
broadcast_original.loc[(broadcast_original['position_in_break']=='1') | (broadcast_original['position_in_break']=='First Position'), 'position_in_break_new'] = 'First Position'
broadcast_original.loc[(broadcast_original['position_in_break']=='2') | (broadcast_original['position_in_break']=='Second Position'), 'position_in_break_new'] = 'Second Position'
broadcast_original.loc[(broadcast_original['position_in_break']=='99') | (broadcast_original['position_in_break']=='Last Position'), 'position_in_break_new'] = 'Last Position'

####### Create groups of channels ##################################################################################################################################
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

####### Create dummies for certain program catogories ##################################################################################################################################
broadcast['program_category'] = 'other'
broadcast.loc[(broadcast['program_category_before']=='actuele sportinformatie')|
              (broadcast['program_category_before']=='voetbalreportage')|
              (broadcast['program_category_before']=='overige sportinformatie')|
              (broadcast['program_category_before']=='overige sportreportage')|
              (broadcast['program_category_after']=='actuele sportinformatie')|
              (broadcast['program_category_after']=='voetbalreportage')|
              (broadcast['program_category_after']=='overige sportinformatie')|
              (broadcast['program_category_after']=='overige sportreportage')
              ,'program_category'] = 'sports'

broadcast.loc[(broadcast['program_category_before']=='btl series: spanning')|
              (broadcast['program_category_before']=='btl series: overig')|
              (broadcast['program_category_before']=='btl series: drama')|
              (broadcast['program_category_before']=='nld series: drama')|
              (broadcast['program_category_before']=='btl series: (sit)comedy')|
              (broadcast['program_category_before']=='btl series: soap')|
              (broadcast['program_category_before']=='nld series: spanning')|
              (broadcast['program_category_before']=='nld series: (sit)comedy')|
              (broadcast['program_category_before']=='nld series: overig')|
              (broadcast['program_category_before']=='nld series: soap')|
              (broadcast['program_category_after']=='btl series: spanning')|
              (broadcast['program_category_after']=='btl series: overig')|
              (broadcast['program_category_after']=='btl series: drama')|
              (broadcast['program_category_after']=='nld series: drama')|
              (broadcast['program_category_after']=='btl series: (sit)comedy')|
              (broadcast['program_category_after']=='btl series: soap')|
              (broadcast['program_category_after']=='nld series: spanning')|
              (broadcast['program_category_after']=='nld series: (sit)comedy')|
              (broadcast['program_category_after']=='nld series: overig')|
              (broadcast['program_category_after']=='nld series: soap')
              ,'program_category'] = 'series'

broadcast.loc[(broadcast['program_category_before']=='btl films: spanning')|
              (broadcast['program_category_before']=='nld films: drama')|
              (broadcast['program_category_before']=='nld films: comedy')|
              (broadcast['program_category_before']=='btl films: comedy')|
              (broadcast['program_category_before']=='nld films: spanning')|
              (broadcast['program_category_before']=='btl films: drama')|
              (broadcast['program_category_before']=='btl films: overig')|
              (broadcast['program_category_after']=='btl films: spanning')|
              (broadcast['program_category_after']=='nld films: drama')|
              (broadcast['program_category_after']=='nld films: comedy')|
              (broadcast['program_category_after']=='btl films: comedy')|
              (broadcast['program_category_after']=='nld films: spanning')|
              (broadcast['program_category_after']=='btl films: drama')|
              (broadcast['program_category_after']=='btl films: overig')
              ,'program_category'] = 'films'

broadcast.loc[(broadcast['program_category_before']=='kinderfilms: tekenfilm/animatie/poppen')|
              (broadcast['program_category_before']=='kinderen: non fictie')|
              (broadcast['program_category_after']=='kinderfilms: tekenfilm/animatie/poppen')|
              (broadcast['program_category_after']=='kinderen: non fictie')
              ,'program_category'] = 'kids'

broadcast.loc[(broadcast['program_category_before']=='nieuws')|
              (broadcast['program_category_after']=='nieuws')
              ,'program_category'] = 'news'

####### Drop unnecessary variables ##################################################################################################################################
broadcast = broadcast.drop(['date_time_minus_3'
                            ,'date_time_plus_3'
                            ,'date_time_prev_broadcast_plus_3'
                            ,'date_time_next_broadcast_minus_3'
                            ,'overlaps_with_other'
                            ,'visits_pre_window_3'
                            ,'visits_post_window_3'
                            ,'country'
                            ,'position_in_break_new'
                            ,'channel'
                            ],axis=1)

####### Convert categoricals to dummy variables ##################################################################################################################################
# Dummies for the length of the commercial
broadcast['ad_long'] = np.where(broadcast['length_of_spot']=='30 + 10 + 5', 1, 0)
broadcast['ad_mid'] = np.where(broadcast['length_of_spot']=='30 + 10', 1, 0)
broadcast['ad_short'] = np.where(broadcast['length_of_spot']=='30', 1, 0)

# Dummy for whether the commercial is broadcasted in between to different programs
broadcast['between_programs'] = np.where(broadcast['program_before']==broadcast['program_after'], 0, 1)

# Dummies for day of the week
# Default is Sunday
broadcast['Monday'] = np.where(broadcast['day_of_week']==0, 1, 0)
broadcast['Tuesday'] = np.where(broadcast['day_of_week']==1, 1, 0)
broadcast['Wednesday'] = np.where(broadcast['day_of_week']==2, 1, 0)
broadcast['Thursday'] = np.where(broadcast['day_of_week']==3, 1, 0)
broadcast['Friday'] = np.where(broadcast['day_of_week']==4, 1, 0)
broadcast['Saturay'] = np.where(broadcast['day_of_week']==5, 1, 0)
broadcast['Sunday'] = np.where(broadcast['day_of_week']==6, 1, 0)

# Dummies for hour of day
# Default is Sunday
broadcast = broadcast.join(pd.get_dummies(broadcast['hour'],prefix='hour'))

# Dummies for channel
broadcast = broadcast.join(pd.get_dummies(broadcast['channel_group']))

# Dummies for position in break
broadcast = broadcast.join(pd.get_dummies(broadcast['position_in_break']))

# Dummies for product category
broadcast = broadcast.join(pd.get_dummies(broadcast['product_category']))

broadcast = broadcast.join(pd.get_dummies(broadcast['program_category'],prefix='program'))

# Dummies for primetime
broadcast['night'] = np.where((broadcast['hour']>=1) & (broadcast['hour']<7), 1, 0)
broadcast['morning'] = np.where((broadcast['hour']>=7) & (broadcast['hour']<12), 1, 0)
broadcast['afternoon'] = np.where((broadcast['hour']>=12) & (broadcast['hour']<18), 1, 0)
broadcast['prime'] = np.where((broadcast['hour']>=18) & (broadcast['hour']<23), 1, 0)
broadcast['late_evening'] = np.where((broadcast['hour']>=23) & (broadcast['hour']<1), 1, 0)

broadcast['weekend'] = np.where((broadcast['day_of_week']==5) | (broadcast['day_of_week']==6), 1, 0)

##################################################################################################################################
####### SAVE THE DATASETS FOR USE IN MODELS ##################################################################################################################################
##################################################################################################################################
# For data exploration only
broadcast_original.to_csv("XXX/XXX/broadcast_incl_feat.csv", index = False)
broadcast_relevant_NL.to_csv("XXX/XXX/broadcast_relevant_NL.csv", index = False)

# For use in models
broadcast.to_csv("XXX/XXX/broadcast_for_models.csv")
traffic.to_csv("XXX/XXX/traffic_relevant_NL.csv", index = False)

