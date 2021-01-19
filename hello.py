# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:43:51 2021

@author: Dell
"""
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Import data
broadcasting = pd.read_csv("C:/Users/Dell/Documents/ELJA/Master/Case studies/broadcasting_data")
traffic = pd.read_csv("C:/Users/Dell/Documents/ELJA/Master/Case studies/traffic_data")

## Convert to date time objects 
broadcasting['date_time'] = broadcasting['date'] +' '+ broadcasting['time']

broadcasting['date_time'] = pd.to_datetime(broadcasting['date_time'])
traffic['date_time'] = pd.to_datetime(traffic['date_time'], format="%Y-%m-%d %H:%M:%S") 

traffic['date'] = pd.to_datetime(traffic['date_time']).dt.date
traffic['time'] = pd.to_datetime(traffic['date_time']).dt.time
broadcasting['date'] = pd.to_datetime(broadcasting['date_time']).dt.date
broadcasting['time'] = pd.to_datetime(broadcasting['date_time']).dt.time

#traffic_relevant = traffic.loc[(traffic['visit_source'].isin(["direct","search","paid_search"])) & (traffic['country'] == 'Netherlands')]
traffic_relevant = traffic.loc[traffic['country'] == 'Netherlands']


selected_day = datetime.date(2019,6,20)
selected_time_min = datetime.time(18,00,00)
selected_time_max = datetime.time(19,00,00)

data = traffic_relevant.loc[traffic_relevant["date"] == selected_day]
data = data.loc[(data["time"]>selected_time_min) & (data["time"]<selected_time_max)]
data2 = broadcasting.loc[broadcasting["date"] == selected_day]
data2 = data2.loc[(data2["time"]>selected_time_min) & (data2["time"]<selected_time_max)]


plt.hist(data['date_time'], bins=60)
plt.vlines(x=data2['date_time'], ymin=0, ymax=60)
plt.show()

visits_per_min = traffic.groupby('date_time').count().reset_index()
visits_per_min = traffic.groupby('date_time').size().hist()


traffic['campaign_yn'] = traffic[traffic['date'].isin(broadcasting['date'])]

for i in range(0,5):
    print(i)

broadcast['pre_window']
