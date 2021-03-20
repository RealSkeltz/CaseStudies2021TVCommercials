# Import packages
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Import data
traffic_relevant_NL = pd.read_csv("XXX/XXX/traffic_relevant_NL.csv")
broadcast_NL = pd.read_csv("XXX/XXX/broadcast_relevant_NL.csv")

# Necessary steps
traffic_relevant_NL['date_time'] = pd.to_datetime(traffic_relevant_NL['date_time'], format="%Y-%m-%d %H:%M:%S") 
traffic_relevant_NL['time'] = pd.to_datetime(traffic_relevant_NL['date_time']).dt.time

broadcast_NL['date_time'] = broadcast_NL['date'] +' '+ broadcast_NL['time']
broadcast_NL['date_time'] = pd.to_datetime(broadcast_NL['date_time'])
broadcast_NL['time'] = pd.to_datetime(broadcast_NL['time']).dt.time

# 4 random days which show the consistent peak at 00:00 every night

# Report: Figure 10
plt.subplot(2,2,1)
plt.plot(traffic_relevant_NL.groupby(traffic_relevant_NL['date_time'])['visits_index'].sum(),color='black') 
plt.xlim(datetime.datetime(2019,3,9,23,57,0), datetime.datetime(2019,3,10,0,3,0))
plt.title("2019-03-09 (NL)")

plt.subplot(2,2,2)
plt.plot(traffic_relevant_NL.groupby(traffic_relevant_NL['date_time'])['visits_index'].sum(),color='black') 
plt.xlim(datetime.datetime(2019,4,9,23,57,0), datetime.datetime(2019,4,10,0,3,0))
plt.title("2019-04-09 (NL)")

plt.subplot(2,2,3)
plt.plot(traffic_relevant_NL.groupby(traffic_relevant_NL['date_time'])['visits_index'].sum(),color='black') 
plt.xlim(datetime.datetime(2019,4,23,23,57,0), datetime.datetime(2019,4,24,0,3,0))
plt.title("2019-04-23 (NL)")

plt.subplot(2,2,4)
plt.plot(traffic_relevant_NL.groupby(traffic_relevant_NL['date_time'])['visits_index'].sum(),color='black') 
plt.xlim(datetime.datetime(2019,5,7,23,57,0), datetime.datetime(2019,5,8,0,3,0))
plt.title("2019-04-23 (NL)")


# Pay attention to the ads just before 00h
broadcasts_around_0h = broadcast_NL.loc[broadcast_NL['time'].isin([datetime.time(0,0,0),
                                                                   datetime.time(23,57,0),
                                                                   datetime.time(23,58,0),
                                                                   datetime.time(23,59,0)])]

outlier_ad_dates = broadcasts_around_0h['date_time']

# Appears only two situations slightly  more difficult, separate the situations
outliers_simple = broadcasts_around_0h.loc[(broadcasts_around_0h['date_time'] != datetime.datetime(2019,6,29,23,59,0)) 
                                                   & (broadcasts_around_0h['date_time'] != datetime.datetime(2019,3,2,0,0,0))]
outliers_simple = outliers_simple.sort_values(by='date_time')

outliers_notsimple = broadcasts_around_0h.loc[(broadcasts_around_0h['date_time'] == datetime.datetime(2019,6,29,23,59,0)) 
                                                   | (broadcasts_around_0h['date_time'] == datetime.datetime(2019,3,2,0,0,0))]
outliers_notsimple = outliers_notsimple.sort_values(by='date_time')


# Plot the situations with ad just before 00:00, but that do not cause issues
i = 1
plt.figure(figsize=(50,10))
for date_time in outliers_simple['date_time']:
    begin = pd.to_datetime(date_time - datetime.timedelta(minutes=5))
    end = pd.to_datetime(date_time + datetime.timedelta(minutes=5))
    
    date_time_plus_day = date_time + datetime.timedelta(days=1)
    begin = str(date_time)[:11] + '23:55:00'
    end = str(date_time_plus_day)[:11] + '00:05:00'
    datetime00h = str(date_time_plus_day)[:11] + '00:00:00'
    
    plt.subplot(1,5,i)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2)
    plt.plot(traffic_relevant_NL.groupby(by='date_time')['visits_index'].sum(),color='black',linewidth=2)
    plt.vlines(x=date_time,ymin=0,ymax=1,linewidth=2)
    plt.vlines(x=datetime00h,ymin=0,ymax=1,color='red',linewidth=2)
    plt.xlim(begin, end)
    plt.xticks([])
    plt.yticks([0,0.5,1],fontsize=30)
    plt.ylim(0,1)
    plt.title(str(date_time),fontsize=30)
    i = i+1

# Plot the two situations that require some extra attention
i = 1
for date_time in outliers_notsimple['date_time']:
    begin = pd.to_datetime(date_time - datetime.timedelta(minutes=5))
    end = pd.to_datetime(date_time + datetime.timedelta(minutes=5))
    
    date_time_plus_day = date_time + datetime.timedelta(days=1)
    begin = str(date_time)[:11] + '23:55:00'
    end = str(date_time_plus_day)[:11] + '00:05:00'
    datetime00h = str(date_time_plus_day)[:11] + '00:00:00'
    
    plt.figure()
    plt.plot(traffic_relevant_NL.groupby(by='date_time')['visits_index'].sum(),color='black',linewidth=0.5)
    plt.vlines(x=date_time,ymin=0,ymax=1)
    plt.vlines(x=datetime00h,ymin=0,ymax=1,color='red',linewidth=0.5)
    plt.xlim(begin, end)
    #plt.xticks([])
    plt.yticks([0,0.5,1],fontsize=12)
    plt.ylim(0,1)
    plt.title(str(date_time),fontsize=12)
    i = i+1
