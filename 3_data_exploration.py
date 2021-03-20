# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# IMPORT
# Raw data
broadcast = pd.read_csv("XXX/XXX/broadcast_incl_feat.csv")
traffic = pd.read_csv("XXX/XXX/traffic_data")

# Filtered data
broadcast_NL = pd.read_csv("XXX/XXX/broadcast_relevant_NL.csv")
traffic_relevant_NL = pd.read_csv("XXX/XXX/traffic_relevant_NL.csv")

# CONVERT datetime objects - seems to be necessary to do that again
traffic_relevant_NL['date_time'] = pd.to_datetime(traffic_relevant_NL['date_time'], format="%Y-%m-%d %H:%M:%S") 

broadcast_NL['date_time'] = broadcast_NL['date'] +' '+ broadcast_NL['time']
broadcast_NL['date_time'] = pd.to_datetime(broadcast_NL['date_time'])

broadcast_NL['date'] = pd.to_datetime(broadcast_NL['date']).dt.date
broadcast_NL['hour'] = pd.to_datetime(broadcast_NL['date_time']).dt.hour
broadcast_NL['time'] = pd.to_datetime(broadcast_NL['time']).dt.time

traffic_relevant_NL['date'] = pd.to_datetime(traffic_relevant_NL['date_time']).dt.date
traffic_relevant_NL['hour'] = pd.to_datetime(traffic_relevant_NL['date_time']).dt.hour
traffic_relevant_NL['time'] = pd.to_datetime(traffic_relevant_NL['date_time']).dt.time
traffic_relevant_NL['day_of_week'] = pd.to_datetime(traffic_relevant_NL['date_time']).dt.dayofweek

###############################################################################
###############################################################################
####### Data exploration ######################################################
###############################################################################
###############################################################################

# TRAFFIC DATA 

# Report: Table 13 and 14  
# Raw data description (for appendix)
traffic.describe()
variables = ['visit_source','country','bounces','page_category','medium']
for variable in variables:
    print(traffic[variable].value_counts(dropna=False)) #number of rows
    print(round(100*traffic[variable].value_counts(dropna=False)/len(traffic),2),'%') #proportion of rows

# Report: Table 1 and 2
# Preprocessed data exploration
traffic_relevant_NL.describe()
sum_vi_relevant_NL = traffic_relevant_NL['visits_index'].sum()
variables = ['visit_source', 'page_category', 'medium']
for variable in variables:
    print("Sum of visit_index and proportion of sum of visit_index")
    print(round(100*traffic_relevant_NL.groupby(by=variable)['visits_index'].sum()/sum_vi_relevant_NL,2)) #percentage of visit index
    

# Report: Figure 2
# Total visits per day
plt.plot(traffic_relevant_NL.groupby(traffic_relevant_NL['date'])['visits_index'].sum()) 
plt.xlim(datetime.date(2019,1,1), datetime.date(2019,7,1))
plt.ylim(0, 600)
plt.title("Sum visits_index per day (NL)")

# Report: Figure 3
# Average total visits per hour of the day
new = traffic_relevant_NL.groupby([traffic_relevant_NL['hour'], traffic_relevant_NL['date']])['visits_index'].sum().reset_index() #NEW version
mu = new.groupby(new['hour']).mean()
sd = new.groupby(new['hour']).std()
minus = np.squeeze(mu - sd)
plus = np.squeeze(mu + sd)

x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
plt.xlim(0,23)
x_ticks = [0, 3.8, 7.7, 11.5, 15.4, 19.2, 23]
x_hour = ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00', '24:00']
plt.xticks(x_ticks, x_hour)
plt.plot(mu)
plt.fill_between(x, minus, plus, alpha=0.3)
plt.title("Average and std of (sum visits_index) per hour of the day (NL)")

# BROADCAST DATA

# Report: Table 15 and 16
# Raw data description
broadcast.describe()
variables2 = ['operator', 'channel', 'length_of_spot',
              'program_category_before', 'program_category_after',
              'product_category', 'country','position_in_break_new']
for variable in variables2:
    print(broadcast[variable].value_counts(dropna=False))

# Report: Table 3 and 4
# Preprocessed data exploration
broadcast_NL.describe()
variables2 = ['operator', 'channel', 'length_of_spot',
              'program_category_before', 'program_category_after',
              'product_category', 'country','position_in_break']
for variable in variables2:
    print(round(100*broadcast_NL[variable].value_counts(dropna=False)/len(broadcast_NL),2))
    
# Report: Figure 4    
# Total number of broadcasting campaigns per day
plt.hist(broadcast_NL['date'],bins=181)
plt.xlim(datetime.date(2019,1,1), datetime.date(2019,7,1))
plt.title("Histogram of number of broadcasting campaigns per day (NL)")

# Report: Figure 5
# Average total number of broadcasting campaigns per hour of the day
new = broadcast_NL.groupby([broadcast_NL['hour'], broadcast_NL['date']]).size().reset_index()
mu = new.groupby(new['hour']).mean()
sd = new.groupby(new['hour']).std()
minus = np.squeeze(mu - sd)
plus = np.squeeze(mu + sd)

x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
plt.xlim(0,23)
x_ticks = [0, 3.8, 7.7, 11.5, 15.4, 19.2, 23]
x_hour = ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00', '24:00']
plt.xticks(x_ticks, x_hour)
plt.plot(mu.iloc[:,1])
plt.title("Average total number of broadcast commercials per hour of the day (NL)")

# Report: Figure 6
# Combined plot of sum visits_index and broadcasts on specific time frame
broadcast_oneday = broadcast_NL[broadcast_NL['date'] == datetime.date(2019,3,9)]
plt.vlines(x=broadcast_oneday['date_time'], ymin=0, ymax=1.4, color='black',linewidth=1)
plt.plot(traffic_relevant_NL.groupby(traffic_relevant_NL['date_time'])['visits_index'].sum()) 
plt.xlim(datetime.datetime(2019,3,9,20,0,0), datetime.datetime(2019,3,9,22,0,0))
plt.ylim(0,1.4)
plt.title("Sum of visits_index per minute on 2019-03-09 (NL)")
