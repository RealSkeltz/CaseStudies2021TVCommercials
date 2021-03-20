# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Broadcast
broadcast_NL = pd.read_csv("XXX/XXX/broadcast_for_models.csv")

# Traffic
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


################### INSPECTING PEAKS AND PRE-POST EFFECTS #####################
################### in order to determine length of window, cutoff points #####

################################ Original ####################################

# Report: Figure 7 and 8
# Select a certain datetime window
chosen_datetime_low = datetime.datetime(2019,3,9,20,10,0)
chosen_datetime_up = datetime.datetime(2019,3,9,20,25,0)
broadcast_oneday = broadcast_NL.loc[(broadcast_NL['date_time'] >= chosen_datetime_low)
                                    & (broadcast_NL['date_time'] < chosen_datetime_up)
                                    ]

chosen_date_string = chosen_datetime_low.strftime("%A %d-%b-%Y")


# Zoomed in
# Choose a peak and set the (same) date as above and the zoomed in time window
# We show mon 3-11, sat 3-9
zoomed_datetime_low = datetime.datetime(2019,3,9,20,10,0)
zoomed_datetime_up = datetime.datetime(2019,3,9,20,25,0)

plt.plot(traffic_relevant_NL.groupby(traffic_relevant_NL['date_time'])['visits_index'].sum(), linewidth=0.5)
plt.vlines(x=broadcast_oneday['date_time'], ymin=-0.25, ymax=1.4,linewidth=0.5,color='black')


plt.scatter(broadcast_oneday['date_time'],broadcast_oneday['effect_prepost_window_3'],color='blue')
plt.vlines(x=broadcast_oneday['date_time'] + datetime.timedelta(minutes=3),ymin=-0.25, ymax=1.4,linewidth=0.5)
plt.vlines(x=broadcast_oneday['date_time'] - datetime.timedelta(minutes=3),ymin=-0.25, ymax=1.4,linewidth=0.5)

plt.scatter(broadcast_oneday['date_time'],broadcast_oneday['effect_prepost_window_5'],color='red')
plt.vlines(x=broadcast_oneday['date_time'] + datetime.timedelta(minutes=5),ymin=-0.25, ymax=1.4,linewidth=0.5,color='red')
plt.vlines(x=broadcast_oneday['date_time'] - datetime.timedelta(minutes=5),ymin=-0.25, ymax=1.4,linewidth=0.5,color='red')

plt.axhline(linewidth=1,color='black')
plt.xlim(zoomed_datetime_low, zoomed_datetime_up)
plt.ylim(-0.25,1.4)
#plt.title("Sum of visits_index per minute on " + chosen_date_string + " (NL, zoomed)")



############## Relative pre-post effects -> inspect in order to determine cut-off effect! #############

# Select a certain datetime window
chosen_datetime_low = datetime.datetime(2019,3,9,14,40,0)
chosen_datetime_up = datetime.datetime(2019,3,9,14,55,0)
broadcast_oneday = broadcast_NL.loc[(broadcast_NL['date_time'] >= chosen_datetime_low)
                                    & (broadcast_NL['date_time'] < chosen_datetime_up)
                                    ]

chosen_date_string = chosen_datetime_low.strftime("%A %d-%b-%Y")


# Zoomed out
plt.scatter(broadcast_oneday['date_time'],broadcast_oneday['effect_prepost_window_3_perc'])
plt.plot(traffic_relevant_NL.groupby(traffic_relevant_NL['date_time'])['visits_index'].sum(),color='black', linewidth=0.5)
plt.vlines(x=broadcast_oneday['date_time'], ymin=-0.25, ymax=1.4,linewidth=0.5)
plt.axhline(linewidth=1)
plt.xlim(chosen_datetime_low, chosen_datetime_up)
plt.title("Sum of visits_index per minute on " + chosen_date_string + ' (NL)')

# Report: Figure 9
# Zoomed in
# Choose a peak and set the (same) date as above and the zoomed in time window
zoomed_datetime_low = datetime.datetime(2019,3,9,14,40,0)
zoomed_datetime_up = datetime.datetime(2019,3,9,14,55,0)

plt.plot(traffic_relevant_NL.groupby(traffic_relevant_NL['date_time'])['visits_index'].sum(), linewidth=0.5)
plt.vlines(x=broadcast_oneday['date_time'], ymin=-0.25, ymax=1.4,linewidth=0.5,color='black')

plt.scatter(broadcast_oneday['date_time'],broadcast_oneday['effect_prepost_window_3_perc'],color='blue')
plt.vlines(x=broadcast_oneday['date_time'] + datetime.timedelta(minutes=3),ymin=-0.25, ymax=1.4,linewidth=0.5)
plt.vlines(x=broadcast_oneday['date_time'] - datetime.timedelta(minutes=3),ymin=-0.25, ymax=1.4,linewidth=0.5)

plt.scatter(broadcast_oneday['date_time'],broadcast_oneday['effect_prepost_window_5_perc'],color='red')
plt.vlines(x=broadcast_oneday['date_time'] + datetime.timedelta(minutes=5),ymin=-0.25, ymax=1.4,linewidth=0.5,color='red')
plt.vlines(x=broadcast_oneday['date_time'] - datetime.timedelta(minutes=5),ymin=-0.25, ymax=1.4,linewidth=0.5,color='red')

plt.axhline(linewidth=1,color='black')
plt.xlim(zoomed_datetime_low, zoomed_datetime_up)
plt.ylim(-0.25,1.4)
#plt.title("Sum of visits_index per minute on " + chosen_date_string + " (NL, zoomed)")





