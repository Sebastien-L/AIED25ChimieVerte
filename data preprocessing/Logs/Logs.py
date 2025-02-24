"""
(2023-04-04 by XX)

"""

import pandas as pd
import datetime
import numpy as np
import os
from datetime import datetime
import time

def to_human_time(mytime):
    #
    return datetime.fromtimestamp((int(mytime))/1000000).time() 
	
#Read CSV files
tdata = pd.read_csv('Tobii MouseKeyboard Events.csv')
pdata = pd.read_csv('panopto_log_filtered.csv')

timedata = pd.read_csv('Participants_timestamps_2024.csv') 
uids = [i for i in range(1,15)] + [i for i in range(16,30)]

#timedata = pd.read_csv('Participants_timestamps_2023.csv')
#uids = [i for i in range(1,26)]

#Initialize variables
pid = 0
state = "pause"
pevent_idx = 0
play_time = -1 


for i in uids:
    pid = 'P24_'+str(i) if i > 9 else 'P24_0'+str(i)
    #pid = 'P23_'+str(i) if i > 9 else 'P23_0'+str(i)
    
    tdata_temp = tdata[tdata.ParticipantID == pid]
    pdata_temp = pdata[pdata.ParticipantID == pid]
    
    tdata_temp = tdata_temp.sort_values(by=['Recording timestamp'])
    pdata_temp = pdata_temp.sort_values(by=['Index'])
    
    pevent_idx = pdata_temp.index[0]
    play_time = 0
    last_action = ""
    state = "pause"
    
    #Iterate over target data
    for _, tevent in tdata_temp.iterrows():
        action = tevent['Action']

        if action in ['pause', 'play']:
            if state == "play":
                #here the video was paying and is now paused, so we write the start and end time of the previous Panopto segment
                #and we update the state to "pause"
                pdata_temp.loc[pevent_idx, 'starttobii'] = play_time
                pdata_temp.loc[pevent_idx, 'endtobii'] = tevent['Recording timestamp']
                pdata_temp.loc[pevent_idx, 'ActionStart'] = last_action
                pdata_temp.loc[pevent_idx, 'ActionEnd'] = tevent['Action']
                
                pdata_temp.loc[pevent_idx, 'starttobii_HMS'] = to_human_time(play_time) 
                pdata_temp.loc[pevent_idx, 'endtobii_HMS'] = to_human_time(tevent['Recording timestamp'])
                
                pevent_idx += 1
                state = "pause"
            else:
                #here
                play_time = tevent['Recording timestamp']
                state = "play"
                last_action = tevent['Action']
        elif action in ['ff', 'fb']:
            if state == "play":
                #here
                pdata_temp.loc[pevent_idx, 'starttobii'] = play_time
                pdata_temp.loc[pevent_idx, 'endtobii'] = tevent['Recording timestamp']
                pdata_temp.loc[pevent_idx, 'ActionStart'] = last_action
                pdata_temp.loc[pevent_idx, 'ActionEnd'] = tevent['Action']
                
                pdata_temp.loc[pevent_idx, 'starttobii_HMS'] = to_human_time(play_time) 
                pdata_temp.loc[pevent_idx, 'endtobii_HMS'] = to_human_time(tevent['Recording timestamp'])
                
                pevent_idx += 1
                play_time = tevent['Recording timestamp']
                last_action = tevent['Action']
            else:
                #here nothing to do because fb/ff are ignored by Panopto when the video is paused
                continue

    #Set the end time for the last event (because there is no Stop action in the tobii logs)
    temp_end = timedata.loc[timedata.ParticipantID == pid]["Task end"].values[0]
    temp_end = datetime.strptime(temp_end, "%H:%M:%S.%f").timetuple()
    temp_end = (temp_end.tm_min*60 + temp_end.tm_sec)*1000000
    
    pdata_temp.loc[pevent_idx, 'starttobii'] = play_time
    pdata_temp.loc[pevent_idx,'endtobii'] = temp_end
    pdata_temp.loc[pevent_idx, 'ActionStart'] = last_action
    pdata_temp.loc[pevent_idx, 'ActionEnd'] = tevent['Action']
    
    pdata_temp.loc[pevent_idx, 'starttobii_HMS'] = to_human_time(play_time) 
    pdata_temp.loc[pevent_idx, 'endtobii_HMS'] = to_human_time(temp_end)
    
    print (pdata_temp)
    pdata_temp.to_csv(r'new panopto/panopto_data_updated_'+pid+'.csv', index=False)
    
    #break 