import json

with open('configs/motion_location.json','r') as file:
    content_config = json.load(file)

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import utils

target_shape = (224,224,3)

slides_path = 'slides/'
slide_files = sorted(os.listdir(slides_path), key=lambda x: int(x.split('.')[0][4:]))
slide_files = [os.path.join(slides_path, x) for x in slide_files]

org_slides, slides = utils.load_slides(slide_files, target_shape, content_config)

def compare_time(t1, t2):
    if t1 is np.nan or t2 is np.nan:
        return False
    try:
        t1 = datetime.strptime(t1,"%H:%M:%S")
        t2 = datetime.strptime(t2,"%H:%M:%S")
    except:
        return False
    return t1 >= t2


def _merge_times(slides, slide_time, code_time, include_code_image=True):
    start_i = 0
    inserted = False
    for c_t in code_time:
        for i in range(start_i, len(slide_time)):
            if compare_time(slide_time[i], c_t):
                slide_time.insert(i, c_t)
                if include_code_image and slides[i] != 'templates/colab/colab.jpg':
                    slides.insert(i, 'templates/colab/colab.jpg')
                start_i = i
                inserted = True
                break
        
        if inserted == False:
            slide_time.append(c_t)
            if include_code_image and slides[-1] != 'templates/colab/colab.jpg':
                slides.append('templates/colab/colab.jpg')
        else: 
            inserted = False
    return slides, slide_time

import joblib
end_timestamps = joblib.load('end_timestamps.pkl')
start_timestamps = ['0:0:0'] + end_timestamps[:-1] 
# code_start = ['1:0:2', '1:4:37']
# code_end = ['1:4:39','1:13:20']
code_start = ['0:29:00', '0:41:00']
code_end = ['0:31:00','0:43:00']

slide_files, start_timestamps = _merge_times(slide_files, start_timestamps, code_start)
slide_files, end_timestamps = _merge_times(slide_files, end_timestamps, code_end,include_code_image=False)

print('Length')
print(len(slide_files))
print(len(end_timestamps))
print(len(start_timestamps))

data = {}
data['Slide'] = list(range(0, len(slide_files)))
data['Image'] = [' ']*len(slide_files)
data['End Time'] = end_timestamps
data['Start Time'] = start_timestamps 
data_df = pd.DataFrame(data)

start_time = pd.to_datetime(data_df['Start Time'], format='%H:%M:%S')
end_time = pd.to_datetime(data_df['End Time'], format='%H:%M:%S')

def get_duration(start, end):
    duration = [np.nan]*len(end)
    for i, end_time in enumerate(end):
        if end_time is pd.NaT:
            continue
        start_i = i
        while start[start_i] is pd.NaT and start_i>=0:
            start_i -= 1
        if i == 46:
            print(start_i)
        if start[start_i] is not pd.NaT:
            duration[i] = (end_time - start[start_i]).total_seconds()
    return duration
data_df['Duration (s)'] = get_duration(start_time, end_time)

start_time = '11:17:39'
start_time = datetime.strptime(start_time, '%H:%M:%S')
from collections import OrderedDict
import re
transcript = OrderedDict()

with open('transcripts/apple.txt', 'r') as file:
    for line in file:
        time_match = re.search('[0-9]{2}:[0-9]{2}:[0-9]{2}', line)
        current_time = datetime.strptime(line[time_match.start():time_match.end()], '%H:%M:%S')
        delta = current_time - start_time
        delta = utils.get_time(delta.total_seconds())
        transcript[delta] = line[time_match.end()+1:].strip()

data_df['Captions'] = ['']*len(data_df['Slide'])



current_slide = 0
end_triggered = False
for cc_time in transcript.keys():
    if utils.compare_time(data_df['End Time'][current_slide], cc_time):
        data_df['Captions'][current_slide] = data_df['Captions'][current_slide] + transcript[cc_time] + ' '
    elif end_triggered:
        data_df['Captions'][current_slide] = data_df['Captions'][current_slide] + transcript[cc_time] + ' '
    else:
        while True:
            current_slide += 1
            if current_slide == len(data_df['End Time'])-1:
                end_triggered = True
                break
            elif data_df['End Time'][current_slide] is not np.nan and utils.compare_time(data_df['End Time'][current_slide], cc_time):
                break
        data_df['Captions'][current_slide] = data_df['Captions'][current_slide] + ' ' + transcript[cc_time]

data_df['Total Words'] = data_df['Captions'].apply(lambda x: len(x))

wpm = [np.nan] * len(data_df['Captions'])
for i, (duration, cc) in enumerate(zip(data_df['Duration (s)'], data_df['Captions'])):
    if cc is np.nan:
        continue
    text = cc.split()
    wpm[i] = len(text) * 60 / duration
data_df['Words per Minute (wpm)'] = wpm

data_df = data_df[['Slide','Image','Start Time','End Time','Duration (s)', 'Total Words', 'Words per Minute (wpm)', 'Captions']]

excel_name = 'timestamps-apple.xlsx'
utils.to_excel(excel_name, data_df, slide_files)


