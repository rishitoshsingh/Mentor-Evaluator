import json

from tensorflow.python.training.tracking import util
with open('configs/motion_location.json','r') as file:
    content_config = json.load(file)
content_config = content_config[0]

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import distance
import utils

import pre_trained_model
target_shape = (224,224,3)
model = pre_trained_model.get_mobilenet()
def _get_features(X):
    if X.ndim == 4:
        return model.predict(X)
    else:
        return model.predict(np.expand_dims(X,0))

DETECTION_THRESHOLD = 85.
FPM = 60
WINDOW_SEARCH_SIZE = 6

slides_path = 'slides/'
slide_files = sorted(os.listdir(slides_path), key=lambda x: int(x.split('.')[0][4:]))
slide_files = [os.path.join(slides_path, x) for x in slide_files]

org_slides, slides = utils.load_slides(slide_files, target_shape, content_config)
slides_feature = _get_features(slides)

video = 'videos/1.mp4'
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
FRAME_SKIP = utils.get_frame_skip(video, FPM)

current_slide = 0
triggered = False
old_frame = None
frame_i = 0

end_timestamps = [np.nan]*len(slide_files)
start_time = datetime.fromtimestamp(0)

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
total_seconds = int(frames / fps)
end_time = utils.get_time(total_seconds)

while True:
    grabbed, org_frame = cap.read()
    frame_i += 1
    if not grabbed:
        cap.release()
        break
    
    if frame_i == 1 or frame_i % FRAME_SKIP == 0:
        frame = org_frame[content_config['corner_1'][1]:content_config['corner_2'][1],content_config['corner_1'][0]:content_config['corner_2'][0]]
        frame = cv2.resize(frame, target_shape[:-1])
        current_frame_feature = _get_features(frame)
        
        cos_d, sim_index = utils.get_cosine_score(current_frame_feature, slides_feature[current_slide: current_slide + WINDOW_SEARCH_SIZE])
        cos_d *= 100
        
        # org_frame = cv2.putText(org_frame, str(int(cos_d)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        # cv2.imshow('Video', org_frame)
        # cv2.imshow('Slide', org_slides[current_slide])
    
        if current_slide < current_slide + sim_index and cos_d >= DETECTION_THRESHOLD:
            current_slide = current_slide + sim_index
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            seconds = timedelta(milliseconds=int(timestamp)).total_seconds()
            timestamp = utils.get_time(seconds)
            end_timestamps[current_slide-1] = timestamp
            print('Slide {} - {}'.format(current_slide-1, end_timestamps[current_slide-1]))
        
    # if cv2.waitKey(1) == ord('q'):
    #     cap.release()
    #     break

end_timestamps[-1] = end_time
data = {}
data['Slide'] = list(range(0, len(slide_files)))
data['Image'] = [' ']*len(slide_files)
data['End Time'] = end_timestamps
data_df = pd.DataFrame(data)

start_timestamps = ['0:0:0'] + end_timestamps[:-1] 
data_df['Start Time'] = start_timestamps
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

def compare_time(t1, t2):
    if t1 is np.nan or t2 is np.nan:
        return False
    t1 = datetime.strptime(t1,"%H:%M:%S")
    t2 = datetime.strptime(t2,"%H:%M:%S")
    return t1 >= t2

current_slide = 0
end_triggered = False
for cc_time in transcript.keys():
    if compare_time(data_df['End Time'][current_slide], cc_time):
        data_df['Captions'][current_slide] = data_df['Captions'][current_slide] + transcript[cc_time] + ' '
    elif end_triggered:
        data_df['Captions'][current_slide] = data_df['Captions'][current_slide] + transcript[cc_time] + ' '
    else:
        while True:
            current_slide += 1
            if current_slide == len(data_df['End Time'])-1:
                end_triggered = True
                break
            elif data_df['End Time'][current_slide] is not np.nan and compare_time(data_df['End Time'][current_slide], cc_time):
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
