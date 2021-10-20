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
FPM = 30
WINDOW_SEARCH_SIZE = 6

slides_path = 'slides/'
slide_files = sorted(os.listdir(slides_path), key=lambda x: int(x.split('.')[0][4:]))
slide_files = [os.path.join(slides_path, x) for x in slide_files]

org_slides, slides = utils.load_slides(slide_files, target_shape, content_config)
slides_feature = _get_features(slides)

video = 'videos/3.mp4'
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
FRAME_SKIP = utils.get_frame_skip(video, FPM)

current_slide = 0
triggered = False
old_frame = None
frame_i = 0

end_timestamps = ['']*len(slide_files)
start_time = datetime.fromtimestamp(0)
def get_time(seconds):
    hour = seconds // 3600
    seconds = seconds - hour*3600
    minute = seconds // 60
    seconds = seconds - minute*60
    return "{}:{}:{}".format(int(hour), int(minute), int(seconds))

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
        
        org_frame = cv2.putText(org_frame, str(int(cos_d)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow('Video', org_frame)
        cv2.imshow('Slide', org_slides[current_slide])
    
        if current_slide < current_slide + sim_index and cos_d >= DETECTION_THRESHOLD:
            current_slide = current_slide + sim_index
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            seconds = timedelta(milliseconds=int(timestamp)).total_seconds()
            timestamp = get_time(seconds)
            end_timestamps[current_slide-1] = timestamp
            print('Topic End Triggered - ',end_timestamps[current_slide-1])
        
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        break

data = {}
data['slide_no'] = list(range(0, len(slide_files)))
data['slide'] = [' ']*len(slide_files)
data['end_time'] = end_timestamps
data_df = pd.DataFrame(data)
excel_name = 'timestamps.xlsx'
utils.to_excel(excel_name, data_df, slide_files)
