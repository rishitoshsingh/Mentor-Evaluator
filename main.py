from tensorflow.python.training.tracking import util
import pre_trained_model
target_shape = (224,224,3)
model = pre_trained_model.get_mobilenet()
DETECTION_THRESHOLD = 93.
FPM = 10

import json
with open('configs/motion_location.json','r') as file:
    content_config = json.load(file)
content_config = content_config[0]

import os
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import distance
import utils

def _get_features(X):
    if X.ndim == 4:
        return model.predict(X)
    else:
        return model.predict(np.expand_dims(X,0))

def load_slides(files):
    images = np.zeros((len(files),)+target_shape)
    org_images = np.zeros((len(files),720,1280,3))
    for i, file in enumerate(files):
        org_image = cv2.imread(file)
        org_image = cv2.resize(org_image, (1280, 720))
        org_images[i] = org_image
        image = org_image[content_config['corner_1'][1]:content_config['corner_2'][1],content_config['corner_1'][0]:content_config['corner_2'][0]]
        image = cv2.resize(image, target_shape[:-1])
        images[i] = image
    return org_images, images

slides_path = 'slides/'
slide_files = sorted(os.listdir(slides_path), key=lambda x: int(x.split('.')[0][4:]))
slide_files = [os.path.join(slides_path, x) for x in slide_files]

org_slides, slides = load_slides(slide_files)
slides_feature = _get_features(slides)

video = 'videos/2.mp4'
cap = cv2.VideoCapture(video)

def _get_frame_skip(video, fpm):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * 60 / fpm)
    return frame_skip
FRAME_SKIP = _get_frame_skip(video, FPM)

current_slide = 1
triggered = False
old_frame = None
frame_i = 0
data = {}
data['slide'] = []
data['content'] = []

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
        
        cos_d = distance.cosine(slides_feature[current_slide], current_frame_feature)
        cos_d = (1 - cos_d)*100
        
        org_frame = cv2.putText(org_frame, str(int(cos_d)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow('Video', org_frame)
        cv2.imshow('Slide',org_slides[current_slide])
    
        if not triggered and cos_d >= DETECTION_THRESHOLD:
            triggered = True
            print('Triggered')
        
        elif triggered and cos_d < DETECTION_THRESHOLD:
            triggered = False
            current_slide += 1
            print('Triggered End')
            content = utils.ocr(old_frame)
            data['slide'].append(current_slide)
            data['content'].append(content)
        
        old_frame = org_frame
        
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        break

data_df = pd.DataFrame(data)
data_df.to_csv('OCR.csv')