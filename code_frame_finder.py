import json
from re import I

with open('configs/colab_h_left_banner_location.json','r') as file:
    banner_config = json.load(file)

import os
from datetime import datetime, timedelta
import numpy as np
import cv2
import utils

template_locations = [
    'templates/colab/chrome_refresh.png',
    'templates/colab/colab_code_snippets.png',
    'templates/colab/colab_commands.png',
    'templates/colab/colab_files.png',
    'templates/colab/colab_find.png',
    'templates/colab/colab_sections.png',
    'templates/colab/colab_terminal.png'
]

templates = {}
for loc in template_locations:
    icon = cv2.imread(loc)
    icon = cv2.resize(icon, (32,32))
    icon = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    icon = cv2.Canny(icon, 50, 200)
    (tH, tW) = icon.shape[:2]
    templates.update({loc: {'mat':icon, 'shape': (tH, tW)}})
    print(loc, (tH, tW))
    # cv2.imshow(loc, icon)

input()
cv2.destroyAllWindows()

FPM = 60
video = 'videos/1.mp4'
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
FRAME_SKIP = utils.get_frame_skip(video, FPM)

current_slide = 0
triggered = False
old_frame = None
frame_i = 0


while True:
    grabbed, frame = cap.read()
    
    if not grabbed:
        cap.release()
        break
    crop_frame = frame[banner_config['corner_1'][1]:banner_config['corner_2'][1],banner_config['corner_1'][0]:banner_config['corner_2'][0]]
    frame_bw = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    print('frame: ', frame_bw.shape)
    results = []
    for loc in templates.keys():
        result = cv2.matchTemplate(templates[loc]['mat'], frame_bw, cv2.TM_CCOEFF_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        print(minVal, maxVal)   
    