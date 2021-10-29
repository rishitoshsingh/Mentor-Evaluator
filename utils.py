from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_slides(files, target_shape, content_config):
    images = np.zeros((len(files),)+target_shape)
    # org_images = np.zeros((len(files),720,1280,3))
    org_images = np.zeros((len(files),72,128,3))
    for i, file in enumerate(files):
        org_image = cv2.imread(file)
        org_image = cv2.resize(org_image, (1280, 720))
        image = org_image[content_config['corner_1'][1]:content_config['corner_2'][1],content_config['corner_1'][0]:content_config['corner_2'][0]]
        image = cv2.resize(image, target_shape[:-1])
        org_image = cv2.resize(org_image, (128, 72))
        org_images[i] = org_image
        images[i] = image
    return org_images, images

def get_frame_skip(video, fpm):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * 60 / fpm)
    return frame_skip

def get_time(seconds):
    hour = seconds // 3600
    seconds = seconds - hour*3600
    minute = seconds // 60
    seconds = seconds - minute*60
    return "{}:{}:{}".format(int(hour), int(minute), int(seconds))

def compare_time(t1, t2):
    if t1 is np.nan or t2 is np.nan:
        return False
    t1 = datetime.strptime(t1,"%H:%M:%S")
    t2 = datetime.strptime(t2,"%H:%M:%S")
    return t1 >= t2

def get_cosine_score(X, y):
    ravel = False
    if X.ndim == 1:
        ravel = True
        X = np.expand_dims(X,0)
    if y.ndim == 1:
        ravel = True
        y = np.expand_dims(y,0)
    score = cosine_similarity(X,y)
    if ravel:
        score = score.ravel()
    return np.max(score), np.argmax(score)

def _get_middle_values(start, end, n):
    start = datetime.strptime(start, "%H:%M:%S")
    end = datetime.strptime(end, "%H:%M:%S")
    diff = end - start
    req_diff = diff / (n+1)
    times = []
    for i in range(n):
        times.append((start + req_diff*(i+1)).strftime("%H:%M:%S"))
    return times

def fill_null_time(series):
    gap_found = False
    start_i, end_i = None, None
    for i in range(len(series)):
        if series[i] != ' ': #not np.isnan(series[i]):
            if gap_found == True:
                end_i = i
                values = _get_middle_values(series[start_i], series[end_i], (end_i-start_i-1))
                for temp_i, value in zip(range(start_i+1,end_i), values):
                    series[temp_i] = value
                print(series[start_i:end_i])
                gap_found = False
            else:
                continue
        else:
            if gap_found == False:
                gap_found = True
                start_i = i-1    
    return series

def to_excel(excel_file, data_df, images):
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    data_df.to_excel(writer, sheet_name='Sheet1', index=False)

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    worksheet.set_column('B:B',13)
    worksheet.set_default_row(40)
    
    row, col = (1,1)
    args = {
        'x_scale':0.05,
        'y_scale':0.05,
        'object_position': 2
    }
    for image_path in images:
        worksheet.insert_image(row, col, image_path, args)
        row += 1

    writer.save()