import pytesseract
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import xlsxwriter

def ocr(image_mat):
    image_mat = cv2.cvtColor(image_mat, cv2.COLOR_BGR2RGB)
    content = pytesseract.image_to_string(image_mat)
    return content

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

def to_excel(excel_file, data_df, images):
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    data_df.to_excel(writer, sheet_name='Sheet1', index=False)

    workbook  = writer.book
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