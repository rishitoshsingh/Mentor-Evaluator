import pytesseract
import cv2

def ocr(image_mat):
    image_mat = cv2.cvtColor(image_mat, cv2.COLOR_BGR2RGB)
    content = pytesseract.image_to_string(image_mat)
    return content