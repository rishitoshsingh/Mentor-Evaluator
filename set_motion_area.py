import cv2
from imutils.video import VideoStream
import numpy as np
import json

from threading import Thread

UPDATE_NUM = 10
RESOLUTION = (1280, 720)

NEW_MOTION_BOX = False

motion_boxes = []
def get_new_motion_box(corner_1, corner_2):
    motion_box = {
        'corner_1': corner_1,
        'corner_2': corner_2
    }
    return motion_box
motion_box = get_new_motion_box(corner_1=[0,0], corner_2=list(RESOLUTION))
motion_boxes.append(motion_box)

video_file = 'videos/How Apple Siri Voice Recognition works - 15 May 2021-8s7zQbnquz0.mp4'

def showRecordingVideo(motion_boxes):
    cap = VideoStream(video_file).start()
    try:
        frame = cap.read()
        frame = cv2.resize(frame, RESOLUTION)
        (height, width) = frame.shape[:2]

        while True:
            frame = cap.read()
            frame = cv2.resize(frame, RESOLUTION)
            
            for motion_box in motion_boxes:
                frame = cv2.rectangle(frame, tuple(motion_box['corner_1']), tuple(motion_box['corner_2']), (225,0,0),2)
            
            cv2.imshow('Recording', frame)
            key = cv2.waitKey(1)
            if key == ord('w'):
                motion_boxes[-1]['corner_1'][1] = (motion_boxes[-1]['corner_1'][1] - UPDATE_NUM) % height
            elif key == ord('a'):
                motion_boxes[-1]['corner_1'][0] = (motion_boxes[-1]['corner_1'][0] - UPDATE_NUM) % width
            elif key == ord('s'):
                motion_boxes[-1]['corner_1'][1] = (motion_boxes[-1]['corner_1'][1] + UPDATE_NUM) % height
            elif key == ord('d'):
                motion_boxes[-1]['corner_1'][0] = (motion_boxes[-1]['corner_1'][0] + UPDATE_NUM) % width
            
            elif key == ord('i'):
                motion_boxes[-1]['corner_2'][1] = (motion_boxes[-1]['corner_2'][1] - UPDATE_NUM) % height
            elif key == ord('j'):
                motion_boxes[-1]['corner_2'][0] = (motion_boxes[-1]['corner_2'][0] - UPDATE_NUM) % width
            elif key == ord('k'):
                motion_boxes[-1]['corner_2'][1] = (motion_boxes[-1]['corner_2'][1] + UPDATE_NUM) % height
            elif key == ord('l'):
                motion_boxes[-1]['corner_2'][0] = (motion_boxes[-1]['corner_2'][0] + UPDATE_NUM) % width
            elif key == 13:
                motion_box = get_new_motion_box(motion_box['corner_1'].copy(), motion_box['corner_2'].copy())
                motion_boxes.append(motion_box)
            elif key == 27:
                break
            
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        cv2.destroyAllWindows()
    return motion_boxes[:-1]

motion_boxes = showRecordingVideo(motion_boxes)
print(motion_boxes)
print(len(motion_boxes))

with open('configs/motion_location.json','w') as file:
    json.dump(motion_boxes, file, indent=4)
    
# with open('configs/video_play_button_location.json','w') as file:
#     json.dump(motion_boxes, file, indent=4)
    
# with open('configs/colab_logo_location.json','w') as file:
#     json.dump(motion_boxes, file, indent=4)
    
